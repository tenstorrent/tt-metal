// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/erisc_e2h2h2e_pipeline.hpp"

#include <utility>

namespace tt::tt_metal::experimental {

E2H2H2EPipeline::E2H2H2EPipeline(
    std::unique_ptr<EriscH2HSocket> h2h,
    std::unique_ptr<E2HLeg> e2h,
    std::unique_ptr<H2ELeg> h2e,
    uint32_t max_per_poll) :
    h2h_(std::move(h2h)), e2h_(std::move(e2h)), h2e_(std::move(h2e)), max_per_poll_(max_per_poll) {}

E2H2H2EPipeline::~E2H2H2EPipeline() = default;

uint32_t E2H2H2EPipeline::poll(const Inspect& inspect) {
    uint32_t progress = 0;
    if (e2h_) {
        // Hop 1: bridge ring -> H2H queue. A refused submit leaves the frame for the next turn.
        uint32_t taken = 0;
        progress += e2h_->poll([&](const BridgeSendTask& t) {
            if (taken >= max_per_poll_ || !h2h_->submit(t)) {
                return false;
            }
            ++taken;
            return true;
        });
        counters_.forwarded += taken;
        counters_.out_of_order = e2h_->out_of_order();
    }
    // Hops 2-3: landed puts go into the far router; pages the peer credited go back to the ERISC.
    progress += h2h_->poll(
        [&](uint32_t arena, uint32_t pages) {
            if (e2h_) {
                e2h_->retire(arena, pages);
            }
        },
        [&](const BridgeDeliverTask& d) {
            if (!h2e_ || !h2e_->publish(d)) {
                return false;  // no far router here, or it has not kept up: re-offered next turn
            }
            if (inspect) {
                inspect(d);
            }
            h2h_->consumed(d.arena, 1);  // the MMIO copy is done, so the peer may reuse the slot
            ++counters_.injected;
            return true;
        });
    (void)h2h_->publish_credits();

    // HOP 4: fabric's credits home. The far receiver recorded totals in L1 because it has no
    // cable; this side publishes them, the other turns the difference into register writes.
    if (relay_ != nullptr && relay_->rx) {
        tt::tt_fabric::erisc_bridge::CreditSnapshot snap;
        auto& env = *relay_->env;
        if (tt::tt_fabric::erisc_bridge::sample_recv_credits(
                env, relay_->dev, relay_->eth, relay_->recv_credits_addr, relay_->channels, snap)) {
            std::uint64_t comp = 0;
            std::uint64_t ack = 0;
            for (std::uint32_t i = 0; i < relay_->channels; ++i) {
                comp += snap.completions[i];
                ack += snap.acks[i];
            }
            bool published = false;
            if (comp != relay_->sampled_completions) {
                relay_->sampled_completions = comp;
                published = h2h_->publish_word(relay_->arena, tt::tt_fabric::BridgeCreditWord::EdmCompletions, comp);
            }
            if (ack != relay_->sampled_acks) {
                relay_->sampled_acks = ack;
                published = h2h_->publish_word(relay_->arena, tt::tt_fabric::BridgeCreditWord::EdmAcks, ack) || published;
            }
            // One flush for both words: remote completion, which publish_word does not give.
            if (published) {
                (void)h2h_->flush_words();
            }
        }
    }
    if (relay_ != nullptr && relay_->tx) {
        auto& env = *relay_->env;
        std::string err;
        const auto comp = static_cast<std::uint32_t>(
            h2h_->peer_word(relay_->arena, tt::tt_fabric::BridgeCreditWord::EdmCompletions));
        const auto ack =
            static_cast<std::uint32_t>(h2h_->peer_word(relay_->arena, tt::tt_fabric::BridgeCreditWord::EdmAcks));
        if (relay_->use_streams) {
            (void)tt::tt_fabric::erisc_bridge::inject_sender_credits_stream(
                env,
                relay_->dev,
                relay_->eth,
                env.get_hal(),
                relay_->completed_stream,
                relay_->acked_stream,
                comp,
                ack,
                relay_->injected_completions,
                relay_->injected_acks,
                err);
        } else {
            (void)tt::tt_fabric::erisc_bridge::inject_sender_credits(
                env, relay_->dev, relay_->eth, relay_->completions_addr, relay_->acks_addr, comp, ack, err);
            relay_->injected_completions = comp;
            relay_->injected_acks = ack;
        }
    }

    if (h2e_) {
        counters_.landed = h2e_->drained(0);  // also refreshes H2E's credit gate
    }
    return progress;
}

std::string E2H2H2EPipeline::first_error() const {
    if (!h2h_->first_error().empty()) {
        return h2h_->first_error();
    }
    return h2e_ ? h2e_->first_error() : std::string{};
}

}  // namespace tt::tt_metal::experimental
