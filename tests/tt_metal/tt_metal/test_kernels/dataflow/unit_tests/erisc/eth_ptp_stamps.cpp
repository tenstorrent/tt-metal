// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ActiveEthPtpStamps' kernel: one end of a link's stamped frame exchange, as initiator or echo. The initiator then
// sends a run of two-step stamped frames.

#include <cstddef>
#include <cstdint>
#include <optional>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "eth_ptp_stamps.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;
using namespace eth_ptp_stamps;

constexpr bool kInitiator = get_compile_time_arg_val(0) != 0;
// A TX queue and header row that neither the firmware (eth_ptp.hpp) nor the fabric routers (queues 0 and 1) use, and
// any TCAM row and label.
constexpr uint32_t kTxq = 2, kHeaderRow = 3, kTcamRow = 63, kLabel = 0x15;
// A destination no firmware frame has (eth_ptp.hpp), and a rule that matches only it.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;
// A 1 in the mask means don't care, so the rule compares only the four 0xA5 bytes.
constexpr eth_ptp::RxTcamNonIpPattern kStampRowValues{.da = {0xA5A5'A500u, 0x0000'00A5u}};
constexpr eth_ptp::RxTcamNonIpPattern kStampRowMask{
    .sa = {~0u, ~0u, ~0u, ~0u},
    .da = {0x0000'00FFu, 0xFFFF'FF00u, ~0u, ~0u},
    .addr_flags = {.augmented_da = 0xF, .augmented_sa = 0xF},
    .ethertype = {.value = 0xFFFF, .augmented = 0xF},
    .priority = {.pcp = 7}};
// The MAC writes the egress stamp into `stamp`.
struct Frame {
    eth_ptp::FrameStampSlot stamp;
    uint32_t key;
    uint32_t echo_key;
};
static_assert(
    eth_ptp::kFrameStampOffsetBytes + eth_ptp::kFrameStampBytes <= offsetof(Frame, key) &&
    sizeof(Frame) <= kFrameBytes);
using StampRule = eth_ptp::RxStampRule<kTcamRow, kLabel>;
constexpr uint32_t kSpins = 1u << 20;
constexpr uint32_t kKeyTag = 0x5A000000u;
constexpr uint32_t kTwoStepTag = 0x7B000000u;

template <typename Pred>
bool wait_for(Pred&& pred) {
    for (uint32_t spin = 0; spin < kSpins; spin++) {
        invalidate_l1_cache();
        if (pred()) {
            return true;
        }
    }
    return false;
}

bool send(volatile tt_l1_ptr Frame* frame) {
    eth_ptp::clear_frame_stamp(frame->stamp);
    const uint32_t addr = reinterpret_cast<uint32_t>(frame);
    internal_::eth_send_packet<false>(kTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
    return wait_for([] { return !internal_::eth_txq_is_busy(kTxq); });
}

// The frame's ingress stamp, taken the way the link end takes a burst's: the FIFO holds exactly its entry. The label
// check proves the rule recorded it.
std::optional<uint64_t> take_ingress() {
    namespace rx_stamp_fifo = eth_ptp::rx_stamp_fifo;
    if (rx_stamp_fifo::holds_exactly<1>()) {
        const eth_ptp::RxStampLabel label = eth_ptp::kRxStampLabel.read();
        if (label.valid && label.label == StampRule::kLabel) {
            return rx_stamp_fifo::pop();
        }
    }
    rx_stamp_fifo::flush();
    return std::nullopt;
}

// Reads the clocks just after a refclk update, when the refclk the cores see is newest.
int32_t restart_error_ns(int64_t ptp_minus_refclk_ns) {
    const eth_ptp::ClocksLo update = eth_ptp::await_refclk_update();
    const uint64_t ptp_ns = eth_ptp::read_ptp_ns();
    return static_cast<int32_t>(
        static_cast<uint32_t>(ptp_ns) -
        static_cast<uint32_t>(update.refclk * eth_ptp::kNsPerRefclkTick + ptp_minus_refclk_ns));
}

void handshake(uint32_t base) {
    if constexpr (kInitiator) {
        eth_send_bytes(base, base, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

// Each frame's FIFO entry is taken before the next frame is armed, so no frame can pick up another's tag.
uint32_t send_two_step(volatile tt_l1_ptr Frame* frame) {
    namespace tx_stamp_fifo = eth_ptp::tx_stamp_fifo;
    tx_stamp_fifo::clear();
    uint32_t good = 0;
    for (uint32_t i = 0; i < kTwoStepFrames; i++) {
        const uint32_t tag = kTwoStepTag | i;
        const uint64_t before_ns = eth_ptp::read_ptp_ns();
        eth_ptp::txq_arm_two_step(kTxq, tag);
        if (!send(frame) || !wait_for([] { return !tx_stamp_fifo::empty(); })) {
            break;
        }
        const std::optional<tx_stamp_fifo::Entry> entry = tx_stamp_fifo::pop();
        const uint64_t after_ns = eth_ptp::read_ptp_ns();
        good += entry && entry->tag == tag && entry->time_ns >= before_ns && entry->time_ns <= after_ns;
    }
    eth_ptp::txq_disarm(kTxq);
    return good;
}

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr Result* result = reinterpret_cast<volatile tt_l1_ptr Result*>(base + kResultOffset);
    volatile tt_l1_ptr Frame* frame = reinterpret_cast<volatile tt_l1_ptr Frame*>(base + kFrameOffset);
    frame->key = 0;
    frame->echo_key = 0;
    result->done = 0;
    result->header_select_before = eth_ptp::word_of(eth_ptp::txq_header_select(kTxq).read());
    result->no_match_before = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    result->restart_error_ns = restart_error_ns(eth_ptp::restart_ptp_timer());
    StampRule rule;
    rule.install(kStampRowValues, kStampRowMask);
    eth_ptp::TxHeaderRow<kTxq, kHeaderRow> header;
    header.install(kStampFrameDa);
    eth_ptp::txq_arm_in_frame(kTxq);

    handshake(base);

    uint32_t unstamped = 0, ingress_mismatched = 0, round = 0;
    for (; round < kRounds; round++) {
        const uint32_t key = kKeyTag | (round + 1);
        if constexpr (kInitiator) {
            frame->key = key;
            frame->echo_key = 0;
            if (!send(frame) || !wait_for([&] { return frame->echo_key == key; })) {
                break;
            }
        } else {
            if (!wait_for([&] { return frame->key == key; })) {
                break;
            }
        }
        const uint64_t peer_egress = eth_ptp::frame_stamp_ns(frame->stamp);
        const std::optional<uint64_t> ingress = take_ingress();
        ingress_mismatched += !ingress;
        unstamped += peer_egress == 0;
        result->stamps[round].peer_egress = peer_egress;
        result->stamps[round].ingress = ingress.value_or(0);
        if constexpr (!kInitiator) {
            frame->echo_key = key;
            if (!send(frame)) {
                break;
            }
        }
    }

    // The echo end's last frame is stamped only while its queue is armed, so it disarms once the initiator holds it.
    handshake(base);
    eth_ptp::txq_disarm(kTxq);
    result->two_step_good = kInitiator && round == kRounds ? send_two_step(frame) : 0;
    header.restore();
    rule.remove();
    result->header_select_after = eth_ptp::word_of(eth_ptp::txq_header_select(kTxq).read());
    result->no_match_after = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    result->unstamped = unstamped;
    result->ingress_mismatched = ingress_mismatched;
    result->rounds = round;
    result->done = kDone;
}
