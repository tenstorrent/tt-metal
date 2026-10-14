// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Carries fabric's own credits across a bridged link. The far receiver records totals in L1
// because it has no cable; this samples them, and on the other host writes them to the sender.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/device.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "tt_metal/distributed/erisc_bridge_doorbell.hpp"

namespace tt::tt_fabric::erisc_bridge {

// Totals, never deltas: the sampler runs whenever the pipeline polls, so a delta it happened to
// miss would be lost. Each side keeps its own running total and takes unsigned differences.
struct CreditSnapshot {
    std::vector<std::uint32_t> completions;  // one per sender channel
    std::vector<std::uint32_t> acks;
};

// One bridged link's relay. A rank sets whichever side it owns; the pipeline drives it each poll.
// SINGLE SENDER CHANNEL today: the totals are summed, so two bridged senders on one link would
// have their credits merged and returned to the wrong one.
struct CreditRelay {
    bool rx = false;  // owns the far receiver: samples its L1 totals and publishes them
    bool tx = false;  // owns the bridged sender: injects what the peer published
    // The mesh's own env, not the singleton: handed in because the caller already holds it.
    tt::tt_metal::MetalEnvImpl* env = nullptr;
    tt::tt_metal::IDevice* dev = nullptr;
    tt::tt_metal::CoreCoord eth{};
    std::uint32_t channels = 0;
    std::uint32_t arena = 0;
    std::uint32_t recv_credits_addr = 0;  // rx
    // TX, one of two forms. Counters: write the running total into L1. Registers: write what is
    // new into a stream register that accumulates. Which one is fabric's choice, not ours.
    bool use_streams = false;
    std::uint32_t completions_addr = 0;  // counters: to_sender_channel_remote_completion_counters
    std::uint32_t acks_addr = 0;         // counters: to_sender_channel_remote_ack_counters
    std::uint32_t completed_stream = 0;  // registers
    std::uint32_t acked_stream = 0;      // registers
    std::uint64_t sampled_completions = 0;  // rx: last published
    std::uint64_t sampled_acks = 0;
    std::uint32_t injected_completions = 0;  // tx: last written, for a counter
    std::uint32_t injected_acks = 0;
};

// RX SIDE. Reads the receive-end credit block the bridged router writes -- the layout is
// kBridgeRecvCreditBytes: completions then acks, one uint32 each per sender channel.
inline bool sample_recv_credits(
    tt::tt_metal::MetalEnvImpl& env,
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& eth_logical,
    std::uint32_t credits_addr,
    std::uint32_t channels,
    CreditSnapshot& out) {
    out.completions.assign(channels, 0);
    out.acks.assign(channels, 0);
    std::vector<std::uint32_t> raw(2u * channels, 0);
    if (!read_bytes_from_l1(
            env,
            dev,
            eth_logical,
            credits_addr,
            raw.data(),
            static_cast<std::uint32_t>(raw.size() * sizeof(std::uint32_t)),
            tt::CoreType::ETH)) {
        return false;
    }
    for (std::uint32_t i = 0; i < channels; ++i) {
        out.completions[i] = raw[i];
        out.acks[i] = raw[channels + i];
    }
    return true;
}

// TX SIDE. The sender reads these as running totals and takes an unsigned difference against what
// it has processed, so the host writes the total the relay carried -- no deltas, no encoding.
inline bool inject_sender_credits(
    tt::tt_metal::MetalEnvImpl& env,
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& eth_logical,
    std::uint32_t completions_addr,
    std::uint32_t acks_addr,
    std::uint32_t relayed_completions,
    std::uint32_t relayed_acks,
    std::string& err) {
    if (!write_bytes_to_l1(
            env, dev, eth_logical, completions_addr, &relayed_completions, sizeof(std::uint32_t), tt::CoreType::ETH)) {
        err = "credit relay: completion counter write failed";
        return false;
    }
    if (!write_bytes_to_l1(
            env, dev, eth_logical, acks_addr, &relayed_acks, sizeof(std::uint32_t), tt::CoreType::ETH)) {
        err = "credit relay: ack counter write failed";
        return false;
    }
    return true;
}

// TX SIDE, REGISTER FORM. The register ACCUMULATES, so only what is new may be written -- and
// unlike the counter form a lost or repeated write corrupts the count with nothing to detect it.
inline bool inject_sender_credits_stream(
    tt::tt_metal::MetalEnvImpl& env,
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& eth_logical,
    const tt::tt_metal::Hal& hal,
    std::uint32_t completed_stream,
    std::uint32_t acked_stream,
    std::uint32_t relayed_completions,
    std::uint32_t relayed_acks,
    std::uint32_t& injected_completions,
    std::uint32_t& injected_acks,
    std::string& err) {
    const std::uint32_t new_completions = relayed_completions - injected_completions;
    const std::uint32_t new_acks = relayed_acks - injected_acks;
    // Advanced only on success, so a failed write is retried rather than silently skipped.
    if (new_completions != 0) {
        if (!reg_write(
                env,
                dev,
                eth_logical,
                tt::CoreType::ETH,
                doorbell_addr(hal, completed_stream),
                doorbell_value(static_cast<std::int32_t>(new_completions), kWordsFreeIncShiftBlackhole))) {
            err = "credit relay: completion stream write failed";
            return false;
        }
        injected_completions = relayed_completions;
    }
    if (new_acks != 0) {
        if (!reg_write(
                env,
                dev,
                eth_logical,
                tt::CoreType::ETH,
                doorbell_addr(hal, acked_stream),
                doorbell_value(static_cast<std::int32_t>(new_acks), kWordsFreeIncShiftBlackhole))) {
            err = "credit relay: ack stream write failed";
            return false;
        }
        injected_acks = relayed_acks;
    }
    return true;
}

}  // namespace tt::tt_fabric::erisc_bridge
