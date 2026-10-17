// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Where the bridge's control block sits in a router L1, derived from the live fabric config.
// Here, not distributed/, so the router builder and the socket opener derive the same address.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>

#include "hostdevcommon/erisc_bridge_layout.h"
#include "tt_metal/fabric/bridged_cables.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"

namespace tt::tt_fabric::erisc_bridge {

// The block's own layout is shared with the kernel, so it is re-exported here rather than
// restated: callers keep saying erisc_bridge::BridgeStatus and there is still one definition.
using tt::tt_fabric::BridgeStatus;
using tt::tt_fabric::bridge_desc_scratch_offset;
using tt::tt_fabric::bridge_socket_config_offset;
using tt::tt_fabric::kBridgeBlockAlign;
using tt::tt_fabric::kBridgeBuildStamp;
using tt::tt_fabric::kBridgeChannelStride;
using tt::tt_fabric::kBridgeHostArmed;
using tt::tt_fabric::kBridgeStatusBytes;
using tt::tt_fabric::kBridgeStatusMagic;
using tt::tt_fabric::kStatusOffset;

inline constexpr std::uint32_t kBridgeChannels = static_cast<std::uint32_t>(builder_config::num_max_sender_channels);
inline constexpr std::uint32_t kBridgeBlockBytes = kBridgeStatusBytes + kBridgeChannels * kBridgeChannelStride;
// The receiving end's credit totals: completions then acks, one uint32 per sender channel. Must
// match kE2hRecvCreditWords in host_bridge/erisc_bridge_credit_sender.hpp.
inline constexpr std::uint32_t kBridgeRecvCreditBytes = 2 * kBridgeChannels * sizeof(std::uint32_t);

// One past the highest byte any sender or receiver channel buffer occupies. Returns 0 when the
// allocator is not the static one -- a caller must read that as unknown, not as "nothing here".
inline std::uint64_t bridge_highest_buffer_end(const FabricEriscDatamoverConfig& cfg) {
    const auto alloc = std::dynamic_pointer_cast<FabricStaticSizedChannelsAllocator>(cfg.channel_allocator);
    if (alloc == nullptr) {
        return 0;
    }
    const std::uint64_t slot = cfg.channel_buffer_size_bytes;
    std::uint64_t end = 0;
    for (std::size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (std::size_t c = 0; c < cfg.num_used_sender_channels_per_vc[vc]; ++c) {
            end = std::max(
                end,
                alloc->get_sender_channel_base_address(vc, c) +
                    alloc->get_sender_channel_number_of_slots(vc, c) * slot);
        }
        for (std::size_t c = 0; c < cfg.num_used_receiver_channels_per_vc[vc]; ++c) {
            end = std::max(
                end,
                alloc->get_receiver_channel_base_address(vc, c) +
                    alloc->get_receiver_channel_number_of_slots(vc, c) * slot);
        }
    }
    return end;
}

// Base address, or 0 when the tail cannot hold it. The tail above fabric's last channel buffer,
// not a found gap: if fabric grows into it this returns 0 and the caller must fail.
inline std::uint32_t bridge_block_addr(
    const FabricEriscDatamoverConfig& cfg, std::uint32_t want_bytes = kBridgeBlockBytes) {
    const std::uint64_t buffers_end = bridge_highest_buffer_end(cfg);
    if (buffers_end == 0) {
        return 0;
    }
    const auto l1_top = static_cast<std::uint32_t>(cfg.max_l1_loading_size);
    // The tail is checked BEFORE the subtraction: want_bytes > l1_top wraps the uint32 to a high
    // address, which the `addr < buffers_end` test below cannot catch because it is not low.
    if (l1_top <= buffers_end || want_bytes > l1_top - buffers_end) {
        return 0;
    }
    const std::uint32_t addr = (l1_top - want_bytes) & ~(kBridgeBlockAlign - 1);
    return addr < buffers_end ? 0u : addr;
}

// A virtual-cable router as its kernel numbers channels (actual per-VC counts, VC2_RECEIVER_CHANNEL rule).
inline experimental::BridgedRouterLayout describe_bridged_router(const FabricEriscDatamoverBuilder& builder) {
    const auto& cfg = builder.config;
    const auto& actual_senders = builder.actual_sender_channels_per_vc_;
    const auto& actual_receivers = builder.actual_receiver_channels_per_vc_;
    experimental::BridgedRouterLayout out;
    out.block_addr = bridge_block_addr(cfg);
    // The receive-credit block sits just under the E2H block; 0 if it doesn't fit.
    const std::uint32_t credits = (out.block_addr - kBridgeRecvCreditBytes) & ~(kBridgeBlockAlign - 1);
    const bool fits = out.block_addr >= kBridgeRecvCreditBytes && credits >= bridge_highest_buffer_end(cfg);
    out.recv_credits_addr = fits ? credits : 0u;
    out.slot_bytes = static_cast<std::uint32_t>(cfg.channel_buffer_size_bytes);
    out.completion_counters_addr =
        static_cast<std::uint32_t>(cfg.to_sender_channel_remote_completion_counters_base_addr);
    out.ack_counters_addr = static_cast<std::uint32_t>(cfg.to_sender_channel_remote_ack_counters_base_addr);
    const auto alloc = std::dynamic_pointer_cast<FabricStaticSizedChannelsAllocator>(cfg.channel_allocator);
    auto senders = [&](std::size_t vc) {
        return actual_senders.has_value() ? (*actual_senders)[vc] : cfg.num_used_sender_channels_per_vc[vc];
    };
    auto receivers = [&](std::size_t vc) {
        return actual_receivers.has_value() ? (*actual_receivers)[vc] : cfg.num_used_receiver_channels_per_vc[vc];
    };
    for (std::size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (std::size_t c = 0; c < senders(vc); ++c) {
            out.sender_vc.push_back(static_cast<std::uint32_t>(vc));
            out.sender_index_in_vc.push_back(static_cast<std::uint32_t>(c));
        }
        if (receivers(vc) > 0 && alloc != nullptr) {
            const std::uint32_t kernel_channel =
                vc < 2 ? static_cast<std::uint32_t>(vc) : (builder_config::MAX_NUM_VCS > 1 && senders(1) > 0 ? 2u : 1u);
            out.receiver_channel.push_back(kernel_channel);
            out.receiver_base.push_back(static_cast<std::uint32_t>(alloc->get_receiver_channel_base_address(vc, 0)));
            out.receiver_slots.push_back(
                static_cast<std::uint32_t>(alloc->get_receiver_channel_number_of_slots(vc, 0)));
        } else {
            out.receiver_channel.push_back(experimental::BridgedRouterLayout::kNoReceiver);
            out.receiver_base.push_back(0);
            out.receiver_slots.push_back(0);
        }
    }
    return out;
}

}  // namespace tt::tt_fabric::erisc_bridge
