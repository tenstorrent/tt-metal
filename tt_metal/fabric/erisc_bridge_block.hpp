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

}  // namespace tt::tt_fabric::erisc_bridge
