// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Where the bridge's control block sits in a router L1, derived from the live fabric config.
// Here, not distributed/, so the router builder and the socket opener derive the same address.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"

namespace tt::tt_fabric::erisc_bridge {

// Layout: a fixed 64 B status, then 128 B per channel (socket config, then descriptor scratch).
// One socket per serviced channel -- two senders over one config would share a single cursor.
inline constexpr std::uint32_t kBridgeBlockAlign = 64;  // D2HSocket rejects a page that is not 64-aligned
inline constexpr std::uint32_t kBridgeStatusBytes = 64;
inline constexpr std::uint32_t kBridgeChannelStride = 128;
inline constexpr std::uint32_t kBridgeChannels = static_cast<std::uint32_t>(builder_config::num_max_sender_channels);
inline constexpr std::uint32_t kBridgeBlockBytes = kBridgeStatusBytes + kBridgeChannels * kBridgeChannelStride;

// Offsets from the block base. The kernel computes the identical three from its own copy of
// these constants -- see fabric_router_e2h.hpp, which must not drift from this.
inline constexpr std::uint32_t kStatusOffset = 0;
constexpr std::uint32_t bridge_socket_config_offset(std::uint32_t ch) {
    return kBridgeStatusBytes + ch * kBridgeChannelStride;
}
constexpr std::uint32_t bridge_desc_scratch_offset(std::uint32_t ch) { return bridge_socket_config_offset(ch) + 64; }

// The router's own account of itself, mirroring E2hStatus in fabric_router_e2h.hpp. Without it,
// "no frame arrived" -- kernel, enable, socket or gate -- looks identical from the host.
struct BridgeStatus {
    std::uint32_t magic;  // kBridgeStatusMagic once the init block has run
    std::uint32_t open_tries;
    std::uint32_t opened;       // sockets opened; should equal the bridged channels that sent
    std::uint32_t frames;       // frames the router pushed to the host
    std::uint32_t declined;     // times the gate refused for want of a local socket slot
    std::uint32_t build_stamp;  // kBridgeBuildStamp if THIS binary is the one on the core
    // Must match E2hStatus. `declined` counts only a refusal after can_send was already true,
    // so a router blocked upstream of the bridge reports declined=0.
    std::uint32_t blocked_rx;      // had a packet, far receiver had no free slot
    std::uint32_t blocked_nodata;  // the producer offered nothing
    std::uint32_t free_slots;      // last observed outbound_to_receiver num_free_slots
    std::uint32_t host_armed;      // kBridgeHostArmed once every socket config is fully written
};
inline constexpr std::uint32_t kBridgeStatusMagic = 0x45324853;  // 'E2HS'
inline constexpr std::uint32_t kBridgeBuildStamp = 0x42524447;   // 'BRDG'
inline constexpr std::uint32_t kBridgeHostArmed = 0x41524D44;    // 'ARMD'

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
    if (l1_top <= buffers_end) {
        return 0;
    }
    const std::uint32_t addr = (l1_top - want_bytes) & ~(kBridgeBlockAlign - 1);
    return addr < buffers_end ? 0u : addr;
}

}  // namespace tt::tt_fabric::erisc_bridge
