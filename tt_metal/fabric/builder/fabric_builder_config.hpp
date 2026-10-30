// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"
#include <tt-umd/types/arch.hpp>
#include <vector>
#include <algorithm>

namespace tt::tt_fabric {


/**
 * Memory region definition for fabric channel allocation.
 * Represents a contiguous memory region with start and size.
 */
struct MemoryRegion {
    size_t start_address;
    size_t size;

    MemoryRegion(size_t start, size_t size) : start_address(start), size(size) {
    }

    size_t get_size() const { return size; }
    size_t get_start_address() const { return start_address; }
    size_t get_end_address() const { return start_address + size; }
    bool contains(size_t address) const { return address >= start_address && address < get_end_address(); }
};

// enable extra buffer slots configuration based on sender/receiver channel and EDM type.
struct FabricEriscDatamoverOptions {
    FabricTensixConfig fabric_tensix_config = FabricTensixConfig::DISABLED;
    eth_chan_directions direction = eth_chan_directions::EAST;  // only used by 2D to get the correct router direction
    tt::ARCH arch = tt::ARCH::Invalid;                          // selects the per-arch buffer slot tables
};

namespace builder_config {
// Number of Virtual Channels supported (VC0 and VC1)
static constexpr std::size_t MAX_NUM_VCS = 3;

// linear/mesh/ring/torus: for fabric with tensix extension, only one sender channel will be present on fabric router
static constexpr std::size_t num_sender_channels_with_tensix_config = 1;

// num sender channels based on more accurate topology
static constexpr std::size_t num_sender_channels_1d_neighbor_exchange = 1;
static constexpr std::size_t num_sender_channels_1d_linear = 2;
// Frozen non-express VC0 width used by the tensix/L1 path. router_wiring_rules.hpp asserts that its
// wiring-side derivation matches this value.
static constexpr std::size_t num_sender_channels_2d_mesh = 4;

// Wider family counts are derived from the wiring rules. Express and Z-boundary families both
// reach 5 VC0 / 4 VC1 senders through different wiring.

// VC2: 1 sender channel (worker-type, neighbour exchange) + 1 receiver (non-Z only)
static constexpr std::size_t num_sender_channels_vc2 = 1;
static constexpr std::size_t num_receiver_channels_vc2 = 1;

static constexpr std::size_t num_sender_channels_1d = 2;
// Widest 2D family: 5 VC0 + 4 VC1 senders. VC1 includes Z because a carrier that crossed a mesh
// boundary remains on VC1 and can still decode a Z action. VC2 adds one sender dynamically.
static constexpr std::size_t num_sender_channels_2d = 9;
// Max including VC2 -- used only for array sizing.
static constexpr std::size_t num_sender_channels_2d_with_vc2 = num_sender_channels_2d + num_sender_channels_vc2;
// Without VC2 -- used for firmware CT args and L1 layout when VC2 is disabled.
static constexpr std::size_t num_max_sender_channels_without_vc2 =
    std::max({num_sender_channels_1d, num_sender_channels_2d});
// Absolute maximum -- used for host-side array sizing.
static constexpr std::size_t num_max_sender_channels =
    std::max({num_sender_channels_1d, num_sender_channels_2d_with_vc2});
static constexpr std::size_t num_receiver_channels_1d = 1;
// Without VC2 -- VC2 receiver is added dynamically.
static constexpr std::size_t num_receiver_channels_2d = 2;  // VC0(1) + VC1(1)
// Max including VC2 -- used only for array sizing.
static constexpr std::size_t num_receiver_channels_2d_with_vc2 = num_receiver_channels_2d + num_receiver_channels_vc2;
// Without VC2 -- used for firmware CT args and L1 layout when VC2 is disabled.
static constexpr std::size_t num_max_receiver_channels_without_vc2 =
    std::max({num_receiver_channels_1d, num_receiver_channels_2d});
// Absolute maximum -- used for host-side array sizing.
static constexpr std::size_t num_max_receiver_channels =
    std::max({num_receiver_channels_1d, num_receiver_channels_2d_with_vc2});

static constexpr std::size_t num_downstream_edms_vc0 = 1;
static constexpr std::size_t num_downstream_edms_2d_vc0 = 3;
// Express Y-facing VC0 can continue cardinally, take Z, or turn onto either X direction. The fourth
// downstream stream register is already reserved.
static constexpr std::size_t num_downstream_edms_2d_vc0_express = 4;
static constexpr std::size_t num_downstream_edms_2d_vc1 = 3;  // XY intermesh: 3 mesh directions
static constexpr std::size_t num_downstream_edms_2d_vc1_wide =
    4;  // widest VC1 fanout: 3 mesh + Z (Z-intermesh boundary or express)
static constexpr std::size_t num_downstream_edms_1d = num_downstream_edms_vc0;
static constexpr std::size_t num_downstream_edms_2d = num_downstream_edms_2d_vc0 + num_downstream_edms_2d_vc1;
static constexpr std::size_t max_downstream_edms = 8;

// 2D mesh directions (N, E, S, W)
static constexpr uint32_t num_mesh_directions_2d = 4;

// The Z-facing intermesh boundary family's channel counts (5 VC0 / 4 VC1, family max) are derived
// in builder/router_wiring_rules.* (boundary_vc0/vc1_sender_count) from this direction count, not
// stated here.

// Bubble injection requires at least two downstream slots. Must match
// BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS and fabric_router_mux_extension.cpp;
// the host cannot include the device constant.
static constexpr uint32_t bubble_flow_control_protected_receiver_min_slots = 2;

// Only VC0 uses bubble flow control. VC1's intermesh dependency graph has no bubble proof for
// arbitrary cross-mesh patterns, so its senders remain unguarded. Injection flags and protected-slot
// validation must share this predicate. First-level ACK is independent; VC1 ACK paths do not fit in
// the ACTIVE_ETH kernel config window (see ENABLE_FIRST_LEVEL_ACK_VC1 in erisc_datamover_builder.cpp).
constexpr bool bubble_flow_control_enabled_on_vc(uint32_t vc) { return vc == 0; }

uint32_t get_sender_channel_count(bool is_2D_routing);

uint32_t get_receiver_channel_count(bool is_2D_routing);

uint32_t get_num_used_sender_channel_count(Topology topology);

uint32_t get_num_tensix_sender_channels(Topology topology, tt::tt_fabric::FabricTensixConfig fabric_tensix_config);

uint32_t get_downstream_edm_count(bool is_2D_routing);

uint32_t get_vc0_downstream_edm_count(bool is_2D_routing, bool express_routing_enabled = false);

uint32_t get_vc1_downstream_edm_count(bool is_2D_routing, bool express_routing_enabled = false);

}  // namespace builder_config

/**
 * Which of a router's sender channel slots carry a producer, and how many live slots the channel
 * allocator should size for.
 *
 * Slot positions are fixed by direction (see RouterProducerSlots), so a router whose facing leaves
 * some producer slots unwired -- an N/S router's from-E/from-W slots under dimension-order routing,
 * the from-Z slot on a chip with no Z port -- keeps those slots in its layout but gives them no
 * packet buffering. `sender_live_mask[vc]` has bit `i` set when sender channel `i` of that VC is
 * live. `link_live_sender_counts[vc]` is the live count the allocator fits option rows against: the
 * larger of this router's and its link peer's live counts, so both ends of a link select the same
 * row and therefore the same receiver depth (the sender writes straight into the peer's receiver).
 */
struct RouterChannelLiveness {
    std::array<uint16_t, builder_config::MAX_NUM_VCS> sender_live_mask{};
    std::array<size_t, builder_config::MAX_NUM_VCS> link_live_sender_counts{};

    bool is_sender_live(size_t vc, size_t channel) const {
        return vc < builder_config::MAX_NUM_VCS && ((sender_live_mask[vc] >> channel) & 1U) != 0;
    }

    size_t live_sender_count(size_t vc) const {
        return vc < builder_config::MAX_NUM_VCS ? static_cast<size_t>(__builtin_popcount(sender_live_mask[vc])) : 0;
    }

    // Every slot live, fit against the allocated counts: the pre-liveness layout.
    static RouterChannelLiveness all_live(const std::array<size_t, builder_config::MAX_NUM_VCS>& sender_counts) {
        RouterChannelLiveness liveness;
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            liveness.sender_live_mask[vc] = static_cast<uint16_t>((1U << sender_counts[vc]) - 1U);
            liveness.link_live_sender_counts[vc] = sender_counts[vc];
        }
        return liveness;
    }

    // A total order so configs can be cached per distinct liveness.
    uint64_t key() const {
        uint64_t packed = 0;
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            packed = (packed << 16) | sender_live_mask[vc];
        }
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            packed = (packed << 4) | (link_live_sender_counts[vc] & 0xFU);
        }
        return packed;
    }

    bool operator==(const RouterChannelLiveness& other) const {
        return sender_live_mask == other.sender_live_mask && link_live_sender_counts == other.link_live_sender_counts;
    }
};

/**
 * Structure to hold all parameters needed for allocator construction.
 * This simplifies passing multiple parameters to allocator constructors.
 */
struct AllocatorConstructionParams {
    Topology topology;
    FabricEriscDatamoverOptions options;
    size_t num_used_sender_channels;
    size_t num_used_receiver_channels;
    size_t channel_buffer_size_bytes;
    size_t available_channel_buffering_space;
    std::vector<MemoryRegion> memory_regions;

    AllocatorConstructionParams(
        Topology topology,
        const FabricEriscDatamoverOptions& options,
        size_t num_used_sender_channels,
        size_t num_used_receiver_channels,
        size_t channel_buffer_size_bytes,
        size_t available_channel_buffering_space,
        const std::vector<MemoryRegion>& memory_regions) :
        topology(topology),
        options(options),
        num_used_sender_channels(num_used_sender_channels),
        num_used_receiver_channels(num_used_receiver_channels),
        channel_buffer_size_bytes(channel_buffer_size_bytes),
        available_channel_buffering_space(available_channel_buffering_space),
        memory_regions(memory_regions) {}
};

}  // namespace tt::tt_fabric
