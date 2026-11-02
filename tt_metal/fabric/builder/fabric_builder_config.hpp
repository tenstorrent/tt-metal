// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>

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

// A VC's routing mode, within the fabric's header format: the topology's, fabric-wide (Linear, Ring and
// NeighborExchange route with 1D headers, Mesh and Torus with 2D ones). The host sets one per VC
// (FabricBuilderContext's VC config) and validates the combination; the router follows the per-VC facts the host
// derives from it (fabric_exp/consolidation/LINE_VCS_HOST_DESIGN.md).
enum class VcRouting : uint8_t {
    OFF,         // no channels
    FULL_2D,     // 2D routing with turns (VC0 on 2D fabrics)
    LINE,        // straight on: a worker channel, one forwarding channel, a receiver that forwards straight on
                 // (every VC on 1D fabrics; VC2 in line mode on 2D fabrics)
    NEIGHBOUR,   // one hop: a worker channel, a receiver that only delivers (VC2's default on 2D fabrics)
    INTER_MESH,  // VC1 on multi-mesh 2D fabrics: forwarding channels only, carries the traffic crossed over from
                 // VC0 or VC2 at the intermesh routers
};
using VcRoutings = std::array<VcRouting, MAX_NUM_VCS>;

// Whether a VC in this mode has a worker channel (its first sender channel).
constexpr bool vc_routing_has_worker(VcRouting routing) {
    return routing == VcRouting::FULL_2D || routing == VcRouting::LINE || routing == VcRouting::NEIGHBOUR;
}

// Bubble flow control runs on the VCs that route with turns or as lines (on ring and torus axes): VC0, VC2 in line
// mode, every line VC of a 1D fabric. VC1's intermesh dependency graph has no bubble proof for arbitrary cross-mesh
// patterns, so its senders remain unguarded; a neighbour-mode VC forwards nothing, so it has no ring to guard.
// Injection flags and protected-slot validation must share this predicate. First-level ACK goes with it per VC (VC1's
// inter-mesh ACK paths do not fit in the ACTIVE_ETH kernel config window, see ENABLE_FIRST_LEVEL_ACK_VC1 in
// erisc_datamover_builder.cpp).
constexpr bool bubble_flow_control_enabled_on_vc(VcRouting routing) {
    return routing == VcRouting::FULL_2D || routing == VcRouting::LINE;
}

// NoC transaction IDs per receiver channel (the v4 router's rule, emitted for it as
// NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL): the NoC's 16 IDs shared by a router's receiver channels, each channel's
// count a power of 2 (router_completions.hpp): one channel 16, or 8 under deadlock avoidance; two channels 8 under
// deadlock avoidance, else 4; three channels 4.
constexpr uint32_t transaction_ids_per_receiver_channel(size_t num_receiver_channels, bool deadlock_avoidance) {
    if (num_receiver_channels == 1) {
        return deadlock_avoidance ? 8 : 16;
    }
    return deadlock_avoidance && num_receiver_channels <= 2 ? 8 : 4;
}

uint32_t get_sender_channel_count(bool is_2D_routing);

uint32_t get_receiver_channel_count(bool is_2D_routing);

uint32_t get_num_used_sender_channel_count(Topology topology);

uint32_t get_num_tensix_sender_channels(Topology topology, tt::tt_fabric::FabricTensixConfig fabric_tensix_config);

uint32_t get_downstream_edm_count(bool is_2D_routing);

uint32_t get_vc0_downstream_edm_count(bool is_2D_routing, bool express_routing_enabled = false);

uint32_t get_vc1_downstream_edm_count(bool is_2D_routing, bool express_routing_enabled = false);

}  // namespace builder_config

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
