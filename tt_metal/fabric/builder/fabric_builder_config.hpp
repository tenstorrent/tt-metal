// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"
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
};

namespace builder_config {
// Number of Virtual Channels supported (VC0 and VC1)
static constexpr std::size_t MAX_NUM_VCS = 3;

// linear/mesh/ring/torus: for fabric with tensix extension, only one sender channel will be present on fabric router
static constexpr std::size_t num_sender_channels_with_tensix_config = 1;

// num sender channels based on more accurate topology
static constexpr std::size_t num_sender_channels_1d_neighbor_exchange = 1;
static constexpr std::size_t num_sender_channels_1d_linear = 2;
// The frozen non-express forwarding VC0 count, used by the tensix/L1 path. The wiring-side
// statement of the same number is non_express_vc0_sender_count() in router_wiring_rules (the two
// are static_assert-ed equal there); they unify when the tensix path is widened for express.
static constexpr std::size_t num_sender_channels_2d_mesh = 4;

// Per-family sender counts beyond the frozen non-express 2D width are not constants here: they
// are derived in builder/router_wiring_rules.* from the wiring rules -- the express family as the
// family max over facing of wired-producer arity, the boundary family from the mesh-direction
// count. Both are 5 VC0 / 4 VC1, by unrelated arithmetic.

// VC2: 1 sender channel (worker-type, neighbour exchange) + 1 receiver (non-Z only)
static constexpr std::size_t num_sender_channels_vc2 = 1;
static constexpr std::size_t num_receiver_channels_vc2 = 1;

static constexpr std::size_t num_sender_channels_1d = 2;
// VC0: Worker + 3 of [N/E/S/W], plus the express chord when express routing is on = 4 or 5 channels
// VC1: Up to 4 channels (no worker): 3 of [N/E/S/W] for inter-mesh, plus a Z sender when the device
// has an intermesh Z router or the mesh has express routing -- a carrier that crossed a mesh
// boundary stays on VC1 and can still decode a Z action, so the express output must exist on VC1.
// Total 2D without VC2: 5 + 4 = 9 channels (VC2 added dynamically)
//
// Sized for the widest 2D shape so the flat index space is the same whether or not express routing is
// enabled. num_max_sender_channels is unchanged at 10: express with VC2 reaches it exactly (5+4+1),
// matching the capacity analysis in GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md section 3.6. The
// Z-facing intermesh boundary family is also 5+4(+1), so the ceilings below cover it as well.
static constexpr std::size_t num_sender_channels_2d = 9;
// Max including VC2 — used only for array sizing
static constexpr std::size_t num_sender_channels_2d_with_vc2 = num_sender_channels_2d + num_sender_channels_vc2;
// Without VC2 — used for firmware CT args and L1 layout when VC2 is disabled
static constexpr std::size_t num_max_sender_channels_without_vc2 =
    std::max({num_sender_channels_1d, num_sender_channels_2d});
// = max(2, 9) = 9
// Absolute maximum — used for host-side array sizing (always big enough for any config)
static constexpr std::size_t num_max_sender_channels =
    std::max({num_sender_channels_1d, num_sender_channels_2d_with_vc2});
// = max(2, 10) = 10
static constexpr std::size_t num_receiver_channels_1d = 1;
// Without VC2 — VC2 receiver added dynamically
static constexpr std::size_t num_receiver_channels_2d = 2;  // VC0(1) + VC1(1)
// Max including VC2 — used only for array sizing
static constexpr std::size_t num_receiver_channels_2d_with_vc2 = num_receiver_channels_2d + num_receiver_channels_vc2;
// Without VC2 — used for firmware CT args and L1 layout when VC2 is disabled
static constexpr std::size_t num_max_receiver_channels_without_vc2 =
    std::max({num_receiver_channels_1d, num_receiver_channels_2d});
// = max(1, 2) = 2
// Absolute maximum — used for host-side array sizing (always big enough for any config)
static constexpr std::size_t num_max_receiver_channels =
    std::max({num_receiver_channels_1d, num_receiver_channels_2d_with_vc2});
// = max(1, 3) = 3

static constexpr std::size_t num_downstream_edms_vc0 = 1;
static constexpr std::size_t num_downstream_edms_2d_vc0 = 3;
// With express routing, a Y-facing VC0 receiver can fan out to four downstream routers rather than
// three: continue Y cardinally, take the express chord, and turn onto either X direction. The stream
// register for that fourth edge already exists (vc_0_free_slots_from_downstream_edge_4), so this
// widening needs no new flow-control resource.
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

// Slots an injection channel's downstream receiver must have for bubble flow control to work: it only
// sends when it sees this many free, so a smaller receiver stalls it permanently.
//
// Must match BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS on the device side, which the
// host cannot include. That constant is already duplicated in fabric_router_mux_extension.cpp, so this
// is the third copy; they have to move together.
static constexpr uint32_t bubble_flow_control_protected_receiver_min_slots = 2;

// Which VCs actually realize bubble flow control. VC1 carries intermesh traffic, whose dependency
// graph has no bubble proof for arbitrary cross-mesh patterns, so its senders run unguarded even
// where the express derivation classifies one as a ring acquisition
// (GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md 3.4, assessment 5.7.5).
//
// This is not a preference the call sites may each restate: the injection flags and the
// protected-receiver slot check both read it, and disagreement between them is a silent
// miscompile rather than an error.
//
// First-level ack is not derived from this, and the two only look coupled by coincidence. The
// device's static_assert is one-way -- a sender flagged as injection under deadlock avoidance
// requires first-level ack on its VC, but not the converse -- so an unguarded VC is free to ack
// early. VC1 does not, for the unrelated reason that the ack paths do not fit in the ACTIVE_ETH
// kernel config window (see ENABLE_FIRST_LEVEL_ACK_VC1 in erisc_datamover_builder.cpp).
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
