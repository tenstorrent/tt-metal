// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Which router on a chip leads the chip's local handshake.
//
// The leader's kernel carries the extra handshake code, so ideally we designate the master router role to a router that
// has a small kernel size. A router's size follows from its compile-time arguments:
//   - the route decode (router_route_decode.hpp). A router on a mesh's edge (IS_INTRAMESH_ROUTER_ON_EDGE) decodes for
//     its facing and also checks every packet for leaving the mesh here, whereas an intermesh router 
//     (IS_INTERMESH_ROUTER) only takes in packets that have just landed, so its decode is the one Y-first function.
//   - the live downstreams of each VC it forwards on. Each one adds a copy of the per-packet forward, free-slot check
//     and credit read code (router_downstream.hpp, live_downstreams<VC>()).

// What the choice needs to know about one router, taken from the compile-time arguments of the router's first RISC.
struct MasterRouterCandidate {
    chan_id_t eth_chan;
    eth_chan_directions facing;
    bool is_2d_fabric;
    // IS_INTERMESH_ROUTER and IS_INTRAMESH_ROUTER_ON_EDGE.
    bool is_intermesh_router;
    bool is_intramesh_router_on_edge;
    // VC0_DOWNSTREAM_EDM_SIZE and VC1_DOWNSTREAM_EDM_SIZE; 0 for a VC the router does not forward on.
    std::array<uint32_t, 2> downstream_count_by_vc;
};

// The channel of the router that should lead the chip's local handshake: a router that is not on a mesh's edge (an
// intermesh router, where the chip has one), then the one with the fewest forwarding downstreams, then the lowest
// channel. The last step means routers of the same size give the lowest channel. `candidates` must not be empty.
chan_id_t pick_local_handshake_master_router(std::span<const MasterRouterCandidate> candidates);

}  // namespace tt::tt_fabric
