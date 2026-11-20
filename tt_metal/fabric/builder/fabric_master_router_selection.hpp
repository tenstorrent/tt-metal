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
// The leader's kernel carries the extra handshake code, so the role goes to the router with the smallest kernel. A
// router's size, approximated via a heuristic, follows from two of its compile-time arguments.
//
// 1. Its route decode (router_route_decode.hpp). This differs only on an exit chip, a chip with a link to another
//    mesh. A chip with no such link has plain routers that decode for their facing alone. On an exit chip there are two
//    kinds of router:
//
//       this mesh                             other mesh
//    +--------------------------+           +--------------+
//    |  neighbour     exit chip |           |   chip in    |
//    |  chip  <--[ A ]  [ B ]-----------------> the other   |
//    |                          |           |   mesh       |
//    +--------------------------+           +--------------+
//
//    A = intramesh router (IS_INTRAMESH_ROUTER_ON_EDGE): faces a chip in this mesh.
//    B = intermesh router (IS_INTERMESH_ROUTER): faces the other mesh.
//
//    - An intramesh router decodes for its facing and also checks every packet for leaving the mesh here. This is the
//      largest route decode in terms of code size.
//    - An intermesh router only takes in packets that have just landed in this mesh, so its decode is the single
//      Y-first function. This is the smallest decode.
//
// 2. Its live downstreams on each VC it forwards on: the routers it can pass a packet on to. Each one adds a copy of
//    the per-packet forward, free-slot check and credit read code (router_downstream.hpp, live_downstreams<VC>()).

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

// The channel of the router that should lead the chip's local handshake. Candidates are compared on:
//   1. Not an intramesh router on an exit chip (so an intermesh router wins where the chip has one).
//   2. The fewest forwarding downstreams.
//   3. The lowest eth channel #, when the first two tie.
// `candidates` must not be empty.
chan_id_t pick_local_handshake_master_router(std::span<const MasterRouterCandidate> candidates);

}  // namespace tt::tt_fabric
