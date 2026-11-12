// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for the choice of the router that leads a chip's local handshake.
//
// The leader's kernel carries the handshake code on top of the router's own, so the leader is a router whose kernel
// is already small. These tests pin what makes one router smaller than another and the order the choice weighs it in.

#include <gtest/gtest.h>

#include <vector>

#include "tt_metal/fabric/builder/fabric_master_router_selection.hpp"

namespace tt::tt_fabric {
namespace {

constexpr bool k_2d = true;

// A plain router inside a mesh: no role flags, forwarding on VC0 only.
MasterRouterCandidate make_candidate(chan_id_t eth_chan, eth_chan_directions facing) {
    return MasterRouterCandidate{
        .eth_chan = eth_chan,
        .facing = facing,
        .is_2d_fabric = k_2d,
        .is_intermesh_router = false,
        .is_intramesh_router_on_edge = false,
        .downstream_count_by_vc = {3, 0},
    };
}

// An intermesh router, facing north, forwarding to three downstreams on both VCs. It is not on a mesh's edge: a chip
// with intermesh links has these beside its mesh-edge routers.
MasterRouterCandidate make_intermesh_candidate(chan_id_t eth_chan) {
    auto candidate = make_candidate(eth_chan, eth_chan_directions::NORTH);
    candidate.is_intermesh_router = true;
    candidate.downstream_count_by_vc = {3, 3};
    return candidate;
}

// The router of the Wormhole galaxy split into two 4x4 meshes whose kernel does not fit with the handshake code: facing
// north, on the mesh edge, forwarding to three downstreams on both VCs.
MasterRouterCandidate make_north_mesh_edge_candidate(chan_id_t eth_chan) {
    auto candidate = make_candidate(eth_chan, eth_chan_directions::NORTH);
    candidate.is_intramesh_router_on_edge = true;
    candidate.downstream_count_by_vc = {3, 3};
    return candidate;
}

// Ensure that the tiebreak case isn't hit when an actual case exists (difference in forwarding downstreams).
TEST(MasterRouterSelectionTest, FewerForwardingDownstreamsWinsOverALowerChannel) {
    // The galaxy case: the north-facing mesh-edge router has the lowest channel but the most to forward.
    const std::vector<MasterRouterCandidate> candidates = {
        make_north_mesh_edge_candidate(0),
        make_candidate(3, eth_chan_directions::EAST),
        make_candidate(7, eth_chan_directions::SOUTH),
    };
    EXPECT_EQ(pick_local_handshake_master_router(candidates), 3);
}

// Ensure that intermesh routers win over mesh-edge routers.
TEST(MasterRouterSelectionTest, IntermeshRouterWinsOverMeshEdgeRoutersWithLowerChannels) {
    const std::vector<MasterRouterCandidate> candidates = {
        make_north_mesh_edge_candidate(0),
        make_north_mesh_edge_candidate(1),
        make_intermesh_candidate(6),
    };
    EXPECT_EQ(pick_local_handshake_master_router(candidates), 6);
}

TEST(MasterRouterSelectionTest, NotBeingOnTheMeshEdgeOutweighsFewerDownstreams) {
    // The mesh-edge router forwards to two downstreams on VC0 only, the intermesh router to three on both VCs: the
    // decode, not the downstream count, sets the size.
    auto small_mesh_edge_router = make_north_mesh_edge_candidate(1);
    small_mesh_edge_router.downstream_count_by_vc = {2, 0};
    const std::vector<MasterRouterCandidate> candidates = {small_mesh_edge_router, make_intermesh_candidate(6)};
    EXPECT_EQ(pick_local_handshake_master_router(candidates), 6);
}

TEST(MasterRouterSelectionTest, FewerDownstreamsBreakTiesBetweenIntermeshRouters) {
    auto smaller_intermesh_router = make_intermesh_candidate(5);
    smaller_intermesh_router.downstream_count_by_vc = {2, 2};
    const std::vector<MasterRouterCandidate> candidates = {make_intermesh_candidate(2), smaller_intermesh_router};
    EXPECT_EQ(pick_local_handshake_master_router(candidates), 5);
}

// Tiebreak
TEST(MasterRouterSelectionTest, EqualIntermeshRoutersGiveTheLowestChannel) {
    const std::vector<MasterRouterCandidate> candidates = {make_intermesh_candidate(7), make_intermesh_candidate(3)};
    EXPECT_EQ(pick_local_handshake_master_router(candidates), 3);
}

// Tiebreak
TEST(MasterRouterSelectionTest, EqualRoutersGiveTheLowestChannel) {
    const std::vector<MasterRouterCandidate> candidates = {
        make_candidate(9, eth_chan_directions::NORTH),
        make_candidate(2, eth_chan_directions::NORTH),
        make_candidate(6, eth_chan_directions::NORTH),
    };
    EXPECT_EQ(pick_local_handshake_master_router(candidates), 2);
}

TEST(MasterRouterSelectionTest, OneRouterIsTheMaster) {
    const std::vector<MasterRouterCandidate> candidates = {make_north_mesh_edge_candidate(11)};
    EXPECT_EQ(pick_local_handshake_master_router(candidates), 11);
}

}  // namespace
}  // namespace tt::tt_fabric
