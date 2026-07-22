// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/operations/ccl/all_gather/device/all_gather_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace {

using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::MeshId;
using tt::tt_fabric::Topology;
using ttnn::ccl::FabricNeighborRoutePlan;
using ttnn::ccl::FabricNeighborRouteResolution;

FabricNodeId node(uint32_t chip_id) { return FabricNodeId(MeshId{0}, chip_id); }

class SyntheticFabric {
public:
    void connect(
        uint32_t source, uint32_t destination, uint32_t physical_direction, std::vector<uint32_t> links = {0, 1}) {
        routes_.insert_or_assign(
            std::pair{node(source), node(destination)},
            FabricNeighborRouteResolution{
                .direct = true, .link_indices = std::move(links), .physical_direction = physical_direction});
    }

    void connect_bidirectional(
        uint32_t first,
        uint32_t second,
        uint32_t first_to_second_direction,
        uint32_t second_to_first_direction,
        std::vector<uint32_t> links = {0, 1}) {
        connect(first, second, first_to_second_direction, links);
        connect(second, first, second_to_first_direction, std::move(links));
    }

    FabricNeighborRouteResolution resolve(const FabricNodeId& source, const FabricNodeId& destination) const {
        const auto it = routes_.find(std::pair{source, destination});
        return it == routes_.end() ? FabricNeighborRouteResolution{} : it->second;
    }

private:
    std::map<std::pair<FabricNodeId, FabricNodeId>, FabricNeighborRouteResolution> routes_;
};

std::vector<FabricNodeId> logical_group(uint32_t size) {
    std::vector<FabricNodeId> nodes;
    nodes.reserve(size);
    for (uint32_t rank = 0; rank < size; ++rank) {
        nodes.push_back(node(rank));
    }
    return nodes;
}

FabricNeighborRoutePlan build_plan(
    uint32_t axis,
    Topology topology,
    uint32_t num_links,
    const std::vector<FabricNodeId>& group,
    const SyntheticFabric& fabric) {
    return ttnn::ccl::build_fabric_neighbor_route_plan(
        axis, topology, num_links, {group}, [&fabric](const FabricNodeId& source, const FabricNodeId& destination) {
            return fabric.resolve(source, destination);
        });
}

void expect_terminal_one_hop_edges(const FabricNeighborRoutePlan& plan) {
    ASSERT_TRUE(plan.eligible);
    for (const auto& edge : plan.edges) {
        EXPECT_TRUE(edge.direct);
        EXPECT_EQ(edge.packet_hops, 1u);
        for (uint32_t link = 0; link < plan.num_links; ++link) {
            EXPECT_NE(std::find(edge.link_indices.begin(), edge.link_indices.end(), link), edge.link_indices.end());
        }
    }
}

SyntheticFabric make_line(uint32_t size) {
    SyntheticFabric fabric;
    for (uint32_t rank = 0; rank + 1 < size; ++rank) {
        fabric.connect_bidirectional(rank, rank + 1, 0, 1);
    }
    return fabric;
}

SyntheticFabric make_turning_ring(uint32_t size) {
    SyntheticFabric fabric;
    for (uint32_t rank = 0; rank < size; ++rank) {
        const uint32_t next = (rank + 1) % size;
        // Deliberately alternate physical directions. Logical adjacency does
        // not imply a straight physical row or column.
        fabric.connect_bidirectional(rank, next, rank % 4, (rank + 2) % 4);
    }
    return fabric;
}

TEST(AllGatherNeighborRoutePlan, TwoRankLineHasOneLiveNeighborPerEndpoint) {
    const auto fabric = make_line(2);
    const auto plan = build_plan(1, Topology::Linear, 2, logical_group(2), fabric);

    expect_terminal_one_hop_edges(plan);
    ASSERT_EQ(plan.edges.size(), 2u);
    EXPECT_EQ(
        std::count_if(plan.edges.begin(), plan.edges.end(), [](const auto& edge) { return edge.source_rank == 0; }), 1);
    EXPECT_EQ(
        std::count_if(plan.edges.begin(), plan.edges.end(), [](const auto& edge) { return edge.source_rank == 1; }), 1);
}

TEST(AllGatherNeighborRoutePlan, FourAndEightRankLinesKeepEndpointsSingleSided) {
    for (const uint32_t size : {4u, 8u}) {
        const auto fabric = make_line(size);
        const auto plan = build_plan(0, Topology::Linear, 2, logical_group(size), fabric);

        expect_terminal_one_hop_edges(plan);
        ASSERT_EQ(plan.edges.size(), 2u * (size - 1));
        for (uint32_t rank = 0; rank < size; ++rank) {
            const auto live_directions = std::count_if(
                plan.edges.begin(), plan.edges.end(), [rank](const auto& edge) { return edge.source_rank == rank; });
            EXPECT_EQ(live_directions, rank == 0 || rank + 1 == size ? 1 : 2);
        }
    }
}

TEST(AllGatherNeighborRoutePlan, EightRankRingIncludesDirectWrapInBothDirections) {
    const auto fabric = make_turning_ring(8);
    const auto plan = build_plan(0, Topology::Ring, 2, logical_group(8), fabric);

    expect_terminal_one_hop_edges(plan);
    ASSERT_EQ(plan.edges.size(), 16u);
    EXPECT_EQ(
        std::count_if(
            plan.edges.begin(),
            plan.edges.end(),
            [](const auto& edge) {
                return (edge.source_rank == 0 && edge.destination_rank == 7) ||
                       (edge.source_rank == 7 && edge.destination_rank == 0);
            }),
        2);
}

TEST(AllGatherNeighborRoutePlan, NonDirectWrapRejectsNeighborBackend) {
    const auto fabric = make_line(8);
    const auto plan = build_plan(0, Topology::Ring, 2, logical_group(8), fabric);

    EXPECT_FALSE(plan.eligible);
    EXPECT_EQ(std::count_if(plan.edges.begin(), plan.edges.end(), [](const auto& edge) { return !edge.direct; }), 2);
    EXPECT_EQ(
        std::count_if(plan.edges.begin(), plan.edges.end(), [](const auto& edge) { return edge.packet_hops == 0; }), 2);
}

TEST(AllGatherNeighborRoutePlan, MissingRequestedLinkRejectsNeighborBackend) {
    SyntheticFabric fabric;
    fabric.connect_bidirectional(0, 1, 0, 1, {0});
    const auto plan = build_plan(1, Topology::Linear, 2, logical_group(2), fabric);

    EXPECT_FALSE(plan.eligible);
    ASSERT_EQ(plan.edges.size(), 2u);
    EXPECT_TRUE(std::all_of(plan.edges.begin(), plan.edges.end(), [](const auto& edge) { return edge.direct; }));
}

TEST(AllGatherNeighborRoutePlan, PhysicalTurnsRemainIndependentTerminalPackets) {
    const auto fabric = make_turning_ring(8);
    const auto plan = build_plan(0, Topology::Ring, 2, logical_group(8), fabric);

    expect_terminal_one_hop_edges(plan);
    std::set<uint32_t> directions;
    for (const auto& edge : plan.edges) {
        ASSERT_TRUE(edge.physical_direction.has_value());
        directions.insert(*edge.physical_direction);
    }
    EXPECT_GT(directions.size(), 1u);
}

TEST(AllGatherNeighborRoutePlan, RouteSensitiveProgramKeyChangesForEveryStructuralInput) {
    const auto fabric = make_turning_ring(8);
    const auto base_plan = build_plan(0, Topology::Ring, 2, logical_group(8), fabric);
    const auto axis_plan = build_plan(1, Topology::Ring, 2, logical_group(8), fabric);
    const auto topology_plan = build_plan(0, Topology::Linear, 2, logical_group(8), fabric);
    const auto links_plan = build_plan(0, Topology::Ring, 1, logical_group(8), fabric);
    auto remapped_nodes = logical_group(8);
    std::swap(remapped_nodes[1], remapped_nodes[2]);
    SyntheticFabric remapped_fabric;
    for (uint32_t rank = 0; rank < remapped_nodes.size(); ++rank) {
        const auto& source = remapped_nodes[rank];
        const auto& destination = remapped_nodes[(rank + 1) % remapped_nodes.size()];
        remapped_fabric.connect_bidirectional(
            source.chip_id, destination.chip_id, rank % 4, static_cast<uint32_t>((rank + 2) % 4));
    }
    const auto remapped_plan = build_plan(0, Topology::Ring, 2, remapped_nodes, remapped_fabric);

    ASSERT_TRUE(base_plan.eligible);
    ASSERT_TRUE(axis_plan.eligible);
    ASSERT_TRUE(topology_plan.eligible);
    ASSERT_TRUE(links_plan.eligible);
    ASSERT_TRUE(remapped_plan.eligible);

    EXPECT_NE(base_plan.to_hash(), axis_plan.to_hash());
    EXPECT_NE(base_plan.to_hash(), topology_plan.to_hash());
    EXPECT_NE(base_plan.to_hash(), links_plan.to_hash());
    EXPECT_NE(base_plan.to_hash(), remapped_plan.to_hash());

    ttnn::operations::ccl::AllGatherParams base;
    base.fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D;
    base.axis_topology = {Topology::Ring, Topology::Linear};
    base.axis_num_devices = {8, 1};
    base.axis_num_links = {2, 0};
    base.num_devices = 8;
    base.neighbor_unicast_eligible = true;
    base.neighbor_route_plan_hash = base_plan.to_hash();

    const auto hash_key = [](const auto& params) {
        return ttsl::hash::hash_objects_with_default_seed(params.routing_cache_key());
    };
    const auto base_hash = hash_key(base);

    auto changed = base;
    changed.fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X;
    EXPECT_NE(base_hash, hash_key(changed));
    changed = base;
    changed.axis_topology = {Topology::Linear, Topology::Linear};
    EXPECT_NE(base_hash, hash_key(changed));
    changed = base;
    changed.axis_num_devices = {1, 8};
    EXPECT_NE(base_hash, hash_key(changed));
    changed = base;
    changed.axis_num_links = {1, 0};
    EXPECT_NE(base_hash, hash_key(changed));
    changed = base;
    changed.neighbor_route_plan_hash = remapped_plan.to_hash();
    EXPECT_NE(base_hash, hash_key(changed));
}

}  // namespace
