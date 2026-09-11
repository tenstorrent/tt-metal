// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/// CPU-only PipelineBuilder prefetch: resolve_graph_layout + socket-endpoint validation.
///
/// Mirrors the tt-blaze PipelineGraph.build_topology / build_topology_multimesh path
/// (pipeline_builder/graph.py) without opening a MeshDevice or creating MeshSockets:
///
///   * Every MGD host-rank slice becomes one pipeline stage submesh, using the slice's
///     NATIVE shape (e.g. 4x2 on a 2x2-host mesh, 1x2 on a 4x4-host mesh).  This is the
///     same set of submeshes blaze gets from mesh_device.create_submeshes(stage_shape),
///     one submesh per MPI rank.
///   * A single linear loopback ring (s0->s1->...->sN->s0) over ALL submeshes, in MGD
///     order, is handed to the same C++ resolver blaze calls (resolve_graph_layout).
///     The resolver discovers the physical stage ordering via topological sort +
///     backtracking — we do not impose an ordering.
///   * For a single-mesh MGD this is the uniform-shape build_topology case; for a
///     multi-mesh MGD the ring spans meshes with heterogeneous per-mesh shapes
///     (e.g. 8x 4x2 on M0 + 32x 1x2 on M1), the build_topology_multimesh case.
///
/// The resolved layout is then validated the way the silicon pipeline relies on it:
/// active fabric eth channels on every socket chip, and a direct PSD ethernet link +
/// matching fabric hop per edge.  A chip MAY serve as more than one socket endpoint —
/// including a stage whose entry and exit land on the same chip — because blaze's
/// PipelineBlock places a colliding exit-send kernel on SECOND_PIPELINE_CORE_COORD (a
/// second core on that chip, see blaze/models/pipeline_block.py).  Same-chip / reused
/// socket endpoints are therefore valid and are NOT rejected here.

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include "fabric_fixture.hpp"
#include "utils.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshShape;

/// One pipeline stage submesh: a single MGD host-rank slice at its native shape.
struct SubmeshLayout {
    MeshId mesh_id;
    MeshShape rank_shape;
    std::vector<ChipTuple> chips;
};

/// Build one stage submesh per MGD host-rank slice, in MGD order (mesh-major,
/// host-coord row-major within each mesh).  Mirrors the per-rank submeshes blaze
/// gets from create_submeshes(native_stage_shape): uniform within a single-mesh MGD,
/// heterogeneous across a multi-mesh MGD.
std::vector<SubmeshLayout> build_submesh_layouts_from_mgd(const MeshGraph& mesh_graph) {
    auto mesh_ids = mesh_graph.get_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end());

    std::vector<SubmeshLayout> layouts;
    for (const MeshId mesh_id : mesh_ids) {
        for (const auto& [host_coord, host_rank] : mesh_graph.get_host_ranks(mesh_id)) {
            const MeshShape rank_shape = mesh_graph.get_mesh_shape(mesh_id, host_rank);
            SubmeshLayout layout{mesh_id, rank_shape, {}};
            layout.chips.reserve(rank_shape.mesh_size());

            for (uint32_t row = 0; row < rank_shape[0]; ++row) {
                for (uint32_t col = 0; col < rank_shape[1]; ++col) {
                    const MeshCoordinate local_coord(row, col);
                    const auto chip_id = mesh_graph.coordinate_to_chip(mesh_id, local_coord, host_rank);
                    layout.chips.emplace_back(*mesh_id, chip_id, row, col);
                }
            }
            layouts.push_back(std::move(layout));
        }
    }
    return layouts;
}

std::vector<std::vector<ChipTuple>> to_submesh_chips(const std::vector<SubmeshLayout>& layouts) {
    std::vector<std::vector<ChipTuple>> submesh_chips;
    submesh_chips.reserve(layouts.size());
    for (const auto& layout : layouts) {
        submesh_chips.push_back(layout.chips);
    }
    return submesh_chips;
}

/// Linear loopback ring over all stages: s0->s1->...->s{N-1}->s0 (last edge is loopback).
std::vector<EdgeInputTuple> build_ring_edges(std::size_t num_stages) {
    std::vector<EdgeInputTuple> edges;
    edges.reserve(num_stages);
    for (std::size_t i = 0; i + 1 < num_stages; ++i) {
        edges.emplace_back(fmt::format("s{}", i), fmt::format("s{}", i + 1), false);
    }
    edges.emplace_back(fmt::format("s{}", num_stages - 1), "s0", true);
    return edges;
}

// Declaration-ordered node names for the same ring. resolve_graph_layout takes the node
// list as authoritative, so every endpoint build_ring_edges() references must appear here.
std::vector<std::string> build_ring_nodes(std::size_t num_stages) {
    std::vector<std::string> nodes;
    nodes.reserve(num_stages);
    for (std::size_t i = 0; i < num_stages; ++i) {
        nodes.emplace_back(fmt::format("s{}", i));
    }
    return nodes;
}

FabricNodeId fabric_node_at_local_coord(
    const std::vector<SubmeshLayout>& layouts, std::size_t submesh_idx, uint32_t local_row, uint32_t local_col) {
    const auto& chips = layouts.at(submesh_idx).chips;
    for (const auto& [mesh_id, chip_id, row, col] : chips) {
        if (row == local_row && col == local_col) {
            return FabricNodeId{MeshId{mesh_id}, chip_id};
        }
    }
    return FabricNodeId{layouts.at(submesh_idx).mesh_id, 0};
}

// Returns first validation error, or nullopt if all checks pass.
std::optional<std::string> validate_pipeline_builder_graph_layout_errors(
    const ControlPlane& control_plane,
    const std::vector<SubmeshLayout>& layouts,
    const GraphLayoutResult& result,
    std::optional<size_t> expected_edge_count = std::nullopt) {
    const std::size_t num_stages = result.stage_order.size();

    // (1) Structural shape: the requested edges and one submesh assignment per stage.
    if (result.resolved_edges.size() != expected_edge_count.value_or(num_stages)) {
        return fmt::format("unexpected resolved_edges size {}", result.resolved_edges.size());
    }
    if (result.node_to_submesh.size() != num_stages) {
        return fmt::format("node_to_submesh size {} != stage count {}", result.node_to_submesh.size(), num_stages);
    }

    // Is there a physical ethernet cable between the two ASICs (either direction)?
    auto psd_has_direct_eth_link = [&](const FabricNodeId& a, const FabricNodeId& b) {
        const auto& psd = control_plane.get_physical_system_descriptor();
        const auto asic_a = control_plane.get_asic_id_from_fabric_node_id(a);
        const auto asic_b = control_plane.get_asic_id_from_fabric_node_id(b);
        if (!psd.get_eth_connections(asic_a, asic_b).empty()) {
            return true;
        }
        return !psd.get_eth_connections(asic_b, asic_a).empty();
    };

    // Link kinds: planar N/S/E/W (the in-mesh torus) and Z (the stacking dimension). A real
    // hop uses the same kind on both ends; pairing planar with Z means a cross-wired hop.
    // Kind != intra/inter: inter-mesh links may be planar or Z, and intra-mesh is planar
    // today but will gain Z links soon.
    using EthDir = tt::tt_fabric::eth_chan_directions;
    auto is_z_eth_dir = [](EthDir d) { return d == EthDir::Z; };
    auto is_nesw_eth_dir = [](EthDir d) {
        return d == EthDir::NORTH || d == EthDir::SOUTH || d == EthDir::EAST || d == EthDir::WEST;
    };
    auto eth_dirs_match_kind = [&](EthDir a, EthDir b) {
        return (is_z_eth_dir(a) && is_z_eth_dir(b)) || (is_nesw_eth_dir(a) && is_nesw_eth_dir(b));
    };

    // (2) Per-edge: confirm the two socket chips can actually exchange data over the fabric.
    for (const auto& edge : result.resolved_edges) {
        const std::size_t src_sub = result.node_to_submesh.at(edge.src);
        const std::size_t dst_sub = result.node_to_submesh.at(edge.dst);
        const FabricNodeId exit_fn = fabric_node_at_local_coord(layouts, src_sub, edge.exit_row, edge.exit_col);
        const FabricNodeId entry_fn = fabric_node_at_local_coord(layouts, dst_sub, edge.entry_row, edge.entry_col);

        // (2a) Both socket chips must have active fabric ethernet channels.
        if (control_plane.get_active_fabric_eth_channels(exit_fn).empty()) {
            return fmt::format("Edge exit {} has no active fabric ethernet channels", exit_fn);
        }
        if (control_plane.get_active_fabric_eth_channels(entry_fn).empty()) {
            return fmt::format("Edge entry {} has no active fabric ethernet channels", entry_fn);
        }
        // (2b) The exit and entry chips must be directly wired by a physical cable.
        if (!psd_has_direct_eth_link(exit_fn, entry_fn)) {
            return fmt::format("No direct PSD ethernet edge for {} -> {}", exit_fn, entry_fn);
        }

        // (2c) An active fabric hop must run from exit to entry (saw_hop), with both ends on
        // the same link kind (the direction check catches a cross-wired hop).
        bool saw_hop = false;
        for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
            auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
            if (peer_fn != entry_fn) {
                continue;
            }
            saw_hop = true;
            const EthDir dst_dir = control_plane.get_eth_chan_direction(peer_fn, static_cast<int>(peer_chan));
            if (!eth_dirs_match_kind(src_dir, dst_dir)) {
                return fmt::format("Direction mismatch {} -> {}", exit_fn, entry_fn);
            }
        }
        if (!saw_hop) {
            return fmt::format("No fabric hop from exit {} to entry {}", exit_fn, entry_fn);
        }
    }

    // (3) Stage-0 host-IO sockets (H2D/D2H) must also have active fabric ethernet channels.
    // No chip-distinctness needed: they may share a chip with each other or an edge, since
    // PipelineBlock separates colliding sockets by core.
    const std::size_t stage0_sub = result.node_to_submesh.at(result.stage_order.front());
    const FabricNodeId h2d_fn =
        fabric_node_at_local_coord(layouts, stage0_sub, result.h2d_entry_row, result.h2d_entry_col);
    const FabricNodeId d2h_fn =
        fabric_node_at_local_coord(layouts, stage0_sub, result.d2h_exit_row, result.d2h_exit_col);

    if (control_plane.get_active_fabric_eth_channels(h2d_fn).empty()) {
        return fmt::format("H2D entry {} has no active fabric ethernet channels", h2d_fn);
    }
    if (control_plane.get_active_fabric_eth_channels(d2h_fn).empty()) {
        return fmt::format("D2H exit {} has no active fabric ethernet channels", d2h_fn);
    }

    // Same-chip entry/exit per stage is intentionally allowed: PipelineBlock moves the
    // exit-send kernel to SECOND_PIPELINE_CORE_COORD (blaze/models/pipeline_block.py), so
    // the two kernels never share a core. No distinct entry/exit chip requirement.

    return std::nullopt;
}

std::string describe_layouts(const std::vector<SubmeshLayout>& layouts) {
    std::string desc;
    for (std::size_t i = 0; i < layouts.size(); ++i) {
        desc += fmt::format("{}s{}=M{}:{}", i == 0 ? "" : " ", i, *layouts[i].mesh_id, layouts[i].rank_shape);
    }
    return desc;
}

}  // namespace

TEST(PipelineBuilderLayoutTest, SmallStageDefaultsToTwoSlots) {
    const std::vector<std::string> nodes{"s0"};
    const std::vector<EdgeInputTuple> edges;
    const std::vector<std::vector<ChipTuple>> submesh_chips{{{0, 0, 0, 0}}};

    const GraphLayoutResult result = resolve_graph_layout(nodes, edges, submesh_chips);

    ASSERT_EQ(result.stage_order, nodes);
    EXPECT_EQ(result.node_to_submesh.at("s0"), 0);
    EXPECT_TRUE(result.resolved_edges.empty());
    ASSERT_TRUE(result.h2d_core_slot.has_value());
    ASSERT_TRUE(result.d2h_core_slot.has_value());
    EXPECT_EQ(*result.h2d_core_slot, 0u);
    EXPECT_EQ(*result.d2h_core_slot, 1u);
}

// These focused capacity tests use one synthetic submesh, so connection discovery
// never queries the control plane and no device or fabric initialization is needed.
TEST(PipelineBuilderCapacityTest, SpreadsHostEndpointsAcrossAvailableChips) {
    const std::vector<std::string> nodes{"s0"};
    const std::vector<EdgeInputTuple> edges;
    const std::vector<std::vector<ChipTuple>> submesh_chips{{{0, 0, 0, 0}, {0, 1, 0, 1}}};
    const std::map<std::string, uint32_t> capacities{{"s0", 1}};

    const GraphLayoutResult result = resolve_graph_layout(nodes, edges, submesh_chips, {}, capacities);

    EXPECT_EQ(result.node_to_submesh.at("s0"), 0);
    EXPECT_TRUE(result.resolved_edges.empty());
    EXPECT_NE(std::tie(result.h2d_entry_row, result.h2d_entry_col), std::tie(result.d2h_exit_row, result.d2h_exit_col));
    ASSERT_TRUE(result.h2d_core_slot.has_value());
    ASSERT_TRUE(result.d2h_core_slot.has_value());
    EXPECT_EQ(*result.h2d_core_slot, 0);
    EXPECT_EQ(*result.d2h_core_slot, 0);
}

TEST(PipelineBuilderCapacityTest, FoldsHostEndpointsIntoDistinctSlots) {
    const std::vector<std::string> nodes{"s0"};
    const std::vector<EdgeInputTuple> edges;
    const std::vector<std::vector<ChipTuple>> submesh_chips{{{0, 0, 0, 0}}};
    const std::map<std::string, uint32_t> capacities{{"s0", 2}};

    const GraphLayoutResult result = resolve_graph_layout(nodes, edges, submesh_chips, {}, capacities);

    EXPECT_EQ(std::tie(result.h2d_entry_row, result.h2d_entry_col), std::tie(result.d2h_exit_row, result.d2h_exit_col));
    ASSERT_TRUE(result.h2d_core_slot.has_value());
    ASSERT_TRUE(result.d2h_core_slot.has_value());
    EXPECT_EQ(*result.h2d_core_slot, 0);
    EXPECT_EQ(*result.d2h_core_slot, 1);
}

TEST(PipelineBuilderCapacityTest, RejectsInsufficientHostEndpointCapacity) {
    const std::vector<std::string> nodes{"s0"};
    const std::vector<EdgeInputTuple> edges;
    const std::vector<std::vector<ChipTuple>> submesh_chips{{{0, 0, 0, 0}}};
    const std::map<std::string, uint32_t> capacities{{"s0", 1}};

    try {
        static_cast<void>(resolve_graph_layout(nodes, edges, submesh_chips, {}, capacities));
        FAIL() << "expected capacity exhaustion";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("exact placement/link search exhausted"), std::string::npos);
        EXPECT_NE(message.find("example capacity conflict:"), std::string::npos);
        EXPECT_NE(message.find("stage 's0'"), std::string::npos);
        EXPECT_NE(message.find("needs 2 slots but has capacity 1"), std::string::npos);
        EXPECT_NE(message.find("chip (0,0)"), std::string::npos);
        EXPECT_EQ(message.find("endpoint roles"), std::string::npos);
    }
}

TEST(PipelineBuilderCapacityTest, RejectsUnknownOrZeroStageCapacity) {
    const std::vector<std::string> nodes{"s0"};
    const std::vector<EdgeInputTuple> edges;
    const std::vector<std::vector<ChipTuple>> submesh_chips{{{0, 0, 0, 0}}};

    EXPECT_THROW(
        static_cast<void>(resolve_graph_layout(nodes, edges, submesh_chips, {}, {{"other_stage", 1}})),
        std::runtime_error);
    EXPECT_THROW(
        static_cast<void>(resolve_graph_layout(nodes, edges, submesh_chips, {}, {{"s0", 0}})), std::runtime_error);
}


TEST(PipelineBuilderCapacityTest, SearchRejectsOverlappingSubmeshDomains) {
    const std::vector<std::string> nodes{"a", "b", "c"};
    const std::vector<std::vector<ChipTuple>> chips{
        {{0, 0, 0, 0}, {0, 1, 0, 1}},
        {{0, 2, 0, 0}, {0, 3, 0, 1}},
        {{0, 4, 0, 0}, {0, 5, 0, 1}, {0, 6, 0, 2}}};
    const detail::DirectLinks links;
    // Each stage has candidates, but all three require the same two submeshes.
    EXPECT_THROW(detail::resolve_graph_layout_with_connections(
        nodes, {}, chips, {{"a", 2}, {"b", 2}, {"c", 2}}, {}, 2, &links), std::runtime_error);
    EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(
        nodes, {}, chips, {{"a", 2}, {"b", 2}, {"c", 3}}, {}, 2, &links));
}

TEST(PipelineBuilderCapacityTest, LinkSearchRejectsOversubscribedFork) {
    const auto nodes = build_ring_nodes(4);
    const std::vector<EdgeInputTuple> edges{
        {nodes[0], nodes[1], false}, {nodes[0], nodes[2], false}, {nodes[0], nodes[3], false}};
    std::vector<std::vector<ChipTuple>> chips(4);
    detail::DirectLinks links;
    for (size_t i = 0; i < chips.size(); ++i) {
        chips[i] = {{0, static_cast<uint32_t>(2 * i), 0, 0},
                    {0, static_cast<uint32_t>(2 * i + 1), 0, 1}};
        for (size_t j = 0; j < chips.size(); ++j) {
            if (i != j) { links[{i, j}] = {{0, 0, 0, 0}}; }
        }
    }
    // Connectivity and placement work, but all three outgoing roles need the
    // same chip. The total slot count fits, so link selection must reject it.
    EXPECT_THROW(detail::resolve_graph_layout_with_connections(
        nodes, edges, chips, {}, {}, 2, &links), std::runtime_error);
    EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(
        nodes, edges, chips, {}, {}, 3, &links));
}

TEST(PipelineBuilderCapacityTest, UniformCapacityWithOverrides) {
    const std::vector<std::string> nodes{"s0"};
    const std::vector<std::vector<ChipTuple>> chips{{{0, 0, 0, 0}}};
    auto result = resolve_graph_layout(nodes, {}, chips, {}, {}, 2);
    EXPECT_EQ(result.h2d_core_slot, 0);
    EXPECT_EQ(result.d2h_core_slot, 1);
    EXPECT_THROW(resolve_graph_layout(nodes, {}, chips, {}, {{"s0", 1}}, 2), std::runtime_error);
    EXPECT_NO_THROW(resolve_graph_layout(nodes, {}, chips, {}, {{"s0", 2}}, 1));
    EXPECT_THROW(resolve_graph_layout(nodes, {}, chips, {}, {}, 0), std::runtime_error);
}

TEST(PipelineBuilderCapacityTest, DefaultsAndOverridesRejectDisconnectedPlacement) {
    constexpr size_t count = 16;
    std::vector<std::vector<ChipTuple>> chips(count);
    detail::DirectLinks links;
    for (size_t i = 0; i < count; ++i) {
        chips[i] = {{0, static_cast<uint32_t>(2 * i), 0, 0},
                    {0, static_cast<uint32_t>(2 * i + 1), 0, 1}};
        for (size_t j = 0; j < count; ++j) {
            if (i != j && i / 8 == j / 8) {
                links[{i, j}] = {{0, 0, 0, 1}};
            }
        }
    }
    // Two complete eight-submesh components: a connected 16-stage ring cannot
    // fit with either default or explicit capacities.
    for (auto capacity : {std::optional<uint32_t>{}, std::optional<uint32_t>{2}}) {
        EXPECT_THROW(detail::resolve_graph_layout_with_connections(
            build_ring_nodes(count), build_ring_edges(count), chips, {}, {}, capacity, &links), std::runtime_error);
    }
}

TEST(PipelineBuilderCapacityTest, PrefersSeparateForwardingChipsAndLocalHostEndpoints) {
    const auto nodes = build_ring_nodes(3);
    const auto edges = build_ring_edges(3);
    std::vector<std::vector<ChipTuple>> chips(3);
    for (size_t i = 0; i < chips.size(); ++i) {
        chips[i] = {{0, static_cast<uint32_t>(2 * i), 0, 0},
                    {0, static_cast<uint32_t>(2 * i + 1), 0, 1}};
    }
    detail::DirectLinks links{
        {{0, 1}, {{0, 1, 0, 0}, {0, 1, 0, 1}}},
        {{1, 2}, {{0, 0, 0, 0}}},
        {{2, 0}, {{0, 1, 0, 0}}}};
    // The first incoming link would fold s1. Search must revisit that choice,
    // not accept sharing immediately just because two slots are available.
    for (auto capacity : {std::optional<uint32_t>{}, std::optional<uint32_t>{2}}) {
        const auto result = detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, capacity, &links);
        ASSERT_EQ(result.resolved_edges.size(), 3u);
        EXPECT_EQ(result.resolved_edges[0].entry_col, 1u);
        EXPECT_EQ(result.resolved_edges[1].exit_col, 0u);
        EXPECT_EQ(result.resolved_edges[1].entry_col, 0u);
        EXPECT_EQ(result.resolved_edges[2].exit_col, 1u);
        EXPECT_EQ(result.h2d_entry_col, 1u);
        EXPECT_EQ(result.d2h_exit_col, 0u);
        EXPECT_EQ(result.h2d_core_slot, 1u);
        EXPECT_EQ(result.d2h_core_slot, 1u);
    }
    // When separation is impossible, the same default must still permit sharing.
    links.at({0, 1}).resize(1);
    const auto folded = detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, 2, &links);
    ASSERT_EQ(folded.resolved_edges.size(), 3u);
    EXPECT_EQ(folded.resolved_edges[0].entry_col, folded.resolved_edges[1].exit_col);
    EXPECT_NE(folded.resolved_edges[0].entry_core_slot, folded.resolved_edges[1].exit_core_slot);
}

TEST(PipelineBuilderCapacityTest, DefaultsAllowColocatedEndpointsOnSmallStages) {
    auto nodes = build_ring_nodes(4);
    auto edges = build_ring_edges(4);
    std::vector<std::vector<ChipTuple>> chips(4);
    detail::DirectLinks links;
    for (size_t i = 0; i < 4; ++i) {
        chips[i] = {{0, static_cast<uint32_t>(2 * i), 0, 0},
                    {0, static_cast<uint32_t>(2 * i + 1), 0, 1}};
        links[{i, (i + 1) % 4}] = {{0, 0, 0, 0}};
    }
    // Connectivity succeeds immediately, but all entry/exit ports coincide.
    EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(
        nodes, edges, chips, {}, {}, std::nullopt, &links));
    EXPECT_THROW(detail::resolve_graph_layout_with_connections(
        nodes, edges, chips, {}, {}, 1, &links), std::runtime_error);
    const auto result = detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, 2, &links);
    ASSERT_EQ(result.resolved_edges.size(), 4);
    // Shuffled declaration order must still preserve output order.
    std::reverse(edges.begin(), edges.end());
    const auto reversed = detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, 2, &links);
    for (size_t i = 0; i < edges.size(); ++i) {
        EXPECT_EQ(reversed.resolved_edges[i].src, std::get<0>(edges[i]));
    }
}

// Two chains sharing a source exercise branching rather than just a path.
// Dense fabrics have many equivalent placements; sparse ones restrict
// adjacency. Coincident ports make every internal one-slot stage infeasible.
class PipelineBuilderForkTest : public ::testing::TestWithParam<bool> {};

TEST_P(PipelineBuilderForkTest, PlacesOrRejectsTwoChains) {
    const bool coincident_ports = GetParam();
    for (size_t count : {8u, 10u}) {
        for (bool dense : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "count=" << count << " dense=" << dense);
            const auto nodes = build_ring_nodes(count);
            std::vector<EdgeInputTuple> edges;
            std::vector<std::vector<ChipTuple>> chips(count);
            detail::DirectLinks links;
            for (size_t i = 0; i < count; ++i) {
                chips[i] = {{0, static_cast<uint32_t>(2 * i), 0, 0},
                            {0, static_cast<uint32_t>(2 * i + 1), 0, 1}};
                if (i > 0) {
                    edges.emplace_back(nodes[i == count / 2 ? 0 : i - 1], nodes[i], false);
                }
                for (size_t j = 0; j < count; ++j) {
                    if (i != j && (dense || j == i + 1 || (i == 0 && j == count / 2))) {
                        links[{i, j}] = {{0, 0, 0, coincident_ports ? 0u : 1u}};
                    }
                }
            }
            // The fork source needs two outgoing slots plus its two host roles.
            auto resolve = [&] {
                return detail::resolve_graph_layout_with_connections(
                    nodes, edges, chips, {}, {{nodes.front(), 4}}, 1, &links);
            };
            if (coincident_ports) {
                EXPECT_THROW(resolve(), std::runtime_error);
            } else {
                const auto result = resolve();
                EXPECT_EQ(result.node_to_submesh.size(), count);
                EXPECT_EQ(result.resolved_edges.size(), count - 1);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(PortCapacity, PipelineBuilderForkTest, ::testing::Bool());

TEST(PipelineBuilderCapacityTest, RetainsResolvedLinksAfterBacktracking) {
    const std::vector<std::string> nodes{"a", "b", "c"};
    const std::vector<std::vector<ChipTuple>> chips{
        {{0, 0, 0, 0}, {0, 1, 0, 1}, {0, 2, 0, 2}, {0, 3, 0, 3}},
        {{0, 4, 0, 0}, {0, 5, 0, 1}},
        {{0, 6, 0, 0}, {0, 7, 0, 1}}};
    const detail::DirectLinks links{
        {{0, 1}, {{0, 0, 0, 0}, {0, 0, 0, 1}}},
        {{1, 2}, {{0, 0, 0, 0}}},
        {{2, 0}, {{0, 1, 0, 0}, {0, 1, 0, 1}}}};
    for (bool ring : {false, true}) {
        // Deliberately not in traversal order. The first a->b link reaches b
        // on its outgoing chip and must be undone before the second succeeds.
        // The first closing link also conflicts with a's outgoing endpoint.
        std::vector<EdgeInputTuple> edges{{"b", "c", false}, {"a", "a", true}, {"a", "b", false}};
        if (ring) {
            edges.insert(edges.begin(), {"c", "a", true});
        }
        const auto result = detail::resolve_graph_layout_with_connections(
            nodes, edges, chips, {{"a", 4}, {"b", 2}, {"c", 2}}, {}, 1, &links);
        ASSERT_EQ(result.resolved_edges.size(), ring ? 3u : 2u);
        EXPECT_EQ(result.node_to_submesh.at("a"), 0u);
        EXPECT_EQ(result.node_to_submesh.at("b"), 1u);
        EXPECT_EQ(result.node_to_submesh.at("c"), 2u);
        std::set<std::tuple<std::string, uint32_t, uint32_t, uint32_t>> slots;
        size_t output_index = 0;
        for (const auto& [src, dst, loopback] : edges) {
            if (src == dst) {
                continue;
            }
            const auto& resolved = result.resolved_edges.at(output_index++);
            EXPECT_EQ(resolved.src, src);
            EXPECT_EQ(resolved.dst, dst);
            EXPECT_EQ(resolved.is_loopback, loopback);
            ASSERT_TRUE(resolved.exit_core_slot.has_value());
            ASSERT_TRUE(resolved.entry_core_slot.has_value());
            EXPECT_EQ(*resolved.exit_core_slot, 0u);
            EXPECT_EQ(*resolved.entry_core_slot, 0u);
            EXPECT_TRUE(slots.emplace(src, resolved.exit_row, resolved.exit_col, *resolved.exit_core_slot).second);
            EXPECT_TRUE(slots.emplace(dst, resolved.entry_row, resolved.entry_col, *resolved.entry_core_slot).second);
            EXPECT_EQ(resolved.exit_col, src == "c" ? 1u : 0u);
            EXPECT_EQ(resolved.entry_col, dst == "c" ? 0u : 1u);
        }
        ASSERT_TRUE(result.h2d_core_slot.has_value());
        ASSERT_TRUE(result.d2h_core_slot.has_value());
        EXPECT_EQ(*result.h2d_core_slot, 0u);
        EXPECT_EQ(*result.d2h_core_slot, 0u);
        EXPECT_TRUE(slots.emplace("a", result.h2d_entry_row, result.h2d_entry_col, *result.h2d_core_slot).second);
        EXPECT_TRUE(slots.emplace("a", result.d2h_exit_row, result.d2h_exit_col, *result.d2h_core_slot).second);
    }
}

TEST(PipelineBuilderCapacityTest, RejectsIncompatibleSharedLinkChoices) {
    const std::vector<std::string> nodes{"a", "b", "c", "d"};
    const std::vector<EdgeInputTuple> edges{{"a", "b", false}, {"b", "c", false}, {"c", "d", false}};
    const std::map<std::string, uint32_t> shapes{{"a", 4}, {"b", 2}, {"c", 3}, {"d", 5}};
    std::vector<std::vector<ChipTuple>> chips;
    for (uint32_t count : {4u, 2u, 3u, 5u}) {
        auto& mesh = chips.emplace_back();
        for (uint32_t col = 0; col < count; ++col) {
            mesh.emplace_back(0, static_cast<uint32_t>(chips.size() * 8 + col), 0, col);
        }
    }
    detail::DirectLinks links{
        {{0, 1}, {{0, 0, 0, 0}}},
        {{1, 2}, {{0, 0, 0, 0}, {0, 1, 0, 1}}},
        {{2, 3}, {{0, 1, 0, 0}}}};
    // b needs the second middle link, while c needs the first. Both fit
    // independently, but no shared link choice satisfies both stages.
    EXPECT_THROW(detail::resolve_graph_layout_with_connections(
        nodes, edges, chips, shapes, {}, 1, &links), std::runtime_error);
    links[{2, 3}] = {{0, 0, 0, 0}};
    EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(
        nodes, edges, chips, shapes, {}, 1, &links));
}

TEST(PipelineBuilderLayoutTest, RejectsInvalidInputsBeforeDiscovery) {
    // Two submeshes would access the control plane if discovery ran first.
    const std::vector<std::vector<ChipTuple>> chips{{{0, 0, 0, 0}}, {{0, 1, 0, 0}}};
    auto rejects = [&](const std::vector<std::string>& nodes, const std::vector<EdgeInputTuple>& edges,
                       const std::map<std::string, uint32_t>& capacities, const std::string& message) {
        try {
            resolve_graph_layout(nodes, edges, chips, {}, capacities);
            FAIL() << "expected validation error";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(message), std::string::npos);
        }
    };
    rejects({}, {}, {}, "nodes must not be empty");
    rejects({"a", "a"}, {}, {}, "duplicate stage names");
    rejects({"a"}, {{"a", "missing", false}}, {}, "not found in the explicit nodes list");
    rejects({"a", "b"}, {{"a", "b", false}, {"b", "a", false}}, {}, "cycle detected");
    rejects({"a", "b"}, {}, {{"missing", 1}}, "capacity override for unknown stage");
    rejects({"a"}, {}, {{"a", 0}}, "zero pipeline-core capacity");
}

TEST(PipelineBuilderLayoutTest, RejectsUnknownChipCountStageBeforeDiscovery) {
    // A misspelled constraint must not be silently ignored or reach discovery.
    const std::vector<std::vector<ChipTuple>> chips{{{0, 0, 0, 0}}, {{0, 1, 0, 0}}};
    try {
        resolve_graph_layout({"a", "b"}, {{"a", "b", false}}, chips, {{"missing", 8}});
        FAIL() << "expected validation error";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(
            std::string(error.what()).find("chip-count override for unknown stage 'missing'"), std::string::npos);
    }
}

TEST(PipelineBuilderLayoutTest, RetriesLinksAfterCapacityConflict) {
    const std::vector<std::string> nodes{"a", "b", "c", "d"};
    const std::vector<EdgeInputTuple> edges{{"a", "b", false}, {"b", "c", false}, {"c", "d", false}};
    std::vector<std::vector<ChipTuple>> chips(4);
    const std::map<std::string, uint32_t> shapes{{"a", 4}, {"b", 2}, {"c", 3}, {"d", 5}};
    for (size_t m = 0; m < chips.size(); ++m) {
        for (uint32_t c = 0; c < shapes.at(nodes[m]); ++c) {
            chips[m].emplace_back(0, static_cast<uint32_t>(5 * m + c), 0, c);
        }
    }
    const detail::DirectLinks links{
        {{0, 1}, {{0, 0, 0, 0}}},
        {{1, 2}, {{0, 1, 0, 0}, {0, 0, 0, 1}}},
        {{2, 3}, {{0, 0, 0, 0}}}};
    // The old greedy repair could reintroduce a conflict at b. Defaults now
    // use joint search and assign separate slots for colocated endpoints.
    EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, shapes, {}, std::nullopt, &links));
    EXPECT_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, shapes, {}, 1, &links), std::runtime_error);
    EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, shapes, {}, 2, &links));
}

TEST(PipelineBuilderCapacityTest, DefaultCapacityBoundaryAndOverrides) {
    const auto nodes = build_ring_nodes(2);
    const auto edges = build_ring_edges(2);
    const detail::DirectLinks links{{{0, 1}, {{0, 0, 0, 0}}}, {{1, 0}, {{0, 0, 0, 0}}}};
    for (uint32_t chip_count : {2u, 7u, 8u, 9u}) {
        SCOPED_TRACE(chip_count);
        std::vector<std::vector<ChipTuple>> chips(2);
        for (size_t m = 0; m < chips.size(); ++m) {
            for (uint32_t c = 0; c < chip_count; ++c) {
                chips[m].emplace_back(0, static_cast<uint32_t>(9 * m + c), 0, c);
            }
        }
        if (chip_count < 8) {
            EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, std::nullopt, &links));
        } else {
            EXPECT_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, std::nullopt, &links), std::runtime_error);
        }
        EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, 2, &links));
        EXPECT_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {{"s1", 1}}, 2, &links), std::runtime_error);
    }
}

TEST(PipelineBuilderCapacityTest, DefaultCapacityFollowsCandidateSubmesh) {
    const std::vector<std::string> nodes{"a", "b", "c"};
    const std::vector<EdgeInputTuple> edges{{"a", "b", false}, {"b", "c", false}};
    std::vector<std::vector<ChipTuple>> chips(4);
    const std::array<uint32_t, 4> sizes{4, 8, 2, 3};
    for (size_t m = 0; m < chips.size(); ++m) {
        for (uint32_t c = 0; c < sizes[m]; ++c) {
            chips[m].emplace_back(0, static_cast<uint32_t>(8 * m + c), 0, c);
        }
    }
    const detail::DirectLinks links{
        {{0, 1}, {{0, 0, 0, 0}}}, {{0, 2}, {{0, 0, 0, 0}}},
        {{1, 3}, {{0, 0, 0, 0}}}, {{2, 3}, {{0, 0, 0, 0}}}};
    const std::map<std::string, uint32_t> shapes{{"a", 4}, {"c", 3}};
    // b has no declared shape: the first (8-chip) candidate has only one slot
    // per chip, so search must backtrack to the smaller two-slot candidate.
    auto result = detail::resolve_graph_layout_with_connections(nodes, edges, chips, shapes, {}, std::nullopt, &links);
    EXPECT_EQ(result.node_to_submesh.at("b"), 2u);
    // A partial override changes b only; a and c still use their size defaults.
    result = detail::resolve_graph_layout_with_connections(nodes, edges, chips, shapes, {{"b", 2}}, std::nullopt, &links);
    EXPECT_EQ(result.node_to_submesh.at("b"), 1u);
}

TEST(PipelineBuilderCapacityTest, PlacementMatchesBruteForce) {
    // Independent oracle: enumerate injective placements, then every link
    // combination. Count endpoint roles directly; no production pruning/DP.
    uint32_t random_state = 12345;
    auto random = [&]() {
        random_state = random_state * 1664525u + 1013904223u;
        return random_state >> 16;
    };
    for (size_t sample = 0; sample < 800; ++sample) {
        std::vector<std::string> nodes{"a", "b", "c"};
        if (sample >= 400) { nodes.push_back("d"); }
        SCOPED_TRACE(sample);
        const size_t mesh_count = nodes.size() + sample % 2;
        std::vector<std::vector<ChipTuple>> chips(mesh_count);
        const std::array<uint32_t, 5> chip_counts{1, 2, 7, 8, 9};
        for (size_t m = 0; m < mesh_count; ++m) {
            for (uint32_t c = 0, count = chip_counts[random() % chip_counts.size()]; c < count; ++c) {
                chips[m].emplace_back(0, static_cast<uint32_t>(9 * m + c), 0, c);
            }
        }
        std::vector<EdgeInputTuple> edges;
        switch (sample % 5) {
            case 0: edges = {{"a", "b", false}, {"b", "c", false}}; break;
            case 1: edges = {{"a", "b", false}, {"b", "c", false}, {"c", "a", true}}; break;
            case 2: edges = {{"a", "b", false}, {"a", "c", false}}; break;
            case 3: edges = {{"a", "c", false}, {"b", "c", false}}; break;
            default: edges = {{"a", "b", false}, {"b", "c", true}, {"c", "a", true}}; break;
        }
        if (sample >= 400) {
            // Branching graphs: diamonds, merges with loopbacks, parallel edges,
            // and disconnected components. Non-loopback edges remain acyclic.
            edges.emplace_back("b", "d", false);
            if (sample % 2 == 0) { edges.emplace_back("c", "d", false); }
            if (sample % 3 == 0) { edges.emplace_back("d", "a", true); }
            if (sample % 7 == 0) { edges.emplace_back("a", "b", false); }
            if (sample % 11 == 0) { edges = {{"a", "b", false}, {"c", "d", false}}; }
            std::reverse(edges.begin(), edges.end());
        }
        detail::DirectLinks links;
        for (size_t src = 0; src < mesh_count; ++src) {
            for (size_t dst = 0; dst < mesh_count; ++dst) {
                if (src == dst) { continue; }
                for (size_t i = 0, count = random() % 4; i < count; ++i) {
                    links[{src, dst}].push_back({0, static_cast<uint32_t>(random() % chips[src].size()),
                                                0, static_cast<uint32_t>(random() % chips[dst].size())});
                }
            }
        }
        std::map<std::string, uint32_t> capacities;
        for (const auto& node : nodes) {
            if (sample % 3 == 0 || (sample % 3 == 1 && node == "b")) { capacities[node] = 1 + random() % 2; }
        }
        auto capacity = [&](const std::string& stage, size_t mesh) {
            return capacities.contains(stage) ? capacities.at(stage) : (chips[mesh].size() >= 8 ? 1u : 2u);
        };
        std::map<std::string, uint32_t> shapes;
        if (sample % 3 == 0) { shapes["b"] = chip_counts[random() % chip_counts.size()]; }
        std::map<std::string, size_t> placement;
        std::vector<bool> used(mesh_count);
        std::map<std::pair<std::string, uint32_t>, uint32_t> roles;
        std::function<bool(size_t)> choose_links = [&](size_t index) {
            if (index == edges.size()) {
                uint32_t free = 0;
                for (uint32_t c = 0; c < chips[placement.at("a")].size(); ++c) {
                    free += capacity("a", placement.at("a")) - roles[{"a", c}];
                }
                return free >= 2;
            }
            const auto& [src, dst, loopback] = edges[index];
            auto connection = links.find({placement.at(src), placement.at(dst)});
            if (connection == links.end()) { return false; }
            for (const auto& link : connection->second) {
                auto& exit = roles[{src, link.exit_col}];
                auto& entry = roles[{dst, link.entry_col}];
                if (exit == capacity(src, placement.at(src)) || entry == capacity(dst, placement.at(dst))) { continue; }
                ++exit; ++entry;
                const bool found = choose_links(index + 1);
                --exit; --entry;
                if (found) { return true; }
            }
            return false;
        };
        std::function<bool(size_t)> place = [&](size_t index) {
            if (index == nodes.size()) { return choose_links(0); }
            const auto& node = nodes[index];
            for (size_t m = 0; m < mesh_count; ++m) {
                if (used[m] || (shapes.contains(node) && shapes.at(node) != chips[m].size())) { continue; }
                used[m] = true;
                placement[node] = m;
                const bool found = place(index + 1);
                used[m] = false;
                if (found) { return true; }
            }
            return false;
        };
        const bool expected = place(0);
        bool actual = false;
        try {
            const auto result = detail::resolve_graph_layout_with_connections(
                nodes, edges, chips, shapes, capacities, std::nullopt, &links);
            actual = true;
            std::set<std::tuple<std::string, uint32_t, uint32_t, uint32_t>> occupied;
            auto check_slot = [&](const std::string& stage, uint32_t row, uint32_t col, std::optional<uint32_t> slot) {
                ASSERT_TRUE(slot.has_value());
                EXPECT_LT(*slot, capacity(stage, result.node_to_submesh.at(stage)));
                EXPECT_EQ(row, 0u);
                EXPECT_LT(col, chips[result.node_to_submesh.at(stage)].size());
                EXPECT_TRUE(occupied.emplace(stage, row, col, *slot).second);
            };
            std::set<size_t> assigned;
            for (const auto& [stage, mesh] : result.node_to_submesh) {
                EXPECT_TRUE(assigned.insert(mesh).second);
                if (shapes.contains(stage)) { EXPECT_EQ(chips[mesh].size(), shapes.at(stage)); }
            }
            ASSERT_EQ(result.resolved_edges.size(), edges.size());
            for (size_t i = 0; i < edges.size(); ++i) {
                const auto& edge = result.resolved_edges[i];
                EXPECT_EQ(edge.src, std::get<0>(edges[i]));
                EXPECT_EQ(edge.dst, std::get<1>(edges[i]));
                const auto& candidates = links.at({result.node_to_submesh.at(edge.src), result.node_to_submesh.at(edge.dst)});
                const detail::LinkPair selected{edge.exit_row, edge.exit_col, edge.entry_row, edge.entry_col};
                EXPECT_NE(std::find(candidates.begin(), candidates.end(), selected), candidates.end());
                check_slot(edge.src, edge.exit_row, edge.exit_col, edge.exit_core_slot);
                check_slot(edge.dst, edge.entry_row, edge.entry_col, edge.entry_core_slot);
            }
            check_slot("a", result.h2d_entry_row, result.h2d_entry_col, result.h2d_core_slot);
            check_slot("a", result.d2h_exit_row, result.d2h_exit_col, result.d2h_core_slot);
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("no valid submesh assignment"), std::string::npos);
        }
        EXPECT_EQ(actual, expected);
    }
}

class PipelineBuilderReplayTest : public ::testing::TestWithParam<uint32_t> {};

TEST_P(PipelineBuilderReplayTest, LlamaTopology) {
    // Captured from the CPU mock using the Llama 8B 1x2 pod MGD. Format:
    // submesh count; per-submesh chip count and (mesh, chip, row, col) tuples;
    // then (src submesh, dst submesh, exit row/col, entry row/col) links.
    std::ifstream input("tests/tt_metal/tt_fabric/test_data/llama_1x2_pipeline_connections.txt");
    ASSERT_TRUE(input) << "missing captured Llama topology";
    size_t count;
    ASSERT_TRUE(static_cast<bool>(input >> count));
    std::vector<std::vector<ChipTuple>> chips(count);
    for (auto& stage : chips) {
        size_t chip_count;
        ASSERT_TRUE(static_cast<bool>(input >> chip_count));
        for (size_t i = 0; i < chip_count; ++i) {
            uint32_t mesh, chip, row, col;
            ASSERT_TRUE(static_cast<bool>(input >> mesh >> chip >> row >> col));
            stage.emplace_back(mesh, chip, row, col);
        }
    }
    detail::DirectLinks links;
    size_t src, dst;
    detail::LinkPair link;
    while (input >> src >> dst >> link.exit_row >> link.exit_col >> link.entry_row >> link.entry_col) {
        links[{src, dst}].push_back(link);
    }
    const auto nodes = build_ring_nodes(count);
    const auto edges = build_ring_edges(count);
    const std::optional<uint32_t> capacity = GetParam() == 0 ? std::nullopt : std::optional{GetParam()};
    if (GetParam() != 1) {
        EXPECT_NO_THROW(detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, capacity, &links));
        return;
    }
    try {
        detail::resolve_graph_layout_with_connections(nodes, edges, chips, {}, {}, 1, &links);
        FAIL() << "expected exact infeasibility";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("exact placement/link search exhausted"), std::string::npos);
    }
}

INSTANTIATE_TEST_SUITE_P(CoreCapacity, PipelineBuilderReplayTest, ::testing::Values(0u, 2u, 1u));

// Opt-in benchmark: tt-run supplies the mock descriptor and MGD. Run each
// capacity in a separate process with an external timeout; timeout != infeasible.
TEST(PipelineBuilderMockSweep, RingCapacity) {
    const char* mock = std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    const char* requested_capacity = std::getenv("TT_PIPELINE_TEST_CORE_CAPACITY");
    if (mock == nullptr || *mock == '\0' || requested_capacity == nullptr) {
        GTEST_SKIP() << "requires mock cluster and TT_PIPELINE_TEST_CORE_CAPACITY (0=default)";
    }
    const std::string capacity_text(requested_capacity);
    ASSERT_TRUE(capacity_text == "0" || capacity_text == "1" || capacity_text == "2");
    const std::optional<uint32_t> capacity = capacity_text == "0"
        ? std::nullopt : std::optional<uint32_t>(std::stoul(capacity_text));
    const char* requested_fabric = std::getenv("TT_PIPELINE_TEST_FABRIC");
    const std::string fabric_mode = requested_fabric == nullptr ? "2D" : requested_fabric;
    ASSERT_TRUE(fabric_mode == "2D" || fabric_mode == "TORUS_XY" || fabric_mode == "TORUS_Y");
    const auto fabric_config = fabric_mode == "TORUS_XY" ? FabricConfig::FABRIC_2D_TORUS_XY
        : fabric_mode == "TORUS_Y" ? FabricConfig::FABRIC_2D_TORUS_Y : FabricConfig::FABRIC_2D;

    auto& context = tt::tt_metal::MetalContext::instance();
    context.get_cluster().configure_ethernet_cores_for_fabric_routers(
        fabric_config, std::numeric_limits<uint8_t>::max());
    context.set_default_fabric_topology();
    context.set_fabric_config(fabric_config, FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    context.initialize_fabric_config();
    const auto& control_plane = context.get_control_plane();
    const auto layouts = build_submesh_layouts_from_mgd(control_plane.get_mesh_graph());
    ASSERT_GE(layouts.size(), 2u);
    // Model-equivalent placement graphs only: no weights, programs, or MeshDevice.
    // A smaller graph searches all supplied submeshes, leaving the rest unused.
    const char* requested_graph = std::getenv("TT_PIPELINE_TEST_GRAPH");
    const std::string graph = requested_graph == nullptr ? "native_ring" : requested_graph;
    ASSERT_TRUE(graph == "native_ring" || graph == "ring" || graph == "small_ring" ||
                graph == "llama" || graph == "gemma" || graph == "fork" || graph == "fork_mpi");
    const char* requested_stages = std::getenv("TT_PIPELINE_TEST_STAGES");
    const size_t stage_count = requested_stages == nullptr ? layouts.size() : std::stoul(requested_stages);
    ASSERT_GE(stage_count, 2u);
    ASSERT_LE(stage_count, layouts.size());
    const auto nodes = build_ring_nodes(stage_count);
    auto edges = build_ring_edges(stage_count);
    std::map<std::string, uint32_t> stage_chip_counts;
    if (graph != "native_ring") {
        for (const auto& node : nodes) { stage_chip_counts[node] = graph == "small_ring" ? 2 : 8; }
    }
    if (graph == "llama") {
        ASSERT_EQ(stage_count, 40u);
        for (size_t i = 7; i < 39; ++i) { stage_chip_counts[nodes[i]] = 2; }
    } else if (graph == "gemma") {
        ASSERT_TRUE(stage_count == 5u || stage_count == 72u);
        const size_t narrow_stages = stage_count == 5u ? 4u : 64u;
        for (size_t i = 0; i < stage_count; ++i) { stage_chip_counts[nodes[i]] = i < narrow_stages ? 4 : 16; }
    } else if (graph == "fork" || graph == "fork_mpi") {
        // GPT-OSS/DFlash: router s0, 38 target stages, 11 draft stages.
        // The host-MPI variant excludes the draft return from the fabric graph.
        ASSERT_EQ(stage_count, 50u);
        edges.clear();
        edges.emplace_back("s0", "s1", false);
        for (size_t i = 1; i < 38; ++i) { edges.emplace_back(nodes[i], nodes[i + 1], false); }
        edges.emplace_back("s38", "s0", true);
        edges.emplace_back("s0", "s39", false);
        for (size_t i = 39; i < 49; ++i) { edges.emplace_back(nodes[i], nodes[i + 1], false); }
        if (graph == "fork") { edges.emplace_back("s49", "s0", true); }
    }
    const auto chips = to_submesh_chips(layouts);
    const auto started = std::chrono::steady_clock::now();
    std::optional<GraphLayoutResult> result;
    try {
        result = resolve_graph_layout(nodes, edges, chips, stage_chip_counts, {}, capacity);
    } catch (const std::runtime_error& error) {
        ASSERT_NE(std::string(error.what()).find("exact placement/link search exhausted"), std::string::npos)
            << error.what();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    fmt::print("PIPELINE_SWEEP graph={} stages={} submeshes={} capacity={} fabric={} outcome={} resolve_seconds={:.6f}\n",
               graph, stage_count, layouts.size(), capacity_text, fabric_mode, result ? "feasible" : "infeasible", seconds);
    if (result) {
        EXPECT_EQ(result->resolved_edges.size(), edges.size());
        const auto error = validate_pipeline_builder_graph_layout_errors(control_plane, layouts, *result, edges.size());
        EXPECT_FALSE(error.has_value()) << error.value_or("");
        std::set<size_t> assigned_submeshes;
        for (const auto& [stage, mesh] : result->node_to_submesh) {
            EXPECT_TRUE(assigned_submeshes.insert(mesh).second);
            if (stage_chip_counts.contains(stage)) { EXPECT_EQ(chips.at(mesh).size(), stage_chip_counts.at(stage)); }
        }
        std::set<std::tuple<std::string, uint32_t, uint32_t, uint32_t>> slots;
        auto check_slot = [&](const std::string& stage, uint32_t row, uint32_t col, std::optional<uint32_t> slot) {
            ASSERT_TRUE(slot.has_value());
            EXPECT_LT(*slot, capacity.value_or(chips.at(result->node_to_submesh.at(stage)).size() >= 8 ? 1u : 2u));
            EXPECT_TRUE(slots.emplace(stage, row, col, *slot).second);
        };
        for (const auto& edge : result->resolved_edges) {
            check_slot(edge.src, edge.exit_row, edge.exit_col, edge.exit_core_slot);
            check_slot(edge.dst, edge.entry_row, edge.entry_col, edge.entry_core_slot);
        }
        check_slot(result->stage_order.front(), result->h2d_entry_row, result->h2d_entry_col, result->h2d_core_slot);
        check_slot(result->stage_order.front(), result->d2h_exit_row, result->d2h_exit_col, result->d2h_core_slot);
    }
    context.get_cluster().configure_ethernet_cores_for_fabric_routers(FabricConfig::DISABLED);
}

TEST_F(ControlPlaneFixture, RejectsOneCoreLlamaLayout) {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    const auto& mesh_graph = tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_graph();
    const auto layouts = build_submesh_layouts_from_mgd(mesh_graph);
    const std::size_t num_4x2_stages = std::count_if(layouts.begin(), layouts.end(), [](const SubmeshLayout& layout) {
        return layout.rank_shape == MeshShape(4, 2);
    });
    const std::size_t num_1x2_stages = std::count_if(layouts.begin(), layouts.end(), [](const SubmeshLayout& layout) {
        return layout.rank_shape == MeshShape(1, 2);
    });
    if (layouts.size() != 40 || num_4x2_stages != 8 || num_1x2_stages != 32) {
        GTEST_SKIP() << "test requires the Llama 8B 1x2 pod MGD";
    }

    const auto edges = build_ring_edges(layouts.size());
    const auto nodes = build_ring_nodes(layouts.size());
    const auto submesh_chips = to_submesh_chips(layouts);
    std::map<std::string, uint32_t> capacities;
    for (const auto& node : nodes) {
        capacities.emplace(node, 1);
    }

    const auto default_layout = resolve_graph_layout(nodes, edges, submesh_chips);
    ASSERT_EQ(default_layout.resolved_edges.size(), edges.size());
    std::set<std::tuple<std::string, uint32_t, uint32_t, uint32_t>> default_slots;
    auto check_default_slot = [&](const std::string& stage, uint32_t row, uint32_t col, std::optional<uint32_t> slot) {
        ASSERT_TRUE(slot.has_value());
        const auto chip_count = submesh_chips.at(default_layout.node_to_submesh.at(stage)).size();
        EXPECT_LT(*slot, chip_count >= 8 ? 1u : 2u);
        EXPECT_TRUE(default_slots.emplace(stage, row, col, *slot).second);
    };
    for (const auto& edge : default_layout.resolved_edges) {
        check_default_slot(edge.src, edge.exit_row, edge.exit_col, edge.exit_core_slot);
        check_default_slot(edge.dst, edge.entry_row, edge.entry_col, edge.entry_core_slot);
    }
    check_default_slot("s0", default_layout.h2d_entry_row, default_layout.h2d_entry_col, default_layout.h2d_core_slot);
    check_default_slot("s0", default_layout.d2h_exit_row, default_layout.d2h_exit_col, default_layout.d2h_core_slot);

    // The same discovered topology must also admit a positive-capacity layout.
    const auto two_core_layout = resolve_graph_layout(nodes, edges, submesh_chips, {}, {}, 2);
    ASSERT_EQ(two_core_layout.resolved_edges.size(), edges.size());
    std::set<std::tuple<std::string, uint32_t, uint32_t, uint32_t>> slots;
    for (const auto& edge : two_core_layout.resolved_edges) {
        ASSERT_TRUE(edge.exit_core_slot);
        ASSERT_TRUE(edge.entry_core_slot);
        EXPECT_LT(*edge.exit_core_slot, 2);
        EXPECT_LT(*edge.entry_core_slot, 2);
        EXPECT_TRUE(slots.emplace(edge.src, edge.exit_row, edge.exit_col, *edge.exit_core_slot).second);
        EXPECT_TRUE(slots.emplace(edge.dst, edge.entry_row, edge.entry_col, *edge.entry_core_slot).second);
    }
    ASSERT_TRUE(two_core_layout.h2d_core_slot);
    ASSERT_TRUE(two_core_layout.d2h_core_slot);
    EXPECT_TRUE(slots.emplace("s0", two_core_layout.h2d_entry_row, two_core_layout.h2d_entry_col,
                              *two_core_layout.h2d_core_slot).second);
    EXPECT_TRUE(slots.emplace("s0", two_core_layout.d2h_exit_row, two_core_layout.d2h_exit_col,
                              *two_core_layout.d2h_core_slot).second);

    try {
        static_cast<void>(resolve_graph_layout(nodes, edges, submesh_chips, {}, capacities));
        FAIL() << "expected the one-core Llama layout to be rejected";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_EQ(message.find("search budget exhausted"), std::string::npos) << message;
        EXPECT_NE(message.find("no valid submesh assignment"), std::string::npos) << message;
    }
}

// Resolve and validate the canonical blaze pipeline ring for whatever MGD is loaded:
// one stage per host-rank submesh at its native shape, full loopback ring, single
// resolve_graph_layout call (the exact API tt-blaze build_topology* drives).
TEST_F(ControlPlaneFixture, TestPipelineBuilderCheck) {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    const auto layouts = build_submesh_layouts_from_mgd(mesh_graph);

    // A pipeline ring needs at least two host-rank submeshes (stages).  MGDs with a single
    // host slice (e.g. a 1x1 host_topology owning the whole mesh) describe no pipeline, so
    // skip rather than fail — these MGDs exist for the galaxy layout / corner-pin checks.
    if (layouts.size() < 2) {
        GTEST_SKIP() << "MGD has fewer than 2 host-rank submeshes; no pipeline ring to resolve";
    }

    const auto edges = build_ring_edges(layouts.size());
    const auto nodes = build_ring_nodes(layouts.size());
    const auto submesh_chips = to_submesh_chips(layouts);
    const bool all_single_chip = std::all_of(
        layouts.begin(), layouts.end(), [](const SubmeshLayout& layout) { return layout.rank_shape.mesh_size() == 1; });

    // The default API also allocates core slots. An all-1x1 ring needs four
    // slots on stage zero, so exercise that case with an explicit override below.
    if (!all_single_chip) {
        const GraphLayoutResult default_layout = resolve_graph_layout(nodes, edges, submesh_chips);
        EXPECT_TRUE(default_layout.h2d_core_slot.has_value());
        EXPECT_TRUE(default_layout.d2h_core_slot.has_value());
        for (const auto& edge : default_layout.resolved_edges) {
            EXPECT_TRUE(edge.exit_core_slot.has_value());
            EXPECT_TRUE(edge.entry_core_slot.has_value());
        }
        const auto default_error = validate_pipeline_builder_graph_layout_errors(control_plane, layouts, default_layout);
        EXPECT_FALSE(default_error.has_value())
            << "Default pipeline ring failed validation: " << default_error.value_or("")
            << "\n  layout: " << describe_layouts(layouts);
    }

    std::map<std::string, uint32_t> capacities;
    for (std::size_t i = 0; i < layouts.size(); ++i) {
        // Four is sufficient even for stage 0 on a one-chip submesh: forward exit,
        // loopback entry, H2D, and D2H. Other stages need only entry + exit.
        capacities.emplace(fmt::format("s{}", i), 4);
    }

    const GraphLayoutResult result = resolve_graph_layout(nodes, edges, submesh_chips, {}, capacities);
    const GraphLayoutResult repeated = resolve_graph_layout(nodes, edges, submesh_chips, {}, capacities);

    ASSERT_TRUE(result.h2d_core_slot.has_value());
    ASSERT_TRUE(result.d2h_core_slot.has_value());
    EXPECT_EQ(result.node_to_submesh, repeated.node_to_submesh);
    ASSERT_EQ(result.resolved_edges.size(), repeated.resolved_edges.size());

    using SlotKey = std::tuple<std::string, uint32_t, uint32_t>;
    std::map<SlotKey, std::set<uint32_t>> occupied_slots;
    auto check_slot =
        [&](const std::string& stage_name, uint32_t row, uint32_t col, const std::optional<uint32_t>& slot) {
            ASSERT_TRUE(slot.has_value());
            EXPECT_LT(*slot, capacities.at(stage_name));
            const SlotKey key{stage_name, row, col};
            EXPECT_TRUE(occupied_slots[key].insert(*slot).second)
                << "duplicate slot for " << stage_name << " chip (" << row << "," << col << ")";
        };
    for (std::size_t i = 0; i < result.resolved_edges.size(); ++i) {
        const auto& edge = result.resolved_edges[i];
        const auto& repeated_edge = repeated.resolved_edges[i];
        EXPECT_EQ(
            std::tie(
                edge.exit_row,
                edge.exit_col,
                edge.entry_row,
                edge.entry_col,
                edge.exit_core_slot,
                edge.entry_core_slot),
            std::tie(
                repeated_edge.exit_row,
                repeated_edge.exit_col,
                repeated_edge.entry_row,
                repeated_edge.entry_col,
                repeated_edge.exit_core_slot,
                repeated_edge.entry_core_slot));
        check_slot(edge.src, edge.exit_row, edge.exit_col, edge.exit_core_slot);
        check_slot(edge.dst, edge.entry_row, edge.entry_col, edge.entry_core_slot);
    }
    const std::string& stage0 = result.stage_order.front();
    check_slot(stage0, result.h2d_entry_row, result.h2d_entry_col, result.h2d_core_slot);
    check_slot(stage0, result.d2h_exit_row, result.d2h_exit_col, result.d2h_core_slot);

    for (const auto& edge : result.resolved_edges) {
        const std::size_t src_sub = result.node_to_submesh.at(edge.src);
        const std::size_t dst_sub = result.node_to_submesh.at(edge.dst);
        const FabricNodeId exit_fn = fabric_node_at_local_coord(layouts, src_sub, edge.exit_row, edge.exit_col);
        const FabricNodeId entry_fn = fabric_node_at_local_coord(layouts, dst_sub, edge.entry_row, edge.entry_col);
        log_debug(
            tt::LogTest,
            "EDGE {}->{}{} sub{}->sub{} exit=({},{}){} entry=({},{}){}",
            edge.src,
            edge.dst,
            edge.is_loopback ? "[lb]" : "",
            src_sub,
            dst_sub,
            edge.exit_row,
            edge.exit_col,
            exit_fn,
            edge.entry_row,
            edge.entry_col,
            entry_fn);
    }

    const auto err = validate_pipeline_builder_graph_layout_errors(control_plane, layouts, result);
    EXPECT_FALSE(err.has_value()) << "Pipeline ring (" << layouts.size()
                                  << " stages) failed validation: " << err.value_or("")
                                  << "\n  layout: " << describe_layouts(layouts);
}

}  // namespace tt::tt_fabric::fabric_router_tests
