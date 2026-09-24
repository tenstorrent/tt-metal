// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CPU-only pipeline placement tests using mock cluster descriptors.
// SC20/SC24 rings and spec-decode forks run through the pipeline-placement driver group.
// No MeshDevice or MeshSockets are created.

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fabric_fixture.hpp"
#include "utils.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_context.hpp>

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

void check_graph_capacity(const ControlPlane& control_plane, uint32_t capacity, const std::string& fabric_mode) {
    const auto layouts = build_submesh_layouts_from_mgd(control_plane.get_mesh_graph());
    ASSERT_GE(layouts.size(), 2u);
    // Model-equivalent placement graphs only: no weights, programs, or MeshDevice.
    // A smaller graph searches all supplied submeshes, leaving the rest unused.
    const char* requested_graph = std::getenv("TT_PIPELINE_TEST_GRAPH");
    const std::string graph = requested_graph == nullptr ? "native_ring" : requested_graph;
    ASSERT_TRUE(graph == "native_ring" || graph == "ring" || graph == "fork" || graph == "fork_mpi");
    const char* requested_stages = std::getenv("TT_PIPELINE_TEST_STAGES");
    const size_t stage_count = requested_stages == nullptr ? layouts.size() : std::stoul(requested_stages);
    ASSERT_GE(stage_count, 2u);
    ASSERT_LE(stage_count, layouts.size());
    const auto nodes = build_ring_nodes(stage_count);
    auto edges = build_ring_edges(stage_count);
    std::map<std::string, uint32_t> stage_chip_counts;
    if (graph != "native_ring") {
        for (const auto& node : nodes) {
            stage_chip_counts[node] = 8;
        }
    }
    if (graph == "fork" || graph == "fork_mpi") {
        // Router s0, followed by the target branch and then the draft branch.
        // The host-MPI variant excludes the draft return from the fabric graph.
        const char* requested_target_stages = std::getenv("TT_PIPELINE_TEST_TARGET_STAGES");
        const size_t target_stages = requested_target_stages == nullptr ? 38u : std::stoul(requested_target_stages);
        ASSERT_GT(target_stages, 0u);
        ASSERT_GT(stage_count, target_stages + 1);
        edges.clear();
        edges.emplace_back("s0", "s1", false);
        for (size_t i = 1; i < target_stages; ++i) {
            edges.emplace_back(nodes[i], nodes[i + 1], false);
        }
        edges.emplace_back(nodes[target_stages], "s0", true);
        edges.emplace_back("s0", nodes[target_stages + 1], false);
        for (size_t i = target_stages + 1; i + 1 < stage_count; ++i) {
            edges.emplace_back(nodes[i], nodes[i + 1], false);
        }
        if (graph == "fork") {
            edges.emplace_back(nodes.back(), "s0", true);
        }
    }
    const auto chips = to_submesh_chips(layouts);
    const auto started = std::chrono::steady_clock::now();
    const auto result = resolve_graph_layout(nodes, edges, chips, stage_chip_counts, {}, capacity);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    fmt::print(
        "PIPELINE_SWEEP graph={} stages={} submeshes={} capacity={} fabric={} outcome=feasible "
        "resolve_seconds={:.6f}\n",
        graph,
        stage_count,
        layouts.size(),
        capacity,
        fabric_mode,
        seconds);
    EXPECT_EQ(result.resolved_edges.size(), edges.size());
    const auto error = validate_pipeline_builder_graph_layout_errors(control_plane, layouts, result, edges.size());
    EXPECT_FALSE(error.has_value()) << error.value_or("");
    std::set<size_t> assigned_submeshes;
    for (const auto& [stage, mesh] : result.node_to_submesh) {
        EXPECT_TRUE(assigned_submeshes.insert(mesh).second);
        if (stage_chip_counts.contains(stage)) {
            EXPECT_EQ(chips.at(mesh).size(), stage_chip_counts.at(stage));
        }
    }
    std::set<std::tuple<std::string, uint32_t, uint32_t, uint32_t>> slots;
    auto check_slot = [&](const std::string& stage, uint32_t row, uint32_t col, std::optional<uint32_t> slot) {
        ASSERT_TRUE(slot.has_value());
        EXPECT_LT(*slot, capacity);
        EXPECT_TRUE(slots.emplace(stage, row, col, *slot).second);
    };
    for (const auto& edge : result.resolved_edges) {
        check_slot(edge.src, edge.exit_row, edge.exit_col, edge.exit_core_slot);
        check_slot(edge.dst, edge.entry_row, edge.entry_col, edge.entry_core_slot);
    }
    check_slot(result.stage_order.front(), result.h2d_entry_row, result.h2d_entry_col, result.h2d_core_slot);
    check_slot(result.stage_order.front(), result.d2h_exit_row, result.d2h_exit_col, result.d2h_core_slot);
}

}  // namespace

// tt-run supplies the mock descriptor and MGD. Each case must find a valid layout.
TEST(PipelineBuilderMockTest, GraphCapacity) {
    const char* mock = std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    const char* requested_capacity = std::getenv("TT_PIPELINE_TEST_CORE_CAPACITY");
    if (mock == nullptr || *mock == '\0' || requested_capacity == nullptr) {
        GTEST_SKIP() << "requires mock cluster and TT_PIPELINE_TEST_CORE_CAPACITY";
    }
    const std::string capacity_text(requested_capacity);
    ASSERT_TRUE(capacity_text == "1" || capacity_text == "2");
    const uint32_t capacity = std::stoul(capacity_text);
    const char* requested_fabric = std::getenv("TT_PIPELINE_TEST_FABRIC");
    const std::string fabric_mode = requested_fabric == nullptr ? "2D" : requested_fabric;
    ASSERT_TRUE(fabric_mode == "2D" || fabric_mode == "TORUS_XY" || fabric_mode == "TORUS_Y");
    auto fabric_config = FabricConfig::FABRIC_2D;
    if (fabric_mode == "TORUS_XY") {
        fabric_config = FabricConfig::FABRIC_2D_TORUS_XY;
    } else if (fabric_mode == "TORUS_Y") {
        fabric_config = FabricConfig::FABRIC_2D_TORUS_Y;
    }

    auto& context = tt::tt_metal::MetalContext::instance();
    context.get_cluster().configure_ethernet_cores_for_fabric_routers(
        fabric_config, std::numeric_limits<uint8_t>::max());
    context.set_default_fabric_topology();
    context.set_fabric_config(fabric_config, FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    context.initialize_fabric_config();
    // Initialization is collective; placement uses only local control-plane data.
    const auto& world = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*world->rank() == 0) {
        EXPECT_NO_THROW(check_graph_capacity(context.get_control_plane(), capacity, fabric_mode));
    }
    // The helper keeps fatal assertions from skipping synchronization and cleanup.
    world->barrier();
    context.get_cluster().configure_ethernet_cores_for_fabric_routers(FabricConfig::DISABLED);
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

    // A 1x1 host-rank slice has a single chip.  Same-chip entry/exit sockets are valid at
    // runtime (PipelineBlock splits them across two cores), but resolve_graph_layout still
    // requires a distinct exit chip per stage, so it cannot resolve an all-1x1 ring.  blaze
    // never builds 1x1-stage pipelines, so skip rather than exercise that resolver path.
    if (std::all_of(layouts.begin(), layouts.end(), [](const SubmeshLayout& layout) {
            return layout.rank_shape.mesh_size() == 1;
        })) {
        GTEST_SKIP() << "all-1x1 host-rank slices: resolve_graph_layout requires a distinct exit chip per stage";
    }

    const GraphLayoutResult result = resolve_graph_layout(
        build_ring_nodes(layouts.size()), build_ring_edges(layouts.size()), to_submesh_chips(layouts));

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
