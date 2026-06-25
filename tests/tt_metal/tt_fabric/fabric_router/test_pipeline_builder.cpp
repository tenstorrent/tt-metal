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
/// distinct entry/exit chips per stage, active fabric eth channels, a direct PSD
/// ethernet link + matching fabric hop per edge, and no socket-endpoint chip reuse.

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_set>
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
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::optional<std::string> validate_pipeline_builder_graph_layout_errors(
    const ControlPlane& control_plane, const std::vector<SubmeshLayout>& layouts, const GraphLayoutResult& result) {
    const std::size_t num_stages = result.stage_order.size();

    if (result.resolved_edges.size() != num_stages) {
        return fmt::format("resolved_edges size {} != stage count {}", result.resolved_edges.size(), num_stages);
    }
    if (result.node_to_submesh.size() != num_stages) {
        return fmt::format("node_to_submesh size {} != stage count {}", result.node_to_submesh.size(), num_stages);
    }

    auto psd_has_direct_eth_link = [&](const FabricNodeId& a, const FabricNodeId& b) {
        const auto& psd = control_plane.get_physical_system_descriptor();
        const auto asic_a = control_plane.get_asic_id_from_fabric_node_id(a);
        const auto asic_b = control_plane.get_asic_id_from_fabric_node_id(b);
        if (!psd.get_eth_connections(asic_a, asic_b).empty()) {
            return true;
        }
        return !psd.get_eth_connections(asic_b, asic_a).empty();
    };

    using EthDir = tt::tt_fabric::eth_chan_directions;
    auto is_z_eth_dir = [](EthDir d) { return d == EthDir::Z; };
    auto is_nesw_eth_dir = [](EthDir d) {
        return d == EthDir::NORTH || d == EthDir::SOUTH || d == EthDir::EAST || d == EthDir::WEST;
    };
    auto eth_dirs_match_kind = [&](EthDir a, EthDir b) {
        return (is_z_eth_dir(a) && is_z_eth_dir(b)) || (is_nesw_eth_dir(a) && is_nesw_eth_dir(b));
    };

    std::unordered_set<FabricNodeId> used_socket_nodes;
    used_socket_nodes.reserve(num_stages * 4);

    for (const auto& edge : result.resolved_edges) {
        const std::size_t src_sub = result.node_to_submesh.at(edge.src);
        const std::size_t dst_sub = result.node_to_submesh.at(edge.dst);
        const FabricNodeId exit_fn = fabric_node_at_local_coord(layouts, src_sub, edge.exit_row, edge.exit_col);
        const FabricNodeId entry_fn = fabric_node_at_local_coord(layouts, dst_sub, edge.entry_row, edge.entry_col);

        if (control_plane.get_active_fabric_eth_channels(exit_fn).empty()) {
            return fmt::format("Edge exit {} has no active fabric ethernet channels", exit_fn);
        }
        if (control_plane.get_active_fabric_eth_channels(entry_fn).empty()) {
            return fmt::format("Edge entry {} has no active fabric ethernet channels", entry_fn);
        }
        if (!used_socket_nodes.insert(exit_fn).second) {
            return fmt::format("Exit fabric node {} reused across pipeline socket endpoints", exit_fn);
        }
        if (!used_socket_nodes.insert(entry_fn).second) {
            return fmt::format("Entry fabric node {} reused across pipeline socket endpoints", entry_fn);
        }
        if (!psd_has_direct_eth_link(exit_fn, entry_fn)) {
            return fmt::format("No direct PSD ethernet edge for {} -> {}", exit_fn, entry_fn);
        }

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

    // Each forwarding stage must place its entry and exit on DIFFERENT chips of its own
    // submesh — two persistent kernels on the same core would deadlock.  This is the
    // property the resolver's deconfliction step (pipeline_builder.cpp) guarantees.
    for (std::size_t stage_idx = 1; stage_idx < num_stages; ++stage_idx) {
        const std::string& stage_name = result.stage_order[stage_idx];

        uint32_t entry_row = 0;
        uint32_t entry_col = 0;
        for (const auto& edge : result.resolved_edges) {
            if (!edge.is_loopback && edge.dst == stage_name) {
                entry_row = edge.entry_row;
                entry_col = edge.entry_col;
                break;
            }
        }

        uint32_t exit_row = 0;
        uint32_t exit_col = 0;
        for (const auto& edge : result.resolved_edges) {
            if (edge.src == stage_name) {
                exit_row = edge.exit_row;
                exit_col = edge.exit_col;
                break;
            }
        }

        if (std::make_pair(entry_row, entry_col) == std::make_pair(exit_row, exit_col)) {
            return fmt::format(
                "Stage {} ({}) has colliding entry/exit on submesh for rank shape {}",
                stage_idx,
                stage_name,
                layouts.at(result.node_to_submesh.at(stage_name)).rank_shape);
        }
    }

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

// Resolve and validate the canonical blaze pipeline ring for whatever MGD is loaded:
// one stage per host-rank submesh at its native shape, full loopback ring, single
// resolve_graph_layout call (the exact API tt-blaze build_topology* drives).
TEST_F(ControlPlaneFixture, TestPipelineBuilderGraphLayout) {
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

    // A 1x1 host-rank slice has a single chip, so its entry and exit sockets cannot land
    // on different chips.  blaze never builds 1x1-stage pipelines, so skip rather than fail.
    if (std::all_of(layouts.begin(), layouts.end(), [](const SubmeshLayout& layout) {
            return layout.rank_shape.mesh_size() == 1;
        })) {
        GTEST_SKIP() << "1x1 host-rank slices cannot place distinct pipeline entry/exit socket coords";
    }

    const GraphLayoutResult result = resolve_graph_layout(build_ring_edges(layouts.size()), to_submesh_chips(layouts));

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
