// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/experimental/blitz_decode_pipeline.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental::blitz {

using ::tt::tt_metal::distributed::MeshCoordinate;

namespace {

std::vector<BlitzDecodePipelineStage> build_pipeline_from_topology(bool initialize_loopback) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    auto mesh_ids = mesh_graph.get_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end());
    const auto num_meshes = mesh_ids.size();

    auto fn_to_coord = [&](const tt::tt_fabric::FabricNodeId& fn) {
        return mesh_graph.chip_to_coordinate(fn.mesh_id, fn.chip_id);
    };

    // Track which FabricNodeIds have been claimed as entry or exit nodes.
    // Each hop claims its exit (on mesh_i) and peer (on mesh_{i+1}).
    // When selecting a pair for each hop, skip any pair that would reuse an already-claimed node.
    std::set<tt::tt_fabric::FabricNodeId> used_nodes;

    // Select one inter-mesh pair per hop, avoiding collisions.
    // With loopback:    N hops: mesh_0→mesh_1→...→mesh_{N-1}→mesh_0
    // Without loopback: N-1 hops: mesh_0→mesh_1→...→mesh_{N-1} (no return)
    const std::size_t num_hops = initialize_loopback ? num_meshes : num_meshes - 1;
    std::vector<std::pair<tt::tt_fabric::FabricNodeId, tt::tt_fabric::FabricNodeId>> hops;
    hops.reserve(num_hops);
    for (std::size_t i = 0; i < num_hops; i++) {
        const auto next = initialize_loopback ? (i + 1) % num_meshes : i + 1;
        auto pairs =
            control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(mesh_ids[i], mesh_ids[next]);
        TT_FATAL(!pairs.empty(), "No inter-mesh connection from mesh {} to mesh {}", *mesh_ids[i], *mesh_ids[next]);

        bool found = false;
        for (const auto& pair : pairs) {
            if (used_nodes.contains(pair.first) || used_nodes.contains(pair.second)) {
                continue;
            }
            hops.push_back(pair);
            used_nodes.insert(pair.first);
            used_nodes.insert(pair.second);
            found = true;
            break;
        }
        TT_FATAL(
            found,
            "No non-colliding inter-mesh pair from mesh {} to mesh {} "
            "(all {} candidate pairs overlap with already-claimed nodes)",
            *mesh_ids[i],
            *mesh_ids[next],
            pairs.size());
    }

    // Pipeline data flow (with loopback):
    //   stage 0 → stage 1 → ... → stage N-1 → loopback(stage N) [→ stage 0 intra-mesh]
    //   hop[0]:   mesh_0   → mesh_1     (stage 0 exit   → stage 1 entry)
    //   ...
    //   hop[N-1]: mesh_{N-1} → mesh_0     (stage N-1 exit → loopback entry)
    //   Stage 0 entry and loopback exit are intra-mesh on mesh_0.
    //
    // Without loopback:
    //   stage 0 → stage 1 → ... → stage N-1 (no return path)
    //   hop[0]..hop[N-2]: mesh_0→mesh_1→...→mesh_{N-1}
    //   Stage 0 entry is intra-mesh on mesh_0.

    // Find unclaimed nodes on mesh_0: need 2 with loopback (entry + loopback exit), 1 without.
    const std::size_t unclaimed_needed = initialize_loopback ? 2 : 1;
    auto mesh_0_coord_range = mesh_graph.get_coord_range(mesh_ids[0]);
    std::vector<tt::tt_fabric::FabricNodeId> unclaimed_mesh_0_nodes;
    for (const auto& coord : mesh_0_coord_range) {
        auto chip_id = mesh_graph.coordinate_to_chip(mesh_ids[0], coord);
        tt::tt_fabric::FabricNodeId fn(mesh_ids[0], chip_id);
        if (!used_nodes.contains(fn)) {
            unclaimed_mesh_0_nodes.push_back(fn);
            if (unclaimed_mesh_0_nodes.size() >= unclaimed_needed) {
                break;
            }
        }
    }
    TT_FATAL(
        unclaimed_mesh_0_nodes.size() >= unclaimed_needed,
        "Need {} unclaimed nodes on mesh {} for stage 0 entry{}, found {}",
        unclaimed_needed,
        *mesh_ids[0],
        initialize_loopback ? " and loopback exit" : "",
        unclaimed_mesh_0_nodes.size());

    auto stage_0_entry_fn = unclaimed_mesh_0_nodes[0];

    std::vector<BlitzDecodePipelineStage> stages;
    stages.reserve(initialize_loopback ? num_meshes + 1 : num_meshes);

    // Stage 0: entry is intra-mesh, exit goes to mesh_1 via hop[0]
    stages.emplace_back(BlitzDecodePipelineStage{
        .stage_index = static_cast<std::size_t>(*mesh_ids[0]),
        .entry_node_coord = fn_to_coord(stage_0_entry_fn),
        .exit_node_coord = fn_to_coord(hops[0].first)});

    // Stages 1..N-1: entry from previous hop's peer, exit from current hop.
    // For the last stage without loopback there is no downstream exit; use entry as a placeholder
    // (exit_node_coord is not used by the no-loopback last-stage path in PipelineBlock).
    for (std::size_t i = 1; i < num_meshes; i++) {
        const auto entry = fn_to_coord(hops[i - 1].second);
        const bool is_last_no_loopback = !initialize_loopback && (i == num_meshes - 1);
        stages.emplace_back(BlitzDecodePipelineStage{
            .stage_index = static_cast<std::size_t>(*mesh_ids[i]),
            .entry_node_coord = entry,
            .exit_node_coord = is_last_no_loopback ? entry : fn_to_coord(hops[i].first)});
    }

    if (initialize_loopback) {
        // Loopback stage (on mesh_0): entry from hop[N-1], exit is intra-mesh back to stage 0
        auto loopback_exit_fn = unclaimed_mesh_0_nodes[1];
        stages.emplace_back(BlitzDecodePipelineStage{
            .stage_index = static_cast<std::size_t>(*mesh_ids[0]),
            .entry_node_coord = fn_to_coord(hops[num_meshes - 1].second),
            .exit_node_coord = fn_to_coord(loopback_exit_fn)});
    }

    return stages;
}

void validate_pipeline(const std::vector<BlitzDecodePipelineStage>& stages, bool initialize_loopback) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    auto mesh_ids = mesh_graph.get_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end());
    const auto num_meshes = mesh_ids.size();

    auto coord_str = [](const MeshCoordinate& c) { return fmt::format("({}, {})", c[0], c[1]); };

    const std::size_t expected = initialize_loopback ? num_meshes + 1 : num_meshes;
    TT_FATAL(
        stages.size() == expected,
        "Expected {} stages (num_meshes={}{}) got {}",
        expected,
        num_meshes,
        initialize_loopback ? " + 1 loopback" : "",
        stages.size());

    // 1. No stage has identical entry and exit coords (skip last stage when no loopback — it has no exit)
    const std::size_t check_distinct_until = initialize_loopback ? stages.size() : stages.size() - 1;
    for (std::size_t i = 0; i < check_distinct_until; i++) {
        const auto& s = stages[i];
        TT_FATAL(
            s.entry_node_coord != s.exit_node_coord,
            "Stage [{}] (stage_index={}) has identical entry and exit coords {}",
            i,
            s.stage_index,
            coord_str(s.entry_node_coord));
    }

    // 2. No coord is reused across stages (no overlapping nodes).
    // Skip the last stage's exit when !initialize_loopback — it has no downstream exit.
    std::set<std::pair<std::size_t, std::pair<uint32_t, uint32_t>>> used_coords;
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        const bool skip_exit = !initialize_loopback && (i == stages.size() - 1);
        auto entry_key = std::make_pair(s.stage_index, std::make_pair(s.entry_node_coord[0], s.entry_node_coord[1]));
        TT_FATAL(
            used_coords.insert(entry_key).second,
            "Stage [{}] entry coord {} (stage_index={}) overlaps with a previous stage",
            i,
            coord_str(s.entry_node_coord),
            s.stage_index);
        if (!skip_exit) {
            auto exit_key = std::make_pair(s.stage_index, std::make_pair(s.exit_node_coord[0], s.exit_node_coord[1]));
            TT_FATAL(
                used_coords.insert(exit_key).second,
                "Stage [{}] exit coord {} (stage_index={}) overlaps with a previous stage",
                i,
                coord_str(s.exit_node_coord),
                s.stage_index);
        }
    }

    // 3. Consecutive stages are physically connected via inter-mesh links:
    //    stage[i].exit on mesh_i must be an exit node to the mesh of stage[i+1],
    //    and stage[i+1].entry must be the peer on the other side of that cable.
    for (std::size_t i = 0; i < stages.size() - 1; i++) {
        const auto& curr = stages[i];
        const auto& next = stages[i + 1];

        auto curr_mesh_id = tt::tt_fabric::MeshId{static_cast<uint32_t>(curr.stage_index)};
        auto next_mesh_id = tt::tt_fabric::MeshId{static_cast<uint32_t>(next.stage_index)};

        if (curr_mesh_id == next_mesh_id) {
            continue;
        }

        auto exit_chip_id = mesh_graph.coordinate_to_chip(curr_mesh_id, curr.exit_node_coord);
        auto entry_chip_id = mesh_graph.coordinate_to_chip(next_mesh_id, next.entry_node_coord);
        tt::tt_fabric::FabricNodeId exit_fn(curr_mesh_id, exit_chip_id);
        tt::tt_fabric::FabricNodeId entry_fn(next_mesh_id, entry_chip_id);

        auto pairs =
            control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(curr_mesh_id, next_mesh_id);

        bool found = false;
        for (const auto& [exit_node, peer_node] : pairs) {
            if (exit_node == exit_fn && peer_node == entry_fn) {
                found = true;
                break;
            }
        }
        TT_FATAL(
            found,
            "Stages [{}]->[{}]: exit (M{}D{}) coord {} is not physically connected to entry (M{}D{}) coord {}",
            i,
            i + 1,
            *curr_mesh_id,
            exit_chip_id,
            coord_str(curr.exit_node_coord),
            *next_mesh_id,
            entry_chip_id,
            coord_str(next.entry_node_coord));
    }

    // 4. Loopback stage (last) must have different entry/exit than stage 0
    if (initialize_loopback) {
        const auto& stage_0 = stages[0];
        const auto& loopback = stages.back();
        TT_FATAL(
            loopback.entry_node_coord != stage_0.entry_node_coord ||
                loopback.exit_node_coord != stage_0.exit_node_coord,
            "Loopback stage has identical entry/exit as stage 0: entry={}, exit={}",
            coord_str(loopback.entry_node_coord),
            coord_str(loopback.exit_node_coord));
    }
}

}  // namespace

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(bool initialize_loopback) {
    log_info(
        LogMetal,
        "[generate_blitz_decode_pipeline] rank={} initialize_loopback={} — building topology",
        *tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->rank(),
        initialize_loopback);
    auto stages = build_pipeline_from_topology(initialize_loopback);
    log_info(
        LogMetal,
        "[generate_blitz_decode_pipeline] rank={} — built {} stages, validating",
        *tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->rank(),
        stages.size());
    validate_pipeline(stages, initialize_loopback);
    log_info(
        LogMetal,
        "[generate_blitz_decode_pipeline] rank={} — validated, entering barrier",
        *tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->rank());

    // Synchronize all ranks before returning so that downstream socket creation
    // (which cascades sequentially through stages) starts from a common point.
    // The old implementation had implicit synchronization via MPI broadcasts in
    // get_asic_id_to_mesh_coord_map / create_physical_system_descriptor; without
    // this barrier the initialization-time variance across ranks can push the
    // sequential handshake cascade past the 10-second MeshSocket timeout.
    const auto& ctx = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    ctx->barrier();

    log_info(LogMetal, "[generate_blitz_decode_pipeline] rank={} — barrier complete, returning", *ctx->rank());
    return stages;
}

}  // namespace tt::tt_metal::experimental::blitz
