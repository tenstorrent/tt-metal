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

std::vector<BlitzDecodePipelineStage> build_pipeline_from_topology() {
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
    // hop[i] connects mesh_ids[i] → mesh_ids[(i+1) % N].
    std::vector<std::pair<tt::tt_fabric::FabricNodeId, tt::tt_fabric::FabricNodeId>> hops;
    hops.reserve(num_meshes);
    for (std::size_t i = 0; i < num_meshes; i++) {
        auto pairs = control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(
            mesh_ids[i], mesh_ids[(i + 1) % num_meshes]);
        TT_FATAL(
            !pairs.empty(),
            "No inter-mesh connection from mesh {} to mesh {}",
            *mesh_ids[i],
            *mesh_ids[(i + 1) % num_meshes]);

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
            *mesh_ids[(i + 1) % num_meshes],
            pairs.size());
    }

    // Pipeline data flow:
    //   stage 0 → stage 1 → ... → stage N-1 → loopback(stage N) [→ stage 0 intra-mesh]
    //
    // Inter-mesh hops:
    //   hop[0]:   mesh_0   → mesh_1     (stage 0 exit   → stage 1 entry)
    //   hop[1]:   mesh_1   → mesh_2     (stage 1 exit   → stage 2 entry)
    //   ...
    //   hop[N-2]: mesh_{N-2} → mesh_{N-1} (stage N-2 exit → stage N-1 entry)
    //   hop[N-1]: mesh_{N-1} → mesh_0     (stage N-1 exit → loopback entry)
    //
    // Stage 0 entry and loopback exit are intra-mesh on mesh_0 (not from inter-mesh cables).
    // Find two unclaimed nodes on mesh_0 for these.
    auto mesh_0_coord_range = mesh_graph.get_coord_range(mesh_ids[0]);
    std::vector<tt::tt_fabric::FabricNodeId> unclaimed_mesh_0_nodes;
    for (const auto& coord : mesh_0_coord_range) {
        auto chip_id = mesh_graph.coordinate_to_chip(mesh_ids[0], coord);
        tt::tt_fabric::FabricNodeId fn(mesh_ids[0], chip_id);
        if (!used_nodes.contains(fn)) {
            unclaimed_mesh_0_nodes.push_back(fn);
            if (unclaimed_mesh_0_nodes.size() >= 2) {
                break;
            }
        }
    }
    TT_FATAL(
        unclaimed_mesh_0_nodes.size() >= 2,
        "Need 2 unclaimed nodes on mesh {} for stage 0 entry and loopback exit, found {}",
        *mesh_ids[0],
        unclaimed_mesh_0_nodes.size());

    auto stage_0_entry_fn = unclaimed_mesh_0_nodes[0];
    auto loopback_exit_fn = unclaimed_mesh_0_nodes[1];

    std::vector<BlitzDecodePipelineStage> stages;
    stages.reserve(num_meshes + 1);

    // Stage 0: entry is intra-mesh (from loopback), exit goes to mesh_1 via hop[0]
    stages.emplace_back(BlitzDecodePipelineStage{
        .stage_index = static_cast<std::size_t>(*mesh_ids[0]),
        .entry_node_coord = fn_to_coord(stage_0_entry_fn),
        .exit_node_coord = fn_to_coord(hops[0].first)});

    // Stages 1..N-1: entry from previous hop's peer, exit from current hop
    for (std::size_t i = 1; i < num_meshes; i++) {
        stages.emplace_back(BlitzDecodePipelineStage{
            .stage_index = static_cast<std::size_t>(*mesh_ids[i]),
            .entry_node_coord = fn_to_coord(hops[i - 1].second),
            .exit_node_coord = fn_to_coord(hops[i].first)});
    }

    // Loopback stage (on mesh_0): entry from hop[N-1] (connected to stage N-1 exit),
    // exit is intra-mesh (feeds back into stage 0)
    stages.emplace_back(BlitzDecodePipelineStage{
        .stage_index = static_cast<std::size_t>(*mesh_ids[0]),
        .entry_node_coord = fn_to_coord(hops[num_meshes - 1].second),
        .exit_node_coord = fn_to_coord(loopback_exit_fn)});

    return stages;
}

void validate_pipeline(const std::vector<BlitzDecodePipelineStage>& stages) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    auto mesh_ids = mesh_graph.get_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end());
    const auto num_meshes = mesh_ids.size();

    auto coord_str = [](const MeshCoordinate& c) { return fmt::format("({}, {})", c[0], c[1]); };

    TT_FATAL(
        stages.size() == num_meshes + 1,
        "Expected {} stages (num_meshes={} + 1 loopback), got {}",
        num_meshes + 1,
        num_meshes,
        stages.size());

    // 1. No stage has identical entry and exit coords
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        TT_FATAL(
            s.entry_node_coord != s.exit_node_coord,
            "Stage [{}] (stage_index={}) has identical entry and exit coords {}",
            i,
            s.stage_index,
            coord_str(s.entry_node_coord));
    }

    // 2. No coord is reused across stages (no overlapping nodes)
    std::set<std::pair<std::size_t, std::pair<uint32_t, uint32_t>>> used_coords;
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        auto entry_key = std::make_pair(s.stage_index, std::make_pair(s.entry_node_coord[0], s.entry_node_coord[1]));
        auto exit_key = std::make_pair(s.stage_index, std::make_pair(s.exit_node_coord[0], s.exit_node_coord[1]));
        TT_FATAL(
            used_coords.insert(entry_key).second,
            "Stage [{}] entry coord {} (stage_index={}) overlaps with a previous stage",
            i,
            coord_str(s.entry_node_coord),
            s.stage_index);
        TT_FATAL(
            used_coords.insert(exit_key).second,
            "Stage [{}] exit coord {} (stage_index={}) overlaps with a previous stage",
            i,
            coord_str(s.exit_node_coord),
            s.stage_index);
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
    const auto& stage_0 = stages[0];
    const auto& loopback = stages.back();
    TT_FATAL(
        loopback.entry_node_coord != stage_0.entry_node_coord || loopback.exit_node_coord != stage_0.exit_node_coord,
        "Loopback stage has identical entry/exit as stage 0: entry={}, exit={}",
        coord_str(loopback.entry_node_coord),
        coord_str(loopback.exit_node_coord));
}

}  // namespace

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline() {
    auto stages = build_pipeline_from_topology();
    validate_pipeline(stages);

    // Synchronize all ranks before returning so that downstream socket creation
    // (which cascades sequentially through stages) starts from a common point.
    // The old implementation had implicit synchronization via MPI broadcasts in
    // get_asic_id_to_mesh_coord_map / create_physical_system_descriptor; without
    // this barrier the initialization-time variance across ranks can push the
    // sequential handshake cascade past the 10-second MeshSocket timeout.
    const auto& ctx = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    ctx->barrier();

    return stages;
}

}  // namespace tt::tt_metal::experimental::blitz
