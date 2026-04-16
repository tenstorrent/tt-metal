// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/experimental/blitz_decode_pipeline.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental::blitz {

using ::tt::tt_fabric::FabricNodeId;
using ::tt::tt_fabric::MeshId;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::multihost::Rank;

namespace {

const char* eth_chan_dir_cstr(tt::tt_fabric::eth_chan_directions d) {
    switch (d) {
        case tt::tt_fabric::eth_chan_directions::EAST: return "EAST";
        case tt::tt_fabric::eth_chan_directions::WEST: return "WEST";
        case tt::tt_fabric::eth_chan_directions::NORTH: return "NORTH";
        case tt::tt_fabric::eth_chan_directions::SOUTH: return "SOUTH";
        case tt::tt_fabric::eth_chan_directions::Z: return "Z";
        default: return "UNKNOWN";
    }
}

std::string describe_fabric_node_line(
    const tt::tt_fabric::ControlPlane& control_plane,
    const tt::tt_fabric::MeshGraph& mesh_graph,
    std::size_t stage_index,
    const MeshCoordinate& coord,
    const char* role_label) {
    MeshId mesh_id{static_cast<uint32_t>(stage_index)};
    ChipId chip_id = mesh_graph.coordinate_to_chip(mesh_id, coord);
    FabricNodeId fn(mesh_id, chip_id);
    const auto& topology_mapper = control_plane.get_topology_mapper();
    auto host = topology_mapper.get_hostname_for_fabric_node_id(fn);
    auto tray_id = topology_mapper.get_tray_id_for_fabric_node_id(fn);
    auto asic_loc = topology_mapper.get_asic_location_for_fabric_node_id(fn);
    return fmt::format(
        "{}: host={} tray_id={} asic_location={} mesh_id={} logical_chip_id={} mesh_coord=({}, {}) fabric_node={}",
        role_label,
        host,
        *tray_id,
        *asic_loc,
        *mesh_id,
        chip_id,
        coord[0],
        coord[1],
        fn);
}

std::string describe_direct_eth_links(
    const tt::tt_fabric::ControlPlane& control_plane, FabricNodeId src_fn, FabricNodeId dst_fn) {
    std::vector<std::string> link_parts;
    for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(src_fn)) {
        auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(src_fn, src_chan);
        if (peer_fn != dst_fn) {
            continue;
        }
        tt::tt_fabric::eth_chan_directions dst_dir =
            control_plane.get_eth_chan_direction(peer_fn, static_cast<int>(peer_chan));
        link_parts.push_back(fmt::format(
            "src_chan={} src_port_dir={} dst_chan={} dst_port_dir={}",
            src_chan,
            eth_chan_dir_cstr(src_dir),
            peer_chan,
            eth_chan_dir_cstr(dst_dir)));
    }
    if (link_parts.empty()) {
        return "no single-hop ethernet link found (topology may be multi-hop or unmapped on this rank)";
    }
    std::string joined;
    for (std::size_t i = 0; i < link_parts.size(); i++) {
        if (i != 0) {
            joined += "; ";
        }
        joined += link_parts[i];
    }
    return joined;
}

void log_blitz_decode_pipeline_stages(const std::vector<BlitzDecodePipelineStage>& stages) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& ctx = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    if (*ctx->rank() != 0) {
        return;
    }

    log_info(tt::LogMetal, "Blitz decode pipeline: {} stages (including loopback)", stages.size());
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& stage = stages[i];
        MeshId mesh_id{static_cast<uint32_t>(stage.stage_index)};
        FabricNodeId entry_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, stage.entry_node_coord));
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, stage.exit_node_coord));

        const std::size_t next_i = (i + 1) % stages.size();
        const auto& next_stage = stages[next_i];
        MeshId next_mesh_id{static_cast<uint32_t>(next_stage.stage_index)};
        FabricNodeId next_entry_fn(
            next_mesh_id, mesh_graph.coordinate_to_chip(next_mesh_id, next_stage.entry_node_coord));

        log_info(tt::LogMetal, "BlitzDecode stage [{}] stage_index={}", i, stage.stage_index);
        log_info(
            tt::LogMetal,
            "  {}",
            describe_fabric_node_line(control_plane, mesh_graph, stage.stage_index, stage.entry_node_coord, "entry"));
        log_info(
            tt::LogMetal,
            "  {}",
            describe_fabric_node_line(control_plane, mesh_graph, stage.stage_index, stage.exit_node_coord, "exit"));
        log_info(
            tt::LogMetal,
            "  exit -> next_stage[{}] entry ({} -> {}): {}",
            next_i,
            exit_fn,
            next_entry_fn,
            describe_direct_eth_links(control_plane, exit_fn, next_entry_fn));
    }
}

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
    // Pick two unclaimed chips with a direct intra-mesh ethernet link (loopback_exit -> stage_0_entry).
    // Prefer a non-Z link when available so the loopback hop matches typical NESW mesh wiring.
    auto mesh_0_coord_range = mesh_graph.get_coord_range(mesh_ids[0]);
    std::vector<tt::tt_fabric::FabricNodeId> unclaimed_mesh_0_nodes;
    for (const auto& coord : mesh_0_coord_range) {
        auto chip_id = mesh_graph.coordinate_to_chip(mesh_ids[0], coord);
        tt::tt_fabric::FabricNodeId fn(mesh_ids[0], chip_id);
        if (!used_nodes.contains(fn)) {
            unclaimed_mesh_0_nodes.push_back(fn);
        }
    }
    TT_FATAL(
        unclaimed_mesh_0_nodes.size() >= 2,
        "Need at least 2 unclaimed nodes on mesh {} for stage 0 entry and loopback exit, found {}",
        *mesh_ids[0],
        unclaimed_mesh_0_nodes.size());

    std::optional<tt::tt_fabric::FabricNodeId> stage_0_entry_fn;
    std::optional<tt::tt_fabric::FabricNodeId> loopback_exit_fn;
    bool found_non_z_pair = false;
    for (std::size_t a = 0; a < unclaimed_mesh_0_nodes.size() && !found_non_z_pair; a++) {
        auto fn_a = unclaimed_mesh_0_nodes[a];
        for (const auto& [chan_id, direction] : control_plane.get_active_fabric_eth_channels(fn_a)) {
            auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(fn_a, chan_id);
            (void)peer_chan;
            if (peer_fn.mesh_id != mesh_ids[0]) {
                continue;
            }
            bool peer_unclaimed = !used_nodes.contains(peer_fn) &&
                                  std::find(unclaimed_mesh_0_nodes.begin(), unclaimed_mesh_0_nodes.end(), peer_fn) !=
                                      unclaimed_mesh_0_nodes.end();
            if (!peer_unclaimed) {
                continue;
            }
            bool is_non_z = (direction != tt::tt_fabric::eth_chan_directions::Z);
            if (!loopback_exit_fn.has_value() || (is_non_z && !found_non_z_pair)) {
                loopback_exit_fn = fn_a;
                stage_0_entry_fn = peer_fn;
                found_non_z_pair = is_non_z;
            }
        }
    }
    TT_FATAL(
        loopback_exit_fn.has_value() && stage_0_entry_fn.has_value(),
        "Could not find a directly-connected unclaimed pair on mesh {} for loopback exit -> stage 0 entry",
        *mesh_ids[0]);

    const auto& ctx = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    if (*ctx->rank() == 0) {
        log_info(
            tt::LogMetal,
            "Blitz loopback pair: exit={} -> entry={}, non_z={}",
            *loopback_exit_fn,
            *stage_0_entry_fn,
            found_non_z_pair);
    }

    std::vector<BlitzDecodePipelineStage> stages;
    stages.reserve(num_meshes + 1);

    // Stage 0: entry is intra-mesh (from loopback), exit goes to mesh_1 via hop[0]
    stages.emplace_back(BlitzDecodePipelineStage{
        .stage_index = static_cast<std::size_t>(*mesh_ids[0]),
        .entry_node_coord = fn_to_coord(*stage_0_entry_fn),
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
        .exit_node_coord = fn_to_coord(*loopback_exit_fn)});

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

    // 2b. Entry/exit fabric nodes chosen for the pipeline are not reused across stages.
    std::unordered_set<FabricNodeId> used_fabric_nodes;
    used_fabric_nodes.reserve(stages.size() * 2);
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        auto mesh_id = MeshId{static_cast<uint32_t>(s.stage_index)};
        FabricNodeId entry_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.entry_node_coord));
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.exit_node_coord));
        TT_FATAL(
            used_fabric_nodes.insert(entry_fn).second,
            "Stage [{}] entry fabric node {} is reused across stages",
            i,
            entry_fn);
        TT_FATAL(
            used_fabric_nodes.insert(exit_fn).second,
            "Stage [{}] exit fabric node {} is reused across stages",
            i,
            exit_fn);
    }

    // 3a. Each stage entry and exit must have at least one active fabric ethernet channel (none empty).
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        auto mesh_id = MeshId{static_cast<uint32_t>(s.stage_index)};
        FabricNodeId entry_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.entry_node_coord));
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.exit_node_coord));
        TT_FATAL(
            !control_plane.get_active_fabric_eth_channels(entry_fn).empty(),
            "Stage [{}] entry fabric node {} has no active fabric ethernet channels",
            i,
            entry_fn);
        TT_FATAL(
            !control_plane.get_active_fabric_eth_channels(exit_fn).empty(),
            "Stage [{}] exit fabric node {} has no active fabric ethernet channels",
            i,
            exit_fn);
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

    // 3b. Stage exit -> next stage entry (full ring, including last -> stage 0): ethernet direction kind matches
    //     on each hop (Z with Z, or planar N/E/S/W with N/E/S/W).
    using EthDir = tt::tt_fabric::eth_chan_directions;
    auto is_z_eth_dir = [](EthDir d) { return d == EthDir::Z; };
    auto is_nesw_eth_dir = [](EthDir d) {
        return d == EthDir::NORTH || d == EthDir::SOUTH || d == EthDir::EAST || d == EthDir::WEST;
    };
    auto eth_dirs_match_kind = [&](EthDir a, EthDir b) {
        return (is_z_eth_dir(a) && is_z_eth_dir(b)) || (is_nesw_eth_dir(a) && is_nesw_eth_dir(b));
    };

    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& stage = stages[i];
        const std::size_t next_i = (i + 1) % stages.size();
        const auto& next_stage = stages[next_i];

        auto mesh_id = MeshId{static_cast<uint32_t>(stage.stage_index)};
        auto next_mesh_id = MeshId{static_cast<uint32_t>(next_stage.stage_index)};
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, stage.exit_node_coord));
        FabricNodeId next_entry_fn(
            next_mesh_id, mesh_graph.coordinate_to_chip(next_mesh_id, next_stage.entry_node_coord));

        bool saw_hop = false;
        for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
            auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
            if (peer_fn != next_entry_fn) {
                continue;
            }
            saw_hop = true;
            EthDir dst_dir = control_plane.get_eth_chan_direction(peer_fn, static_cast<int>(peer_chan));
            TT_FATAL(
                eth_dirs_match_kind(src_dir, dst_dir),
                "Stages [{}] exit -> [{}] entry: ethernet direction mismatch {} -> {} (src_chan={} src_dir={} "
                "peer_chan={} dst_dir={})",
                i,
                next_i,
                exit_fn,
                next_entry_fn,
                src_chan,
                static_cast<int>(src_dir),
                peer_chan,
                static_cast<int>(dst_dir));
        }
        TT_FATAL(
            saw_hop,
            "Stages [{}] exit {} has no active fabric ethernet hop to stage [{}] entry {}",
            i,
            exit_fn,
            next_i,
            next_entry_fn);
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
    log_blitz_decode_pipeline_stages(stages);

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
