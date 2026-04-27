// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/experimental/blitz_decode_pipeline.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>

#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental::blitz {

using ::tt::tt_fabric::FabricNodeId;
using ::tt::tt_fabric::MeshId;
using ::tt::tt_metal::distributed::MeshCoordinate;

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

    std::vector<BlitzDecodePipelineStage> stages;
    stages.reserve(initialize_loopback ? num_meshes + 1 : num_meshes);

    // Stage 0: entry is intra-mesh, exit goes to mesh_1 via hop[0]
    stages.emplace_back(BlitzDecodePipelineStage{
        .stage_index = static_cast<std::size_t>(*mesh_ids[0]),
        .entry_node_coord = fn_to_coord(*stage_0_entry_fn),
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
        stages.emplace_back(BlitzDecodePipelineStage{
            .stage_index = static_cast<std::size_t>(*mesh_ids[0]),
            .entry_node_coord = fn_to_coord(hops[num_meshes - 1].second),
            .exit_node_coord = fn_to_coord(*loopback_exit_fn)});
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
    if (initialize_loopback) {
        const auto& stage_0 = stages[0];
        const auto& loopback = stages.back();
        TT_FATAL(
            loopback.entry_node_coord != stage_0.entry_node_coord || loopback.exit_node_coord != stage_0.exit_node_coord,
            "Loopback stage has identical entry/exit as stage 0: entry={}, exit={}",
            coord_str(loopback.entry_node_coord),
            coord_str(loopback.exit_node_coord));
    }

    // 5. Router symmetry and physical cable correctness for every inter-mesh hop.
    //    Validates that:
    //    (a) Both sides have active router channels on matching physical ethernet channels
    //    (b) Direction kinds match (Z-Z or NESW-NESW)
    //    (c) The physical cable actually connects the claimed ASICs on the claimed
    //        (src_chan, peer_chan) pair — not just src_chan. After the
    //        intermesh_chan_to_peer_ redesign, get_connected_mesh_chip_chan_ids returns
    //        the PHYSICAL peer channel for inter-mesh links, sourced from PSD.
    //    (d) Per-chip Z-direction goes to at most one neighbor mesh
    //    (e) Reverse-lookup symmetry: peer_fn:peer_chan resolves back to exit_fn:src_chan.
    //        Catches asymmetric intermesh_chan_to_peer_ entries left over from
    //        clear-and-rebuild reconciliation.
    //    (f) No double-claim: no physical (asic, chan) is used by two different pipeline
    //        hops. A physical channel can only carry one inter-mesh route.
    //
    //    An asymmetric configuration causes the ERISC handshake to hang at STARTED.
    //    A wrong physical channel (from stolen Z port reconciliation bugs) causes the
    //    router to handshake with the wrong peer or a peer that has no router.
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto& topology_mapper = control_plane.get_topology_mapper();

    // Track per-chip Z neighbors to detect one-Z-neighbor-per-chip violations
    std::map<std::pair<uint32_t, ChipId>, std::set<uint32_t>> chip_z_neighbors;

    // Track physical channel ownership for double-claim detection. Key: (asic, chan)
    // physical endpoint; Value: (stage_index, role) that first claimed it.
    std::map<std::pair<tt::tt_metal::AsicID, tt::tt_fabric::chan_id_t>, std::pair<std::size_t, const char*>>
        physical_chan_owner;

    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& stage = stages[i];
        const std::size_t next_i = (i + 1) % stages.size();
        const auto& next_stage = stages[next_i];

        auto curr_mesh_id = MeshId{static_cast<uint32_t>(stage.stage_index)};
        auto next_mesh_id = MeshId{static_cast<uint32_t>(next_stage.stage_index)};

        if (curr_mesh_id == next_mesh_id) {
            continue;
        }

        FabricNodeId exit_fn(curr_mesh_id, mesh_graph.coordinate_to_chip(curr_mesh_id, stage.exit_node_coord));
        FabricNodeId entry_fn(next_mesh_id, mesh_graph.coordinate_to_chip(next_mesh_id, next_stage.entry_node_coord));

        auto exit_asic = topology_mapper.get_asic_id_from_fabric_node_id(exit_fn);
        auto entry_asic = topology_mapper.get_asic_id_from_fabric_node_id(entry_fn);
        auto eth_conns = psd.get_eth_connections(exit_asic, entry_asic);

        bool exit_has_channel_to_entry = false;
        for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
            auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
            if (peer_fn != entry_fn) {
                continue;
            }
            exit_has_channel_to_entry = true;

            // (5a) Peer must have a matching active router
            auto entry_channels = control_plane.get_active_fabric_eth_channels(entry_fn);
            bool entry_has_matching_channel = false;
            for (const auto& [entry_chan, entry_dir] : entry_channels) {
                if (entry_chan == peer_chan) {
                    entry_has_matching_channel = true;
                    break;
                }
            }
            TT_FATAL(
                entry_has_matching_channel,
                "Router symmetry violation at stage [{}] -> [{}]: exit {} chan={} connects to entry {} chan={}, "
                "but entry has no active router on that channel. "
                "The ERISC handshake will hang at STARTED.",
                i,
                next_i,
                exit_fn,
                src_chan,
                entry_fn,
                peer_chan);

            // (5b) Direction kinds must match
            using EthDir = tt::tt_fabric::eth_chan_directions;
            EthDir dst_dir = control_plane.get_eth_chan_direction(entry_fn, static_cast<int>(peer_chan));
            bool dirs_ok = (is_z_eth_dir(src_dir) && is_z_eth_dir(dst_dir)) ||
                           (is_nesw_eth_dir(src_dir) && is_nesw_eth_dir(dst_dir));
            TT_FATAL(
                dirs_ok,
                "Router direction mismatch at stage [{}] -> [{}]: exit {} chan={} dir={} -> entry {} chan={} dir={}",
                i,
                next_i,
                exit_fn,
                src_chan,
                eth_chan_dir_cstr(src_dir),
                entry_fn,
                peer_chan,
                eth_chan_dir_cstr(dst_dir));

            // (5c) Physical cable from src_chan on exit ASIC must go to entry ASIC,
            // and the (src_chan, peer_chan) pair must match a real PSD cable —
            // catches the multi-peer / stolen-Z-port bug where reconciliation maps
            // a port to a physically different cable.
            bool psd_has_cable_from_src_chan = false;
            bool psd_has_exact_pair = false;
            for (const auto& conn : eth_conns) {
                if (conn.src_chan == src_chan) {
                    psd_has_cable_from_src_chan = true;
                    if (conn.dst_chan == peer_chan) {
                        psd_has_exact_pair = true;
                        break;
                    }
                }
            }
            TT_FATAL(
                psd_has_cable_from_src_chan,
                "Physical cable mismatch at stage [{}] -> [{}]: exit {} chan={} has a router that "
                "routes toward entry {}, but PSD has no cable from ASIC {} chan={} to ASIC {}. "
                "The reconciliation likely mapped the port to a channel that physically connects "
                "to a different ASIC (stolen Z port_id bug).",
                i,
                next_i,
                exit_fn,
                src_chan,
                entry_fn,
                exit_asic,
                src_chan,
                entry_asic);
            TT_FATAL(
                psd_has_exact_pair,
                "Per-cable channel mismatch at stage [{}] -> [{}]: exit {} chan={} -> entry {} chan={}, but PSD "
                "has no cable matching that exact (src_chan, dst_chan) pair between ASICs {} and {}. "
                "intermesh_chan_to_peer_ disagrees with PSD on the physical cable identity.",
                i,
                next_i,
                exit_fn,
                src_chan,
                entry_fn,
                peer_chan,
                exit_asic,
                entry_asic);

            // (5e) Reverse-lookup symmetry: querying the peer endpoint must round-trip
            // back to the exit endpoint. The redesign populates intermesh_chan_to_peer_
            // symmetrically for both sides of each PSD cable; an asymmetric entry
            // indicates that one side was lost during clear-and-rebuild.
            auto [reverse_fn, reverse_chan] = control_plane.get_connected_mesh_chip_chan_ids(peer_fn, peer_chan);
            TT_FATAL(
                reverse_fn == exit_fn && reverse_chan == src_chan,
                "Reverse-lookup symmetry violation at stage [{}] -> [{}]: exit {} chan={} -> peer {} chan={}, "
                "but reverse query returns {} chan={}. intermesh_chan_to_peer_ entries are not symmetric "
                "for this physical cable.",
                i,
                next_i,
                exit_fn,
                src_chan,
                peer_fn,
                peer_chan,
                reverse_fn,
                reverse_chan);

            // (5f) No double-claim: each physical (asic, chan) endpoint of a cable
            // can only carry one inter-mesh route. If two pipeline hops claim the
            // same physical channel, one of them will silently misroute.
            auto exit_phys = std::make_pair(exit_asic, src_chan);
            auto entry_phys = std::make_pair(entry_asic, peer_chan);
            auto [exit_owner_it, exit_inserted] = physical_chan_owner.try_emplace(exit_phys, std::make_pair(i, "exit"));
            TT_FATAL(
                exit_inserted,
                "Pipeline hop double-claim at stage [{}] -> [{}]: exit {} (ASIC {}) chan={} is also claimed by "
                "stage [{}] as {}. A physical channel can only carry one inter-mesh route.",
                i,
                next_i,
                exit_fn,
                exit_asic,
                src_chan,
                exit_owner_it->second.first,
                exit_owner_it->second.second);
            auto [entry_owner_it, entry_inserted] =
                physical_chan_owner.try_emplace(entry_phys, std::make_pair(next_i, "entry"));
            TT_FATAL(
                entry_inserted,
                "Pipeline hop double-claim at stage [{}] -> [{}]: entry {} (ASIC {}) chan={} is also claimed by "
                "stage [{}] as {}. A physical channel can only terminate one inter-mesh route.",
                i,
                next_i,
                entry_fn,
                entry_asic,
                peer_chan,
                entry_owner_it->second.first,
                entry_owner_it->second.second);

            // (5d) Track Z-direction neighbors per chip
            if (src_dir == EthDir::Z) {
                chip_z_neighbors[{*curr_mesh_id, exit_fn.chip_id}].insert(*next_mesh_id);
            }
        }
        TT_FATAL(
            exit_has_channel_to_entry,
            "Stage [{}] exit {} has no active fabric ethernet channel connecting to stage [{}] entry {}. "
            "The inter-mesh hop will fail.",
            i,
            exit_fn,
            next_i,
            entry_fn);
    }

    // (5d cont.) Enforce one-Z-neighbor-per-chip invariant
    for (const auto& [chip_key, neighbor_meshes] : chip_z_neighbors) {
        TT_FATAL(
            neighbor_meshes.size() <= 1,
            "Chip {} in mesh {} has Z-direction connections to {} neighbor meshes. "
            "This violates the one-Z-neighbor-per-chip invariant and indicates "
            "a bug in Z-port promotion/reconciliation.",
            chip_key.second,
            chip_key.first,
            neighbor_meshes.size());
    }

    // 6. Routing-table forwarding is ready for every stage transition. The decode
    //    pipeline creates a MeshSocket from stage[i].exit → stage[(i+1)].entry and
    //    the runtime calls get_forwarding_direction / get_fabric_route to actually
    //    push packets. If the control plane knows about the cable but routing
    //    tables have a hole, the socket handshakes but data never arrives. The
    //    full ring (including the intra-mesh loopback last→stage_0) must resolve.
    //
    //    get_forwarding_direction is driven by mesh_graph's routing_table_generator
    //    and is globally valid; get_fabric_route / get_forwarding_eth_chans_to_chip
    //    read the host-local routing tables, so we gate them on exit_fn locality.
    auto local_meshes = control_plane.get_local_mesh_id_bindings();
    std::set<MeshId> local_mesh_set(local_meshes.begin(), local_meshes.end());
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& stage = stages[i];
        const std::size_t next_i = (i + 1) % stages.size();
        const auto& next_stage = stages[next_i];

        auto curr_mesh_id = MeshId{static_cast<uint32_t>(stage.stage_index)};
        auto next_mesh_id = MeshId{static_cast<uint32_t>(next_stage.stage_index)};
        FabricNodeId exit_fn(curr_mesh_id, mesh_graph.coordinate_to_chip(curr_mesh_id, stage.exit_node_coord));
        FabricNodeId next_entry_fn(
            next_mesh_id, mesh_graph.coordinate_to_chip(next_mesh_id, next_stage.entry_node_coord));

        auto fwd_dir = control_plane.get_forwarding_direction(exit_fn, next_entry_fn);
        TT_FATAL(
            fwd_dir.has_value(),
            "Stage [{}] -> [{}]: routing table has no forwarding direction from {} to {}. The socket handshake "
            "will succeed but packets will have no route.",
            i,
            next_i,
            exit_fn,
            next_entry_fn);

        if (!local_mesh_set.contains(exit_fn.mesh_id)) {
            continue;
        }

        auto fwd_chans = control_plane.get_forwarding_eth_chans_to_chip(exit_fn, next_entry_fn);
        TT_FATAL(
            !fwd_chans.empty(),
            "Stage [{}] -> [{}]: no forwarding eth channels from {} to {} despite the control plane advertising "
            "the pair. Pipeline socket would fail to pick a send channel.",
            i,
            next_i,
            exit_fn,
            next_entry_fn);

        for (tt::tt_fabric::chan_id_t fwd_chan : fwd_chans) {
            auto route = control_plane.get_fabric_route(exit_fn, next_entry_fn, fwd_chan);
            TT_FATAL(
                !route.empty(),
                "Stage [{}] -> [{}]: get_fabric_route returned empty for {} chan={} -> {}. Forwarding channel "
                "is advertised but routing tables cannot resolve an end-to-end path.",
                i,
                next_i,
                exit_fn,
                fwd_chan,
                next_entry_fn);
            TT_FATAL(
                route.back().first == next_entry_fn,
                "Stage [{}] -> [{}]: fabric route from {} chan={} terminates at {} instead of declared entry {}. "
                "Routing table hop sequence is inconsistent.",
                i,
                next_i,
                exit_fn,
                fwd_chan,
                route.back().first,
                next_entry_fn);
        }
    }
}

}  // namespace

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(bool initialize_loopback) {
    auto stages = build_pipeline_from_topology(initialize_loopback);
    validate_pipeline(stages, initialize_loopback);

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
