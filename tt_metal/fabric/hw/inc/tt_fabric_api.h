// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/risc_attribs.h"
#include <hostdevcommon/common_values.hpp>
#include "api/dataflow/dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "internal/ethernet/dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include "hostdevcommon/fabric_common.h"
#include "internal/tt-1xx/risc_common.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include <array>
#include <type_traits>

using namespace tt::tt_fabric;

namespace tt::tt_fabric {

// Type alias for cleaner access to 2D mesh routing constants
using MeshRoutingFields = RoutingFieldsConstants::Mesh;

inline constexpr std::array<std::uint8_t, static_cast<std::size_t>(eth_chan_directions::COUNT)>
    single_hop_route_cmd_by_direction = {
        MeshRoutingFields::FORWARD_WEST,   // EAST
        MeshRoutingFields::FORWARD_EAST,   // WEST
        MeshRoutingFields::FORWARD_SOUTH,  // NORTH
        MeshRoutingFields::FORWARD_NORTH,  // SOUTH
        MeshRoutingFields::NOOP,           // Z
};

inline eth_chan_directions get_next_hop_router_direction(uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    tt_l1_ptr routing_l1_info_t* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    if (dst_mesh_id == routing_table->my_mesh_id) {
        return static_cast<eth_chan_directions>(
            routing_table->intra_mesh_direction_table.get_original_direction(dst_dev_id));
    } else {
        return static_cast<eth_chan_directions>(
            routing_table->inter_mesh_direction_table.get_original_direction(dst_mesh_id));
    }
}

// Contract: the destination is the final destination and is exactly one EWNS physical fabric hop away.
// Do not use this helper for Z / inter-mesh traffic, which still relies on router recompute metadata.
void fabric_set_single_hop_unicast_route_from_direction(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    eth_chan_directions next_hop_direction,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id) {
    ASSERT(next_hop_direction != eth_chan_directions::Z);

    const auto dir_idx = static_cast<std::uint8_t>(next_hop_direction);
    ASSERT(dir_idx < eth_chan_directions::COUNT);

    // Edge-router recompute still inspects dst_start_node_id and mcast_params_64 on same-mesh worker traffic.
    packet_header->mcast_params_64 = 0;
    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    packet_header->routing_fields.value = 0;
    packet_header->route_buffer[0] = single_hop_route_cmd_by_direction[dir_idx];
}

void fabric_set_single_hop_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id) {
    fabric_set_single_hop_unicast_route_from_direction(
        packet_header, get_next_hop_router_direction(dst_mesh_id, dst_dev_id), dst_dev_id, dst_mesh_id);
}

template <bool mcast = false>
void fabric_set_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    eth_chan_directions direction,
    uint32_t branch_forward,
    uint32_t start_hop,
    uint32_t num_hops,
    bool terminate = false) {
    uint32_t local_packet = 0;
    uint32_t forward_packet = 0;
    uint32_t value = 0;
    switch (direction) {
        case eth_chan_directions::EAST:
            local_packet = MeshRoutingFields::FORWARD_WEST;
            forward_packet = MeshRoutingFields::FORWARD_EAST;
            if constexpr (mcast) {
                packet_header->routing_fields.branch_east_offset = start_hop;
            } else {
                packet_header->routing_fields.branch_east_offset = start_hop + 1;
            }
            break;
        case eth_chan_directions::WEST:
            local_packet = MeshRoutingFields::FORWARD_EAST;
            forward_packet = MeshRoutingFields::FORWARD_WEST;
            if constexpr (mcast) {
                packet_header->routing_fields.branch_west_offset = start_hop;
            } else {
                packet_header->routing_fields.branch_west_offset = start_hop + 1;
            }
            break;
        case eth_chan_directions::NORTH:
            local_packet = MeshRoutingFields::FORWARD_SOUTH;
            forward_packet = MeshRoutingFields::FORWARD_NORTH | branch_forward;
            break;
        case eth_chan_directions::SOUTH:
            local_packet = MeshRoutingFields::FORWARD_NORTH;
            forward_packet = MeshRoutingFields::FORWARD_SOUTH | branch_forward;
            break;
        default: ASSERT(false);
    }

    volatile tt_l1_ptr uint8_t* route_vector = packet_header->route_buffer;
    uint32_t local_val;
    uint32_t forward_val;
    uint32_t end_hop = start_hop + num_hops;
    ASSERT(end_hop <= FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE);
    for (uint32_t i = start_hop; i < end_hop; i++) {
        if constexpr (mcast) {
            // If forward north or forward south is set, then it may be 2d mcast and requires east/west forwarding, in
            // addition to spine forwards on north/south. forward_packet bit 0 and 1 determine if mcast has to branch
            // east/west from spine. If this is not a north/south mcast, then it cannot be a 2D mcast, and we dont need
            // to branch.
            uint32_t mcast_branch = forward_packet & MeshRoutingFields::WRITE_AND_FORWARD_NS
                                        ? forward_packet & MeshRoutingFields::WRITE_AND_FORWARD_EW
                                        : 0;
            forward_val = i == end_hop - 1 ? mcast_branch : forward_packet;
            local_val = local_packet;
        } else {
            forward_val = terminate ? (i == end_hop - 1 ? 0 : forward_packet) : forward_packet;
            local_val = terminate ? (i == end_hop - 1 ? local_packet : 0) : 0;
        }
        route_vector[i] = local_val | forward_val;
    }
    packet_header->routing_fields.hop_index = 0;
}

template <bool called_from_router = false, eth_chan_directions my_direction = eth_chan_directions::COUNT>
bool fabric_set_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id = MAX_NUM_MESHES);

// Defined in the indexed codec section below; the express worker path delegates to it.
inline bool fabric_set_indexed_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size);

#if defined(FABRIC_EXPRESS_ENABLED) && !defined(FABRIC_EXPRESS_MESH_Y_SIZE)
// The host emits the shape defines only to worker kernels; the ERISC compile selects the ABI with
// its own named args and never instantiates the worker flavor, so zeros are sufficient here.
#define FABRIC_EXPRESS_MESH_Y_SIZE 0
#define FABRIC_EXPRESS_MESH_X_SIZE 0
#endif

#if defined(FABRIC_EXPRESS_ENABLED) && (FABRIC_EXPRESS_MESH_Y_SIZE > 0)
// widen_indexed_route_to_chip fills route_buffer[0..Y) with the Y map and route_buffer[Y..Y+X) with
// the X map, so the header needs Y + X bytes -- two more than the (Y-1) + (X-1) hop count the buffer
// tiers were originally sized from. FabricContext::compute_packet_specifications accounts for that,
// and this catches any future mismatch at build time: the runtime ASSERT guarding the same condition
// inside widen_indexed_route_to_chip compiles out of release kernels, so without this a shape that
// outgrows its buffer writes past the end of the packet header instead of failing.
static_assert(
    FABRIC_EXPRESS_MESH_Y_SIZE + FABRIC_EXPRESS_MESH_X_SIZE <= sizeof(HybridMeshPacketHeader::route_buffer),
    "Express mesh shape requires a larger 2D route buffer than the packet header provides.");
#endif

template <bool called_from_router = false>
void fabric_set_mcast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops) {
    uint32_t spine_hops = 0;
    uint32_t mcast_branch = 0;
    packet_header->routing_fields.value = 0;
    if constexpr (!called_from_router) {
        tt_l1_ptr routing_l1_info_t* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
        packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
        packet_header->mcast_params_64 = ((uint64_t)s_num_hops << 48) | ((uint64_t)n_num_hops << 32) |
                                         ((uint64_t)w_num_hops << 16) | ((uint64_t)e_num_hops);
        packet_header->is_mcast_active = 0;
        if (routing_table->my_mesh_id != dst_mesh_id) {
            // TODO: refactoring
            fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
            packet_header->mcast_params_64 = ((uint64_t)s_num_hops << 48) | ((uint64_t)n_num_hops << 32) |
                                             ((uint64_t)w_num_hops << 16) | ((uint64_t)e_num_hops);
            return;
        }
    }
#if defined(FABRIC_EXPRESS_ENABLED)
    if constexpr (!called_from_router) {
        // Same-mesh 2D mcast has no indexed encode yet, and the indexed kernel decode would
        // misread a legacy spine/branch hop program. Remote-mesh carriers already returned above
        // (that leg is unicast-style and takes the indexed path).
        ASSERT(false);
    }
#endif

    // For 2D Mcast, mcast spine runs N/S and branches are E/W
    // If api is called with east and/or west hops != 0, it may be a 2D mcast
    // If so, set the forwarding flags for east and/or west branches.
    if (e_num_hops) {
        mcast_branch |= MeshRoutingFields::FORWARD_EAST;
    }
    if (w_num_hops) {
        mcast_branch |= MeshRoutingFields::FORWARD_WEST;
    }

    if (n_num_hops) {
        // Is a 2D mcast if mcast_branch != 0
        fabric_set_route<true>(packet_header, eth_chan_directions::NORTH, mcast_branch, 0, n_num_hops);
        spine_hops = n_num_hops;
    } else if (s_num_hops) {
        // Is a 2D mcast if mcast_branch != 0
        fabric_set_route<true>(packet_header, eth_chan_directions::SOUTH, mcast_branch, 0, s_num_hops);
        spine_hops = s_num_hops;
    }
    if (e_num_hops) {
        // Is a line mcast if spine_hops == 0
        fabric_set_route<true>(packet_header, eth_chan_directions::EAST, 0, spine_hops, e_num_hops);
        spine_hops += e_num_hops;
    }
    if (w_num_hops) {
        // Is a line mcast if spine_hops == 0
        fabric_set_route<true>(packet_header, eth_chan_directions::WEST, 0, spine_hops, w_num_hops);
    }
}

#if defined(COMPILE_FOR_ERISC)
// Called only from fabric_erisc_router.cpp
void fabric_set_mcast_route(volatile tt_l1_ptr HybridMeshPacketHeader* packet_header) {
    auto e_num_hops = packet_header->mcast_params[eth_chan_directions::EAST];
    auto w_num_hops = packet_header->mcast_params[eth_chan_directions::WEST];
    auto n_num_hops = packet_header->mcast_params[eth_chan_directions::NORTH];
    auto s_num_hops = packet_header->mcast_params[eth_chan_directions::SOUTH];
    e_num_hops = e_num_hops > 0 ? e_num_hops + 1 : 0;
    w_num_hops = w_num_hops > 0 ? w_num_hops + 1 : 0;
    n_num_hops = n_num_hops > 0 ? n_num_hops + 1 : 0;
    s_num_hops = s_num_hops > 0 ? s_num_hops + 1 : 0;
    fabric_set_mcast_route<true>(
        packet_header,
        packet_header->dst_start_chip_id,
        packet_header->dst_start_mesh_id,
        e_num_hops,
        w_num_hops,
        n_num_hops,
        s_num_hops);
}
#endif

uint8_t get_router_direction(uint32_t eth_channel) {
    tt_l1_ptr tensix_fabric_connections_l1_info_t* connection_info =
        reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(FABRIC_CONNECTIONS_BASE);
    return connection_info->read_only[eth_channel].edm_direction;
}

// Overload: Fill route_buffer of HybridMeshPacketHeader and initialize hop_index/branch offsets for 2D.
template <bool called_from_router, eth_chan_directions my_direction>
bool fabric_set_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id) {
#if defined(FABRIC_EXPRESS_ENABLED)
    if constexpr (!called_from_router) {
        // Workers on express meshes widen the destination's indexed action maps; the mesh shape
        // comes from the per-kernel FABRIC_EXPRESS_MESH_*_SIZE defines. called_from_router (the
        // edge rewrite path) keeps the legacy hop-program encode below.
        return fabric_set_indexed_unicast_route(
            packet_header, dst_dev_id, dst_mesh_id, FABRIC_EXPRESS_MESH_Y_SIZE, FABRIC_EXPRESS_MESH_X_SIZE);
    }
#endif
    if constexpr (!called_from_router) {
        packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
        packet_header->mcast_params_64 = 0;
        packet_header->is_mcast_active = 0;
    }
    auto* routing_info = reinterpret_cast<tt_l1_ptr intra_mesh_routing_path_t<2, true>*>(ROUTING_PATH_BASE_2D);
    auto* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    if (dst_mesh_id < MAX_NUM_MESHES && routing_table->my_mesh_id != dst_mesh_id) {
        auto exit_node_table = reinterpret_cast<tt_l1_ptr uint8_t*>(EXIT_NODE_TABLE_BASE);
        dst_dev_id = exit_node_table[dst_mesh_id];
        dst_mesh_id = routing_table->my_mesh_id;
    }
    bool ok = false;
    if constexpr (called_from_router) {
        // This is to prepend additional one step, which is not needed for worker sender.
        auto set_forward = [&](eth_chan_directions dir) {
            switch (dir) {
                case eth_chan_directions::EAST: packet_header->route_buffer[0] = MeshRoutingFields::FORWARD_EAST; break;
                case eth_chan_directions::WEST: packet_header->route_buffer[0] = MeshRoutingFields::FORWARD_WEST; break;
                case eth_chan_directions::NORTH:
                    packet_header->route_buffer[0] = MeshRoutingFields::FORWARD_NORTH;
                    break;
                case eth_chan_directions::SOUTH:
                    packet_header->route_buffer[0] = MeshRoutingFields::FORWARD_SOUTH;
                    break;
                case eth_chan_directions::Z:
                    // Z exit port will use NOOP to indicate forward to Z
                    packet_header->route_buffer[0] = MeshRoutingFields::NOOP;
                    break;
                default: ASSERT(false); break;
            }
        };
        eth_chan_directions next_direction = get_next_hop_router_direction(dst_mesh_id, dst_dev_id);
        if (next_direction < eth_chan_directions::COUNT) {
            // when arrive at another mesh, but dst chip is not itself. -> go to next chip -> prepend FORWARD_<DIR> ->
            // add route
            ok = routing_info->decode_route_to_buffer(dst_dev_id, packet_header->route_buffer, true);
        } else {
            if (routing_table->my_mesh_id == packet_header->dst_start_mesh_id) {
                // when arrive at destination mesh, and dst chip is itself. -> DRAIN -> prepend FORWARD_<DIR> -> done
                set_forward(my_direction);
            } else {
                // when arrive at non-destination mesh, but dst chip is itself (exit node). -> go to next mesh ->
                // prepend FORWARD_<DIR> -> done
                next_direction =
                    get_next_hop_router_direction(packet_header->dst_start_mesh_id, packet_header->dst_start_chip_id);
                set_forward(next_direction);
            }
            packet_header->route_buffer[1] = MeshRoutingFields::NOOP;
            return true;  // early return, route_buffer[0] is enough
        }
    } else {
        ok = routing_info->decode_route_to_buffer(dst_dev_id, packet_header->route_buffer);
    }
    packet_header->routing_fields.value = 0;

    const auto& compressed_route = routing_info->paths[dst_dev_id];
    uint8_t ns_hops = compressed_route.get_ns_hops();
    uint8_t ew_hops = compressed_route.get_ew_hops();
    uint8_t ew_direction = compressed_route.get_ew_direction();
    uint8_t turn_point = compressed_route.get_turn_point() + called_from_router;

    if (ns_hops > 0 && ew_hops > 0) {
        // 2D routing: turn from NS to EW at turn_point
        if (ew_direction) {
            packet_header->routing_fields.branch_east_offset = turn_point;  // turn to EAST after NS
        } else {
            packet_header->routing_fields.branch_west_offset = turn_point;  // turn to WEST after NS
        }
    } else if (ns_hops > 0) {
        packet_header->routing_fields.branch_east_offset = turn_point;
        packet_header->routing_fields.branch_west_offset = turn_point;
    } else if (ns_hops == 0 && ew_hops > 0) {
        // East/West only routing: branch offset is set at position 1 (start_hop + 1)
        if (ew_direction) {
            packet_header->routing_fields.branch_east_offset = 1;  // East only: branch at hop 1
        } else {
            packet_header->routing_fields.branch_west_offset = 1;  // West only: branch at hop 1
        }
    } else if (ns_hops == 0 && ew_hops == 0) {
        // NOTE: this is not needed from functionality perspective, but just to follow original behavior
        packet_header->routing_fields.branch_west_offset = 1;
    }

    return ok;
}

// ============================================================================
// Indexed 2D route codec (destination-indexed ABI) — worker/edge producers
// ============================================================================
// Target encode for 2D mesh routing: the packet carries widened action-byte maps,
// route_buffer[0..Y) = route_buffer_y then route_buffer[Y..Y+X) = route_buffer_x (contiguous
// layout), instead of a hop program + branch offsets.

// Widen core shared by the worker unicast producer and the intermesh landing encoder: install
// destination dst_dev_id's action maps from the given vector table — the Y row
// widened one-hot into route_buffer[0..Y), the X row into route_buffer[Y..Y+X), LOCAL_DELIVER OR-ed
// onto the destination's X slot. The installed maps are a pure function of the destination
// (destination-indexed, suffix-consistent tables), so the landing encoder reuses this verbatim with
// no notion of encode_root.
inline void widen_indexed_route_to_chip(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    const std::uint8_t* vectors,
    uint16_t dst_dev_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    ASSERT(dst_dev_id < (uint32_t)mesh_y_size * mesh_x_size);
    ASSERT((uint32_t)mesh_y_size + mesh_x_size <= sizeof(packet_header->route_buffer));

    const uint32_t dst_y = dst_dev_id / mesh_x_size;
    const uint32_t dst_x = dst_dev_id % mesh_x_size;

    const std::uint8_t* y_vec = IndexedMeshRoutingFields::y_row(vectors, mesh_y_size, dst_y);
    for (uint32_t i = 0; i < mesh_y_size; ++i) {
        packet_header->route_buffer[i] =
            IndexedMeshRoutingFields::widen_y(IndexedMeshRoutingFields::get_action_2bit(y_vec, i));
    }
    const std::uint8_t* x_vec = IndexedMeshRoutingFields::x_row(vectors, mesh_y_size, mesh_x_size, dst_x);
    for (uint32_t i = 0; i < mesh_x_size; ++i) {
        packet_header->route_buffer[mesh_y_size + i] =
            IndexedMeshRoutingFields::widen_x(IndexedMeshRoutingFields::get_action_2bit(x_vec, i));
    }
    // Full-widen delivery marker lives only on the X slot: at dst_y the Y row widens to STOP (0),
    // and decode falls through to route_buffer_x.
    packet_header->route_buffer[mesh_y_size + dst_x] |= IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
}

// Full unicast widen. mesh_{y,x}_size are the local mesh shape; callers supply them from
// their own compile-time context (device-side id->coord math mirrors MeshGraph::chip_to_coordinate:
// coord = (id / x_size, id % x_size)). Always returns true; the widen is total and violations fail
// ASSERTs instead.
inline bool fabric_set_indexed_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    tt_l1_ptr routing_l1_info_t* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    // Final destination is retained up front and never overwritten by the exit swap below.
    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    packet_header->mcast_params_64 = 0;
    packet_header->is_mcast_active = 0;
    packet_header->routing_fields.value = 0;

    if (dst_mesh_id < MAX_NUM_MESHES && routing_table->my_mesh_id != dst_mesh_id) {
        // Remote final mesh: widen unicast-style maps to the temporary exit chip so the maps decode
        // to exactly LOCAL_DELIVER at the exit chip.
        auto exit_node_table = reinterpret_cast<tt_l1_ptr uint8_t*>(EXIT_NODE_TABLE_BASE);
        dst_dev_id = exit_node_table[dst_mesh_id];
    }

    widen_indexed_route_to_chip(
        packet_header, routing_table->indexed_route_vectors.data, dst_dev_id, mesh_y_size, mesh_x_size);
    return true;
}

// Intermesh landing encode. Runs on the boundary-facing router BEFORE ordinary decode: the packet
// still carries source-mesh maps, so they are re-installed from THIS mesh's own vector table.
// Intermediate landing widens unicast-style maps toward this mesh's next exit for the final mesh
// (exit_node_table); destination landing widens to the retained final chip. dst_start_node_id and
// mcast_params_64 are always preserved — temporary exits live only in the installed maps and must
// never overwrite the final destination.
inline void fabric_set_indexed_intermesh_landing_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    const routing_l1_info_t& routing_table,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    const uint16_t final_mesh_id = packet_header->dst_start_mesh_id;
    // Bounds before exit_node_table indexing.
    ASSERT(final_mesh_id < MAX_NUM_MESHES);
    uint16_t target_dev_id;
    if (final_mesh_id != routing_table.my_mesh_id) {
        // Intermediate landing: this mesh's next exit toward the final mesh. Multicast-safe as-is:
        // anchor/extents stay retained and no target-range fanout begins at an intermediate mesh.
        target_dev_id = routing_table.exit_node_table[final_mesh_id];
        ASSERT(target_dev_id != (uint16_t)eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
    } else {
        // Destination landing: widen to the final destination. Destination-landing multicast is not
        // implemented; producers emit unicast only, so a multicast carrier here fail-stops.
        ASSERT(packet_header->mcast_params_64 == 0);
        target_dev_id = packet_header->dst_start_chip_id;
    }
    widen_indexed_route_to_chip(
        packet_header, routing_table.indexed_route_vectors.data, target_dev_id, mesh_y_size, mesh_x_size);
}

// Single-hop poke. Same client contract as the legacy single-hop helpers (dest is exactly
// one fabric hop away) but destination-indexed: writes LOCAL_DELIVER at the destination's map
// slot on the hop's axis. No source-slot action is written — the source chip never decodes; the
// worker's choice of which local router to push the packet to is the hop. Z is allowed when the
// express topology provides it (the legacy helper ASSERTs against Z because hop programs had no
// Z command).
inline void fabric_set_indexed_single_hop_unicast_route_from_direction(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    eth_chan_directions next_hop_direction,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    ASSERT(next_hop_direction < eth_chan_directions::COUNT);
    ASSERT(dst_dev_id < (uint32_t)mesh_y_size * mesh_x_size);
    ASSERT((uint32_t)mesh_y_size + mesh_x_size <= sizeof(packet_header->route_buffer));

    const uint32_t dst_y = dst_dev_id / mesh_x_size;
    const uint32_t dst_x = dst_dev_id % mesh_x_size;

    // Clear both axis maps: pool headers are reused, and a router executes whatever bits it finds
    // in the slot its facing selects, so a stale bit in an unpoked slot is a silently valid extra
    // action. A wrong-link arrival or a multi-hop call lands on exactly such an unpoked slot;
    // zeroed maps fail-stop there instead.
    for (uint32_t i = 0; i < (uint32_t)mesh_y_size + mesh_x_size; ++i) {
        packet_header->route_buffer[i] = 0;
    }

    // The receiving router faces back along the arrival link, so its decode axis matches the hop
    // axis: N/S/Z hops land on N/S/Z-facing routers (Y byte first), E/W hops on E/W-facing routers
    // (X byte only).
    switch (next_hop_direction) {
        case eth_chan_directions::NORTH:
        case eth_chan_directions::SOUTH:
        case eth_chan_directions::Z:
            packet_header->route_buffer[dst_y] = IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
            break;
        case eth_chan_directions::EAST:
        case eth_chan_directions::WEST:
            packet_header->route_buffer[mesh_y_size + dst_x] = IndexedMeshRoutingFields::ACTION_LOCAL_DELIVER;
            break;
        default: ASSERT(false); break;
    }

    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    packet_header->mcast_params_64 = 0;
    packet_header->is_mcast_active = 0;
    packet_header->routing_fields.value = 0;
}

inline void fabric_set_indexed_single_hop_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    fabric_set_indexed_single_hop_unicast_route_from_direction(
        packet_header,
        get_next_hop_router_direction(dst_mesh_id, dst_dev_id),
        dst_dev_id,
        dst_mesh_id,
        mesh_y_size,
        mesh_x_size);
}

// Overload: For 1D LowLatencyPacketHeader
// 1D need to choose between target_as_dev true/false and compressed true/false
// TODO: compare performance of compressed true/false
//       https://github.com/tenstorrent/tt-metal/issues/29449
template <bool target_as_dev = true, bool compressed = true>
bool fabric_set_unicast_route(volatile tt_l1_ptr LowLatencyPacketHeader* packet_header, uint16_t target_num) {
    if constexpr (compressed) {
        if constexpr (target_as_dev) {
            return decode_route_to_buffer_by_dev(target_num, (volatile uint8_t*)&packet_header->routing_fields.value);
        } else {
            return decode_route_to_buffer_by_hops(target_num, (volatile uint8_t*)&packet_header->routing_fields.value);
        }
    } else {
#if defined(COMPILE_FOR_ERISC)
        static_assert(!target_as_dev, "ACTIVE_ETH doesn't support device id based routing yet");
#endif
        auto* routing_info =
            reinterpret_cast<tt_l1_ptr intra_mesh_routing_path_t<1, compressed>*>(ROUTING_PATH_BASE_1D);
        auto* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
        if constexpr (target_as_dev) {
            uint16_t my_device_id = routing_table->my_device_id;
            uint16_t hops = my_device_id > target_num ? my_device_id - target_num : target_num - my_device_id;
            return routing_info->decode_route_to_buffer(hops, (volatile uint8_t*)&packet_header->routing_fields.value);
        } else {
            return routing_info->decode_route_to_buffer(
                target_num, (volatile uint8_t*)&packet_header->routing_fields.value);
        }
    }
}

// 1D sparse multicast
template <typename HopMaskType>  // HopMaskType is uint8_t, uint16_t, uint32_t, or uint64_t
void fabric_set_sparse_multicast_route(volatile tt_l1_ptr LowLatencyPacketHeader* packet_header, HopMaskType hop_mask) {
    uint32_t temp_routing_fields;
    routing_encoding::encode_1d_sparse_multicast(hop_mask, temp_routing_fields);

    // Copy to volatile output
    packet_header->routing_fields.value = temp_routing_fields;
}

}  // namespace tt::tt_fabric
