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
#include "fabric_2d_route_interface.h"
#include <array>
#include <type_traits>

using namespace tt::tt_fabric;

namespace tt::tt_fabric {

#if !defined(FABRIC_2D_MESH_Y_SIZE)
// Shape defines are emitted only to 2D worker kernels. Every other compile that includes this header
// still *parses* the 2D helpers below -- they are plain function bodies, not templates, so the
// compiler needs the names to exist even where the functions are never called. That covers the ERISC
// router (which never instantiates the worker path) and, importantly, non-2D builds such as the
// cq_dispatch / cq_prefetch kernels, which include this header without -DFABRIC_2D.
//
// Deliberately NOT gated on defined(FABRIC_2D): gating it there is what broke the dispatch kernels,
// because the guard tracked "is this a 2D build" while the actual requirement is "does this
// translation unit parse the 2D helpers", which is always.
#define FABRIC_2D_MESH_Y_SIZE 0
#define FABRIC_2D_MESH_X_SIZE 0
#endif

#if defined(FABRIC_2D) && (FABRIC_2D_MESH_Y_SIZE > 0)
static_assert(ROUTING_TABLE_BASE % alignof(std::uint32_t) == 0, "2D routing-table base must be word aligned");
// The Y and X maps occupy Y + X bytes, two more than the (Y-1) + (X-1) hop count the buffer tiers
// were sized from. Checked here because the equivalent runtime ASSERT compiles out of release
// kernels, where an oversized shape would instead write past the end of the header.
static_assert(
    FABRIC_2D_MESH_Y_SIZE + FABRIC_2D_MESH_X_SIZE <= sizeof(HybridMeshPacketHeader::route_buffer),
    "Express mesh shape requires a larger 2D route buffer than the packet header provides.");
#endif

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

// Defined in the 2D action-map section below; the public path delegates to it.
inline void fabric_set_2d_single_hop_unicast_route_from_direction(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    eth_chan_directions next_hop_direction,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size);

// Contract: the destination is the final destination, is in this mesh, and is exactly one physical
// fabric hop away. Do not use this helper for inter-mesh traffic -- it has no valid coordinates for a
// chip numbered in another mesh's id space.
void fabric_set_single_hop_unicast_route_from_direction(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    eth_chan_directions next_hop_direction,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id) {
    fabric_set_2d_single_hop_unicast_route_from_direction(
        packet_header, next_hop_direction, dst_dev_id, dst_mesh_id, FABRIC_2D_MESH_Y_SIZE, FABRIC_2D_MESH_X_SIZE);
}

void fabric_set_single_hop_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id) {
    fabric_set_single_hop_unicast_route_from_direction(
        packet_header, get_next_hop_router_direction(dst_mesh_id, dst_dev_id), dst_dev_id, dst_mesh_id);
}

template <bool called_from_router = false, eth_chan_directions my_direction = eth_chan_directions::COUNT>
bool fabric_set_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id = MAX_NUM_MESHES);

// Defined in the 2D action-map section below; the public worker path delegates to it.
inline bool fabric_set_2d_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size);

// Defined in the 2D action-map section below; the public worker path delegates to it.
inline std::uint8_t fabric_set_2d_mcast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size);

// Multicast producer. Extents are measured from the anchor; a same-mesh send encodes through this
// chip's reverse trees, a foreign final mesh takes a unicast-style carrier leg toward the exit.
//
// This API sends through one caller-chosen connection, so the root action must have at most one eth
// output -- the per-direction client contract (one operation per outgoing direction). A caller wanting
// the whole rectangle in one operation uses fabric_multicast_source_inject_*.
void fabric_set_mcast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops) {
    const std::uint8_t root_action = fabric_set_2d_mcast_route(
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        e_num_hops,
        w_num_hops,
        n_num_hops,
        s_num_hops,
        FABRIC_2D_MESH_Y_SIZE,
        FABRIC_2D_MESH_X_SIZE);
    const std::uint8_t root_outputs = root_action & Routing2DCodec::ACTION_ETH_MASK;
    ASSERT((root_outputs & (root_outputs - 1)) == 0);
}

uint8_t get_router_direction(uint32_t eth_channel) {
    tt_l1_ptr tensix_fabric_connections_l1_info_t* connection_info =
        reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(FABRIC_CONNECTIONS_BASE);
    return connection_info->read_only[eth_channel].edm_direction;
}

// Overload: Fill route_buffer of HybridMeshPacketHeader and initialize hop_index/branch offsets for 2D.
// 2D unicast. Widens the destination's action maps from this chip's destination-major L1 route table.
//
// The `called_from_router` / `my_direction` template parameters are vestigial: the router-side
// re-encode they selected was the legacy hop-program path, whose only caller (recompute_path in
// fabric_edge_node_router.hpp) is gone. Router-side re-encode is now the intermesh landing encoder,
// which the kernel calls directly. The parameters are retained so the ~149 existing call sites keep
// compiling; both are ignored.
template <bool called_from_router, eth_chan_directions my_direction>
bool fabric_set_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id) {
    static_assert(!called_from_router, "router-side re-encode is fabric_set_2d_intermesh_landing_route()");
    return fabric_set_2d_unicast_route(
        packet_header, dst_dev_id, dst_mesh_id, FABRIC_2D_MESH_Y_SIZE, FABRIC_2D_MESH_X_SIZE);
}

// ============================================================================
// 2D action-map routing (destination-major ABI) — worker/edge producers
// ============================================================================
// The packet carries widened action-byte maps instead of a hop program plus branch offsets:
// route_buffer[0..Y) holds the Y map, route_buffer[Y..Y+X) the X map.

// Unicast setup retains the final destination in packet metadata, then expands one destination-major
// table entry into the packet's [Y | X] map. A remote destination temporarily expands the local exit
// chip instead; the destination-mesh landing later rebuilds the final map.
inline bool fabric_set_2d_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    tt_l1_ptr routing_l1_info_t* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    // Final destination is retained up front and never overwritten by the exit swap below.
    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    packet_header->mcast_params_64 = 0;
    packet_header->routing_fields.value = 0;

    if (dst_mesh_id < MAX_NUM_MESHES && routing_table->my_mesh_id != dst_mesh_id) {
        // Remote final mesh: route to this mesh's exit chip, where the maps decode to LOCAL_DELIVER.
        auto exit_node_table = reinterpret_cast<tt_l1_ptr uint8_t*>(EXIT_NODE_TABLE_BASE);
        dst_dev_id = exit_node_table[dst_mesh_id];
    }

    widen_2d_route_to_chip(packet_header, routing_table->route_table_2d.data, dst_dev_id, mesh_y_size, mesh_x_size);
    return true;
}

// Multicast producer. Encodes the maps for a rectangle given as N/S/E/W extents around an anchor,
// from the reverse trees in this chip's destination-major route table.
//
// A remote final mesh takes a unicast-style carrier leg toward this mesh's exit instead, retaining
// the anchor and extents until the destination mesh's landing rebuilds the tree there.
//
// Returns this chip's own action byte. Multi-output roots are ordinary under 2D action-map routing, so a
// caller holding one connection must check the output count rather than assume it is one.
inline std::uint8_t fabric_set_2d_mcast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t e_num_hops,
    uint16_t w_num_hops,
    uint16_t n_num_hops,
    uint16_t s_num_hops,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    tt_l1_ptr routing_l1_info_t* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    ASSERT((uint32_t)mesh_y_size + mesh_x_size <= sizeof(packet_header->route_buffer));

    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    packet_header->mcast_params_64 = ((uint64_t)s_num_hops << 48) | ((uint64_t)n_num_hops << 32) |
                                     ((uint64_t)w_num_hops << 16) | ((uint64_t)e_num_hops);
    packet_header->routing_fields.value = 0;

    const uint32_t root_y = routing_table->my_mesh_coord_y;
    const uint32_t root_x = routing_table->my_mesh_coord_x;

    if (dst_mesh_id < MAX_NUM_MESHES && routing_table->my_mesh_id != dst_mesh_id) {
        // Carrier leg. The widen never reads dst_start_node_id, so the retained anchor survives and
        // the temporary exit lives only in the installed maps.
        auto exit_node_table = reinterpret_cast<tt_l1_ptr uint8_t*>(EXIT_NODE_TABLE_BASE);
        const uint16_t exit_dev_id = exit_node_table[dst_mesh_id];
        ASSERT(exit_dev_id != (uint16_t)eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
        // A worker on the exit chip itself would get maps with no eth bits and inject nowhere.
        // Leaving the mesh from here needs the INTERMESH egress connection, which this path does
        // not model.
        ASSERT(exit_dev_id != (uint16_t)((uint32_t)root_y * mesh_x_size + root_x));
        widen_2d_route_to_chip(
            packet_header, routing_table->route_table_2d.data, exit_dev_id, mesh_y_size, mesh_x_size);
        // Same fall-through as the router's decode: the Y byte wins when nonzero, else the X byte
        // carries it.
        const std::uint8_t action_y = packet_header->route_buffer[root_y];
        return action_y != 0 ? action_y : packet_header->route_buffer[mesh_y_size + root_x];
    }

    // Tree pruning repeatedly clears and ORs map bytes, so keep those read-modify-writes in local
    // staging. Once complete, commit the contiguous [Y | X] map to volatile packet L1 in chunks.
    constexpr uint32_t route_buffer_bytes = sizeof(HybridMeshPacketHeader::route_buffer);
    const uint32_t map_bytes = (uint32_t)mesh_y_size + mesh_x_size;
    route_2d_detail::Route2DMapStaging<route_buffer_bytes> maps(map_bytes);
    encode_2d_mcast_maps<route_2d_detail::AlignedMcastTreeEdgeReader>(
        maps.bytes(),
        routing_table->route_table_2d.data,
        mesh_y_size,
        mesh_x_size,
        root_y,
        root_x,
        n_num_hops,
        s_num_hops,
        e_num_hops,
        w_num_hops);

    route_2d_detail::copy_2d_map_to_l1(packet_header->route_buffer, maps.word_data(), map_bytes);

    // Returned rather than acted on: how many outputs a caller can launch depends on the connections
    // it holds. Zero outputs is legal and means deliver locally only.
    return maps.bytes()[root_y];
}

// Intermesh landing encode. Runs on the boundary-facing router before ordinary decode, replacing the
// incoming source-mesh maps with ones built from this mesh's destination-major route table:
//
//   intermediate mesh          maps toward this mesh's next exit
//   destination mesh, unicast  widen to the retained final chip
//   destination mesh, mcast    rebuild the multicast maps rooted here
//
// dst_start_node_id and mcast_params_64 are read but never written.
inline void fabric_set_2d_intermesh_landing_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    const routing_l1_info_t& routing_table,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    const uint16_t final_mesh_id = packet_header->dst_start_mesh_id;
    // Bounds before exit_node_table indexing.
    ASSERT(final_mesh_id < MAX_NUM_MESHES);
    if (final_mesh_id != routing_table.my_mesh_id) {
        // Intermediate landing: route to this mesh's next exit toward the final mesh. Multicast-safe
        // as-is, since the anchor and extents stay retained and fanout must not begin in a mesh that
        // holds none of the targets.
        const uint16_t exit_dev_id = routing_table.exit_node_table[final_mesh_id];
        ASSERT(exit_dev_id != (uint16_t)eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
        widen_2d_route_to_chip(packet_header, routing_table.route_table_2d.data, exit_dev_id, mesh_y_size, mesh_x_size);
        return;
    }

    if (packet_header->mcast_params_64 == 0) {
        // Destination landing, unicast: widen to the final chip.
        widen_2d_route_to_chip(
            packet_header,
            routing_table.route_table_2d.data,
            packet_header->dst_start_chip_id,
            mesh_y_size,
            mesh_x_size);
        return;
    }

    // Destination landing, multicast: rebuild the tree rooted at this chip, with the rectangle still
    // measured from the retained anchor in dst_start_node_id. A multi-output root needs no special
    // handling here, since a landing is already an RX fanout point.
    const uint16_t anchor_dev_id = packet_header->dst_start_chip_id;
    ASSERT(anchor_dev_id < (uint32_t)mesh_y_size * mesh_x_size);

    // Reuse the worker encoder, but root its X tree at this landing column. Local staging avoids
    // volatile updates while pruning; only the completed map is copied over the source-mesh map.
    constexpr uint32_t route_buffer_bytes = sizeof(HybridMeshPacketHeader::route_buffer);
    const uint32_t map_bytes = (uint32_t)mesh_y_size + mesh_x_size;
    route_2d_detail::Route2DMapStaging<route_buffer_bytes> maps(map_bytes);
    encode_2d_mcast_maps<route_2d_detail::AlignedMcastTreeEdgeReader>(
        maps.bytes(),
        routing_table.route_table_2d.data,
        mesh_y_size,
        mesh_x_size,
        anchor_dev_id / mesh_x_size,
        anchor_dev_id % mesh_x_size,
        routing_table.my_mesh_coord_x,
        packet_header->mcast_params[eth_chan_directions::NORTH],
        packet_header->mcast_params[eth_chan_directions::SOUTH],
        packet_header->mcast_params[eth_chan_directions::EAST],
        packet_header->mcast_params[eth_chan_directions::WEST]);

    route_2d_detail::copy_2d_map_to_l1(packet_header->route_buffer, maps.word_data(), map_bytes);
}

// Single-hop poke: the destination is exactly one fabric hop away, so this writes LOCAL_DELIVER at
// the destination's map slot on the hop's axis. Unlike the legacy helper, Z hops are allowed.
inline void fabric_set_2d_single_hop_unicast_route_from_direction(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    eth_chan_directions next_hop_direction,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    ASSERT(next_hop_direction < eth_chan_directions::COUNT);
    // Same-mesh only, per this helper's contract. An inter-mesh destination numbers its chip in the
    // *other* mesh's space, so dst_y/dst_x below would index this mesh's maps with a foreign id.
    // Asserted explicitly rather than left to the bounds check, because the bounds check passes
    // whenever the foreign id happens to be in range and then pokes the wrong slot.
    ASSERT(dst_mesh_id == reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE)->my_mesh_id);
    ASSERT(dst_dev_id < (uint32_t)mesh_y_size * mesh_x_size);
    ASSERT((uint32_t)mesh_y_size + mesh_x_size <= sizeof(packet_header->route_buffer));

    const uint32_t dst_y = dst_dev_id / mesh_x_size;
    const uint32_t dst_x = dst_dev_id % mesh_x_size;

    // The receiving router's decode axis matches the hop axis: N/S/Z hops read the Y byte, E/W hops
    // the X byte. Other map slots are intentionally left untouched under the one-hop contract.
    switch (next_hop_direction) {
        case eth_chan_directions::NORTH:
        case eth_chan_directions::SOUTH:
        case eth_chan_directions::Z: packet_header->route_buffer[dst_y] = Routing2DCodec::ACTION_LOCAL_DELIVER; break;
        case eth_chan_directions::EAST:
        case eth_chan_directions::WEST:
            packet_header->route_buffer[mesh_y_size + dst_x] = Routing2DCodec::ACTION_LOCAL_DELIVER;
            break;
        default: ASSERT(false); break;
    }

    packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    packet_header->mcast_params_64 = 0;
    packet_header->routing_fields.value = 0;
}

// NOTE: there is deliberately no private `fabric_set_2d_single_hop_unicast_route` wrapper here. The
// public `fabric_set_single_hop_unicast_route` above already resolves the direction with
// get_next_hop_router_direction(), then delegates through the public from-direction helper to the 2D
// action-map encoder. A second wrapper doing the same two steps would be dead code.

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
