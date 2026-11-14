// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Horizon (Quasar 2xx) mesh API implementation.
// Uses Horizon crossbar routing via fabric_flat_address_routing_l1_info_t::flat_addr + noc_async_write.
// All functions except fabric_unicast_noc_unicast_write are TODO stubs.

#pragma once

#include <cstdint>
#include <type_traits>
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "tt_metal/fabric/hw/inc/mesh/addrgen_api.h"

using namespace tt::tt_fabric::common::experimental;
namespace tt::tt_fabric::mesh::experimental {

// ATT endpoint parameters for Horizon x2 (values from horizon_2x3_att_config.h).
// Defined here as constexpr to avoid pulling in the config header into kernel code.
static constexpr uint32_t HORIZON_ATT_WORKER_GRID_X = 2;
static constexpr uint32_t HORIZON_ATT_ENDPOINT_SHIFT = 26;
static constexpr uint64_t HORIZON_ATT_OFFSET_MASK = (uint64_t(1) << HORIZON_ATT_ENDPOINT_SHIFT) - 1;

// FABRIC_MAX_PACKET_SIZE is available via macro from addrgen_api.h

template <typename FabricSenderType, bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    [[maybe_unused]] auto* _ci = client_interface;
    [[maybe_unused]] auto* _ph = packet_header;

    const uint64_t noc_addr = noc_unicast_command_header.noc_address;
    // noc_addr is a flat ATT address: REMOTE_WINDOW.compare | (selector << HORIZON_ATT_ENDPOINT_SHIFT) | offset
    const uint64_t local_offset = noc_addr & HORIZON_ATT_OFFSET_MASK;
    const uint32_t selector = static_cast<uint32_t>((noc_addr >> HORIZON_ATT_ENDPOINT_SHIFT) & 0x3F);
    const uint32_t tile_x = selector % HORIZON_ATT_WORKER_GRID_X;
    const uint32_t tile_y = selector / HORIZON_ATT_WORKER_GRID_X;

    ASSERT(dst_mesh_id < tt::tt_fabric::MAX_HORIZON_MESHES);
    ASSERT(dst_dev_id < tt::tt_fabric::MAX_HORIZON_DEVS_PER_MESH);
    ASSERT(tile_x < HORIZON_ATT_WORKER_GRID_X);
    ASSERT(tile_y < tt::tt_fabric::MAX_HORIZON_NOC_Y);
    const auto* routing_table =
        reinterpret_cast<const tt_l1_ptr tt::tt_fabric::fabric_flat_address_routing_l1_info_t*>(ROUTING_TABLE_BASE);
    // flat_addr[mesh][dev_id][noc_x_within_dev][noc_y_within_dev]
    const uint64_t remote_addr = routing_table->flat_addr[dst_mesh_id][dst_dev_id][tile_x][tile_y] + local_offset;

    // Horizon NoC communication can currently only reach the lower 1 MB of each
    // Tensix core's L1; the NOC2AXI crossbar window is 1 MB per tile slot.
    ASSERT(local_offset + size <= (1u << 20));
    // Zero base means an unfilled routing table slot.
    ASSERT(remote_addr - local_offset != 0);
    noc_async_write(src_addr, remote_addr, size);
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <typename FabricSenderType, bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType, bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType, bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <typename FabricSenderType, bool SetRoute>
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

#if defined(FABRIC_2D)
namespace mcast_detail {

struct PreparedMulticastBranchFanout {
    volatile PACKET_HEADER_TYPE* packet_header;
    uint8_t root_outputs;
};

// One call represents one branch of the existing client decomposition: E-only, W-only, N with
// optional E/W teeth, or S with optional E/W teeth. Z is never specified by the client; an express
// reverse tree may select it in addition to, or instead of, the branch's cardinal root output.
FORCE_INLINE bool is_valid_multicast_branch(const MeshMcastRange& branch) {
    const bool has_vertical_trunk = branch.n != 0 || branch.s != 0;
    if (has_vertical_trunk) {
        return (branch.n != 0) != (branch.s != 0);
    }
    return (branch.e != 0) != (branch.w != 0);
}

FORCE_INLINE uint8_t multicast_branch_allowed_root_outputs(const MeshMcastRange& branch) {
    if (branch.n != 0) {
        return Routing2DCodec::ACTION_NORTH | Routing2DCodec::ACTION_Z;
    }
    if (branch.s != 0) {
        return Routing2DCodec::ACTION_SOUTH | Routing2DCodec::ACTION_Z;
    }
    return branch.e != 0 ? Routing2DCodec::ACTION_EAST : Routing2DCodec::ACTION_WEST;
}

// Acquires one shared header and encodes one same-mesh multicast branch, anchored at the manager's
// source chip. Inter-mesh multicast retains its explicit anchor in the lower-level API.
FORCE_INLINE PreparedMulticastBranchFanout prepare_multicast_branch_fanout(
    uint8_t route_id,
    const tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    const MeshMcastRange& branch) {
    ASSERT(is_valid_multicast_branch(branch));

    auto [packet_header, num_headers] = PacketHeaderPool::header_table[route_id];
    ASSERT(num_headers == 1);

    const uint8_t root_action = fabric_set_2d_mcast_route(
        packet_header,
        connection_manager.my_chip_id,
        connection_manager.my_mesh_id,
        branch.e,
        branch.w,
        branch.n,
        branch.s);
    const uint8_t root_outputs = root_action & Routing2DCodec::ACTION_ETH_MASK;
    ASSERT(root_outputs != 0);
    const uint8_t allowed_root_outputs = multicast_branch_allowed_root_outputs(branch);
    ASSERT((root_outputs | allowed_root_outputs) == allowed_root_outputs);
    return {packet_header, root_outputs};
}

template <bool SendPayload>
FORCE_INLINE void submit_multicast_branch_fanout_impl(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    const PreparedMulticastBranchFanout& fanout,
    [[maybe_unused]] uint32_t src_addr,
    [[maybe_unused]] uint32_t size) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void submit_multicast_branch_fanout_with_payload(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    const PreparedMulticastBranchFanout& fanout,
    uint32_t src_addr,
    uint32_t size) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void submit_multicast_branch_fanout_header_only(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, const PreparedMulticastBranchFanout& fanout) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

}  // namespace mcast_detail

FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}
#endif

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType, bool SetRoute>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <typename FabricSenderType, bool SetRoute>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

// ============================================================================
// Addrgen API Overloads
// ============================================================================

// Unicast Write Addrgen Overloads
template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

// Fused Unicast Write + Atomic Inc Addrgen Overloads
template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset,
    bool flush) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset,
    bool flush) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset,
    bool flush) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0,
    uint32_t offset1) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0,
    uint32_t offset1) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0,
    uint32_t offset1) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename AddrGenType, typename>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0,
    uint32_t offset1) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename AddrGenType, typename>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0,
    uint32_t offset1) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0,
    uint32_t offset1) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset,
    bool flush) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename AddrGenType, typename>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset,
    bool flush) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset,
    bool flush) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

}  // namespace tt::tt_fabric::mesh::experimental
