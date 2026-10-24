// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

using namespace tt::tt_fabric::common::experimental;
namespace tt::tt_fabric::linear::experimental {

// static FORCE_INLINE void fabric_set_unicast_route(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t i) {}

// static FORCE_INLINE void fabric_set_mcast_route(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t* range,
//     uint8_t i) {}

// --- Unicast write ---
//
// Horizon crossbar path: no ERISC hop-by-hop routing.
// The destination NOC (x, y) and local L1 offset are extracted from
// noc_unicast_command_header.noc_address (standard NOC address format).
// The flat address is looked up from horizon_routing_l1_info_t at ROUTING_TABLE_BASE
// and noc_async_write is issued directly.
//
// client_interface and packet_header are unused on this path.

// Overload 1: explicit dst_dev_id + dst_mesh_id (mirrors the Mesh API).
template <typename FabricSenderType>
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
    const uint32_t noc_x = NOC_UNICAST_ADDR_X(noc_addr);
    const uint32_t noc_y = NOC_UNICAST_ADDR_Y(noc_addr);
    const uint64_t local_offset = noc_addr & NOC_LOCAL_ADDR_MASK;

    const auto* routing_table =
        reinterpret_cast<const tt_l1_ptr tt::tt_fabric::horizon_routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint64_t remote_addr = routing_table->flat_addr[dst_mesh_id][dst_dev_id][noc_x][noc_y] + local_offset;

    noc_async_write(src_addr, remote_addr, size);
}

// Route-manager variant: dst_dev_id / dst_mesh_id come from connection_manager slots.
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write(
            &slot.sender, packet_header, slot.dst_dev_id, slot.dst_mesh_id, src_addr, size, noc_unicast_command_header);
    });
}

// Overload 2: num_hops-based (linear API compatibility). Delegates to overload 1.
// TODO: num_hops is currently forwarded as dst_dev_id with dst_mesh_id=0; replace
//       with proper hop-to-device conversion once direction encoding is defined.
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops) {
    fabric_unicast_noc_unicast_write(
        client_interface, packet_header, num_hops, 0, src_addr, size, noc_unicast_command_header);
}

FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, num_hops[i]);
    });
}

// template <
//     UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// --- Unicast atomic inc ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
//     uint8_t num_hops) {}

// FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
//     uint8_t* num_hops) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* num_hops,
//     CommandHeaderT command_header) {}

// // --- Unicast scatter write ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_unicast_noc_scatter_write(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
//     uint8_t num_hops) {}

// FORCE_INLINE void fabric_unicast_noc_scatter_write(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
//     uint8_t* num_hops) {}

// template <
//     UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <
//     UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// // --- Unicast fused scatter write + atomic inc ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
//     uint8_t num_hops) {}

// FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
//     uint8_t* num_hops) {}

// template <
//     UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <
//     UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// // --- Unicast inline write ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
//     uint8_t num_hops) {}

// FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
//     uint8_t* num_hops) {}

// template <
//     UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {}

// template <
//     UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {}

// template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header) {}

// template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* num_hops,
//     CommandHeaderT command_header) {}

// // --- Unicast fused unicast write + atomic inc ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
//     uint8_t num_hops) {}

// FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
//     uint8_t* num_hops) {}

// template <
//     UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <
//     UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* num_hops,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// #if 0  // multicast not yet implemented for Horizon

// --- Multicast unicast write ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_multicast_noc_unicast_write(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
//     uint8_t start_distance,
//     uint8_t range) {}

// FORCE_INLINE void fabric_multicast_noc_unicast_write(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
//     uint8_t* start_distance,
//     uint8_t* range) {}

// template <
//     UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t start_distance,
//     uint8_t range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* start_distance,
//     uint8_t* range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// // --- Multicast atomic inc ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
//     uint8_t start_distance,
//     uint8_t range) {}

// FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
//     uint8_t* start_distance,
//     uint8_t* range) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t start_distance,
//     uint8_t range,
//     CommandHeaderT command_header = nullptr) {}

// template <
//     UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* start_distance,
//     uint8_t* range,
//     CommandHeaderT command_header = nullptr) {}

// // --- Multicast scatter write ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_multicast_noc_scatter_write(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
//     uint8_t start_distance,
//     uint8_t range) {}

// FORCE_INLINE void fabric_multicast_noc_scatter_write(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
//     uint8_t* start_distance,
//     uint8_t* range) {}

// template <
//     UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <
//     UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t start_distance,
//     uint8_t range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* start_distance,
//     uint8_t* range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// // --- Multicast inline write ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
//     uint8_t start_distance,
//     uint8_t range) {}

// FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
//     uint8_t* start_distance,
//     uint8_t* range) {}

// template <
//     UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {}

// template <
//     UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {}

// template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t start_distance,
//     uint8_t range,
//     CommandHeaderT command_header) {}

// template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* start_distance,
//     uint8_t* range,
//     CommandHeaderT command_header) {}

// // --- Multicast fused unicast write + atomic inc ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
//     uint8_t start_distance,
//     uint8_t range) {}

// FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
//     uint8_t* start_distance,
//     uint8_t* range) {}

// template <
//     UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <
//     UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t start_distance,
//     uint8_t range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* start_distance,
//     uint8_t* range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// // --- Multicast fused scatter write + atomic inc ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
//     uint8_t start_distance,
//     uint8_t range) {}

// FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
//     uint8_t* start_distance,
//     uint8_t* range) {}

// template <
//     UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
//     typename FabricSenderType,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <
//     UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
//     typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint8_t start_distance,
//     uint8_t range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
// FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint8_t* start_distance,
//     uint8_t* range,
//     CommandHeaderT command_header = nullptr,
//     uint16_t packet_size_bytes = 0) {}

// // --- Sparse multicast ---

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
//     uint16_t hops) {}

// FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write(
//     tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
//     uint8_t route_id,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
//     uint16_t* hops) {}

// template <typename FabricSenderType>
// FORCE_INLINE void fabric_sparse_multicast_noc_scatter_write(
//     tt_l1_ptr FabricSenderType* client_interface,
//     volatile PACKET_HEADER_TYPE* packet_header,
//     uint32_t src_addr,
//     uint32_t size,
//     tt::tt_fabric::NocSparseMulticastWriteCommandHeader sparse_mcast_command_header,
//     uint16_t hops) {}

#endif  // multicast not yet implemented for Horizon

}  // namespace tt::tt_fabric::linear::experimental
