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

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops) {}

FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* num_hops) {}

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
