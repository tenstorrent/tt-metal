// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Note: This file was moved from tt_metal/fabric/hw/inc/mesh/api.h. For full commit history, run `git log --follow api_impl.h`.

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

    if constexpr (SetRoute) {
        fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    }
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write<decltype(slot.sender), SetRoute>(
            &slot.sender, packet_header, slot.dst_dev_id, slot.dst_mesh_id, src_addr, size, noc_unicast_command_header);
    });
}

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, noc_unicast_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_command_header, packet_size_bytes);
    });
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->to_noc_unicast_atomic_inc(noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_atomic_inc(
            &slot.sender, packet_header, slot.dst_dev_id, slot.dst_mesh_id, noc_unicast_atomic_inc_command_header);
    });
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_atomic_inc_command_header);
    });
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
    packet_header->payload_size_bytes = 0;
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header);
    });
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
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    if constexpr (SetRoute) {
        fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    }
    packet_header->to_noc_unicast_scatter_write(noc_unicast_scatter_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_scatter_write<decltype(slot.sender), SetRoute>(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            src_addr,
            size,
            noc_unicast_scatter_command_header);
    });
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
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    if constexpr (SetRoute) {
        fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    }
    packet_header->to_noc_fused_unicast_scatter_write_atomic_inc(
        noc_unicast_scatter_atomic_inc_fused_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_scatter_write_atomic_inc<decltype(slot.sender), SetRoute>(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            src_addr,
            size,
            noc_unicast_scatter_atomic_inc_fused_command_header);
    });
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_fused_scatter_write_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_unicast_scatter_atomic_inc_fused_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state<UpdateMask>(
            &slot.sender,
            packet_header,
            src_addr,
            noc_unicast_scatter_atomic_inc_fused_command_header,
            packet_size_bytes);
    });
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    static_assert(
        has_flag(UpdateMask, UnicastFusedScatterWriteAtomicIncUpdateMask::Flush),
        "When setting state, the Flush field must be updated to set the chunk encoding for the semaphore increment "
        "chunk");
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_fused_scatter_write_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header, packet_size_bytes);
    });
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_scatter_write_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_unicast_scatter_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_scatter_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_scatter_command_header, packet_size_bytes);
    });
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_scatter_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->to_noc_unicast_inline_write(noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_inline_write(
            &slot.sender, packet_header, slot.dst_dev_id, slot.dst_mesh_id, noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_inline_fields<UpdateMask>(packet_header, noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_inline_write_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
    packet_header->payload_size_bytes = 0;
    populate_unicast_inline_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_inline_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header);
    });
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
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    if constexpr (SetRoute) {
        fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    }
    packet_header->to_noc_fused_unicast_write_atomic_inc(noc_fused_unicast_atomic_inc_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc<decltype(slot.sender), SetRoute>(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            src_addr,
            size,
            noc_fused_unicast_atomic_inc_command_header);
    });
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header, packet_size_bytes);
    });
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header, packet_size_bytes);
    });
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
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    if constexpr (SetRoute) {
        fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    }
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_write(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            ranges[i],
            src_addr,
            size,
            noc_unicast_command_header);
    });
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

// A branch leaves on its own cardinal and, when that cardinal's axis carries the express chords, may
// also (or instead) launch a chord from the anchor.
FORCE_INLINE uint8_t multicast_branch_allowed_root_outputs(const MeshMcastRange& branch) {
    uint8_t cardinal = Routing2DCodec::ACTION_WEST;
    Fabric2DAxis axis = Fabric2DAxis::X;
    if (branch.n != 0) {
        cardinal = Routing2DCodec::ACTION_NORTH;
        axis = Fabric2DAxis::Y;
    } else if (branch.s != 0) {
        cardinal = Routing2DCodec::ACTION_SOUTH;
        axis = Fabric2DAxis::Y;
    } else if (branch.e != 0) {
        cardinal = Routing2DCodec::ACTION_EAST;
    }
    return cardinal | (Fabric2DRoutingConfig::axis_actions(axis) & Routing2DCodec::ACTION_Z);
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
    [[maybe_unused]] uint32_t src_addr = 0,
    [[maybe_unused]] uint32_t size = 0) {
    uint8_t pending_outputs = fanout.root_outputs;
    connection_manager.for_each([&](auto& sender, uint32_t, uint32_t tag) {
        if (tag >= static_cast<uint32_t>(eth_chan_directions::COUNT)) {
            return;
        }
        const uint8_t output_bit = Routing2DCodec::action_bit(static_cast<eth_chan_directions>(tag));
        if ((pending_outputs & output_bit) == 0) {
            return;
        }
        pending_outputs &= static_cast<uint8_t>(~output_bit);
        sender.wait_for_empty_write_slot();
        if constexpr (SendPayload) {
            sender.send_payload_without_header_non_blocking_from_address(src_addr, size);
        }
        sender.send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(fanout.packet_header), sizeof(PACKET_HEADER_TYPE));
    });
    ASSERT(pending_outputs == 0);
}

FORCE_INLINE void submit_multicast_branch_fanout_with_payload(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    const PreparedMulticastBranchFanout& fanout,
    uint32_t src_addr,
    uint32_t size) {
    submit_multicast_branch_fanout_impl<true>(connection_manager, fanout, src_addr, size);
}

FORCE_INLINE void submit_multicast_branch_fanout_header_only(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, const PreparedMulticastBranchFanout& fanout) {
    submit_multicast_branch_fanout_impl<false>(connection_manager, fanout);
}

}  // namespace mcast_detail

FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    const auto fanout = mcast_detail::prepare_multicast_branch_fanout(route_id, connection_manager, branch);
    fanout.packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    mcast_detail::submit_multicast_branch_fanout_with_payload(connection_manager, fanout, src_addr, size);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    const auto fanout = mcast_detail::prepare_multicast_branch_fanout(route_id, connection_manager, branch);
    fanout.packet_header->to_noc_unicast_atomic_inc(noc_unicast_atomic_inc_command_header);
    mcast_detail::submit_multicast_branch_fanout_header_only(connection_manager, fanout);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    const auto fanout = mcast_detail::prepare_multicast_branch_fanout(route_id, connection_manager, branch);
    fanout.packet_header->to_noc_fused_unicast_write_atomic_inc(noc_fused_unicast_atomic_inc_command_header, size);
    mcast_detail::submit_multicast_branch_fanout_with_payload(connection_manager, fanout, src_addr, size);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    const auto fanout = mcast_detail::prepare_multicast_branch_fanout(route_id, connection_manager, branch);
    fanout.packet_header->to_noc_fused_unicast_scatter_write_atomic_inc(
        noc_unicast_scatter_atomic_inc_fused_command_header, size);
    mcast_detail::submit_multicast_branch_fanout_with_payload(connection_manager, fanout, src_addr, size);
}
#endif

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, noc_unicast_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_command_header, packet_size_bytes);
    });
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->to_noc_unicast_atomic_inc(noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_atomic_inc(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            ranges[i],
            noc_unicast_atomic_inc_command_header);
    });
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_atomic_inc_command_header);
    });
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header) {
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
    packet_header->payload_size_bytes = 0;
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header);
    });
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
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->to_noc_unicast_scatter_write(noc_unicast_scatter_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_scatter_write(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            ranges[i],
            src_addr,
            size,
            noc_unicast_scatter_command_header);
    });
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_scatter_write_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_unicast_scatter_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_scatter_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_scatter_command_header, packet_size_bytes);
    });
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_scatter_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->to_noc_unicast_inline_write(noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_inline_write(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            ranges[i],
            noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_inline_fields<UpdateMask>(packet_header, noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_inline_write_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_inline_write_command_header);
    });
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header) {
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
    packet_header->payload_size_bytes = 0;
    populate_unicast_inline_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_inline_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header);
    });
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
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    if constexpr (SetRoute) {
        fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    }
    packet_header->to_noc_fused_unicast_write_atomic_inc(noc_fused_unicast_atomic_inc_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            ranges[i],
            src_addr,
            size,
            noc_fused_unicast_atomic_inc_command_header);
    });
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
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    if constexpr (SetRoute) {
        fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    }
    packet_header->to_noc_fused_unicast_scatter_write_atomic_inc(
        noc_unicast_scatter_atomic_inc_fused_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            ranges[i],
            src_addr,
            size,
            noc_unicast_scatter_atomic_inc_fused_command_header);
    });
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_fused_scatter_write_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_unicast_scatter_atomic_inc_fused_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state<UpdateMask>(
            &slot.sender,
            packet_header,
            src_addr,
            noc_unicast_scatter_atomic_inc_fused_command_header,
            packet_size_bytes);
    });
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    static_assert(
        has_flag(UpdateMask, UnicastFusedScatterWriteAtomicIncUpdateMask::Flush),
        "When setting state, the Flush field must be updated to set the chunk encoding for the semaphore increment "
        "chunk");
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_fused_scatter_write_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header, packet_size_bytes);
    });
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header, packet_size_bytes);
    });
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header, packet_size_bytes);
    });
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);

    // Set route once before sending packets
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);

    uint32_t remaining_size = page_size;
    uint32_t current_offset = offset;

    // Send full-size packets (loop skips for small pages)
    while (remaining_size > FABRIC_MAX_PACKET_SIZE) {
        auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Call with SetRoute=false since we already set it
        fabric_unicast_noc_unicast_write<FabricSenderType, false>(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            src_addr,
            FABRIC_MAX_PACKET_SIZE,
            tt::tt_fabric::NocUnicastCommandHeader{noc_address});

        src_addr += FABRIC_MAX_PACKET_SIZE;
        current_offset += FABRIC_MAX_PACKET_SIZE;
        remaining_size -= FABRIC_MAX_PACKET_SIZE;
    }

    // Send remainder packet (for small pages, this is the only packet)
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Call with SetRoute=false since we already set it (no barrier needed after last packet)
    fabric_unicast_noc_unicast_write<FabricSenderType, false>(
        client_interface,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        src_addr,
        remaining_size,
        tt::tt_fabric::NocUnicastCommandHeader{noc_address});
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);

    uint32_t remaining_size = page_size;
    uint32_t current_offset = offset;

    // Send full-size packets (loop skips for small pages)
    while (remaining_size > FABRIC_MAX_PACKET_SIZE) {
        auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Call basic _with_state function
        fabric_unicast_noc_unicast_write_with_state<UpdateMask>(
            client_interface,
            packet_header,
            src_addr,
            tt::tt_fabric::NocUnicastCommandHeader{noc_address},
            FABRIC_MAX_PACKET_SIZE);

        src_addr += FABRIC_MAX_PACKET_SIZE;
        current_offset += FABRIC_MAX_PACKET_SIZE;
        remaining_size -= FABRIC_MAX_PACKET_SIZE;
    }

    // Send remainder packet (for small pages, this is the only packet)
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Call basic _with_state function (no barrier needed after last packet)
    fabric_unicast_noc_unicast_write_with_state<UpdateMask>(
        client_interface, packet_header, src_addr, tt::tt_fabric::NocUnicastCommandHeader{noc_address}, remaining_size);
}

template <UnicastWriteUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Cap initial payload size to hardware limit for large pages
    // The WithState calls in the loop will handle actual chunking
    uint32_t init_payload_size = (page_size > FABRIC_MAX_PACKET_SIZE) ? FABRIC_MAX_PACKET_SIZE : page_size;

    fabric_unicast_noc_unicast_write_set_state<UpdateMask>(
        packet_header, dst_dev_id, dst_mesh_id, tt::tt_fabric::NocUnicastCommandHeader{noc_address}, init_payload_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Set route once
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);

    uint32_t remaining = page_size;
    uint32_t current_src_addr = src_addr;
    uint64_t current_noc_addr = noc_address;

    // Send intermediate chunks as regular writes (loop skips for small pages)
    while (remaining > FABRIC_MAX_PACKET_SIZE) {
        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Call basic unicast write with SetRoute=false
        fabric_unicast_noc_unicast_write<FabricSenderType, false>(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            current_src_addr,
            FABRIC_MAX_PACKET_SIZE,
            tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr});

        current_src_addr += FABRIC_MAX_PACKET_SIZE;
        current_noc_addr += FABRIC_MAX_PACKET_SIZE;
        remaining -= FABRIC_MAX_PACKET_SIZE;
    }

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Final chunk: fused write+atomic inc (for small pages, this is the only packet)
    fabric_unicast_noc_fused_unicast_with_atomic_inc<FabricSenderType, false>(
        client_interface,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        current_src_addr,
        remaining,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{current_noc_addr, semaphore_noc_address, val, flush});
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    uint32_t remaining_size = page_size;
    uint32_t current_offset = offset;
    uint32_t packet_src_addr = src_addr;

    // Send intermediate chunks as regular writes (loop skips for small pages)
    while (remaining_size > FABRIC_MAX_PACKET_SIZE) {
        auto chunk_noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Set noc_send_type to match the command fields
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;

        // Call basic unicast write _with_state for intermediate packets
        fabric_unicast_noc_unicast_write_with_state<
            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
            client_interface,
            packet_header,
            packet_src_addr,
            tt::tt_fabric::NocUnicastCommandHeader{chunk_noc_address},
            FABRIC_MAX_PACKET_SIZE);

        packet_src_addr += FABRIC_MAX_PACKET_SIZE;
        current_offset += FABRIC_MAX_PACKET_SIZE;
        remaining_size -= FABRIC_MAX_PACKET_SIZE;
    }

    // Final chunk: fused write+atomic inc (for small pages, this is the only packet)
    auto final_noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Set noc_send_type back to fused for final chunk
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;

    // Call basic fused atomic inc _with_state for final packet
    fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask>(
        client_interface,
        packet_header,
        packet_src_addr,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{final_noc_address, semaphore_noc_address, val, flush},
        remaining_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Cap initial payload size to hardware limit for large pages
    // The WithState calls in the loop will handle actual chunking
    uint32_t init_payload_size = (page_size > FABRIC_MAX_PACKET_SIZE) ? FABRIC_MAX_PACKET_SIZE : page_size;

    // Call base _set_state to set up all header fields
    // This is typically called once before a loop
    fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{noc_address, semaphore_noc_address, val, flush},
        init_payload_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);

    // Set route once before sending packets
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);

    auto noc_address0 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id0, offset0);
    auto noc_address1 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id1, offset1);

    if (page_size * 2 <= FABRIC_MAX_PACKET_SIZE) {
        // Small pages: use scatter operation
        fabric_unicast_noc_scatter_write<FabricSenderType, false>(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            src_addr,
            page_size * 2,
            tt::tt_fabric::NocUnicastScatterCommandHeader(
                {noc_address0, noc_address1}, {static_cast<uint16_t>(page_size)}));
    } else {
        // Large pages: fall back to separate unicast writes
        // The fabric_unicast_noc_unicast_write calls will handle auto-packetization
        fabric_unicast_noc_unicast_write<FabricSenderType>(
            client_interface, packet_header, dst_dev_id, dst_mesh_id, src_addr, addrgen, page_id0, offset0);

        fabric_unicast_noc_unicast_write<FabricSenderType>(
            client_interface, packet_header, dst_dev_id, dst_mesh_id, src_addr + page_size, addrgen, page_id1, offset1);
    }
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address0 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id0, offset0);
    auto noc_address1 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id1, offset1);

    if (page_size * 2 <= FABRIC_MAX_PACKET_SIZE) {
        // Small pages: use scatter operation
        fabric_unicast_noc_scatter_write_with_state<UpdateMask>(
            client_interface,
            packet_header,
            src_addr,
            tt::tt_fabric::NocUnicastScatterCommandHeader(
                {noc_address0, noc_address1}, {static_cast<uint16_t>(page_size)}),
            page_size * 2);
    } else {
        // Large pages: fall back to separate unicast writes
        // The fabric_unicast_noc_unicast_write calls will handle auto-packetization
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;

        // The fabric_unicast_noc_unicast_write_with_state calls will handle auto-packetization
        fabric_unicast_noc_unicast_write_with_state(
            client_interface, packet_header, dst_dev_id, dst_mesh_id, src_addr, addrgen, page_id0, offset0);

        // Ensure first call completes before starting second
        noc_async_writes_flushed();

        fabric_unicast_noc_unicast_write_with_state(
            client_interface, packet_header, dst_dev_id, dst_mesh_id, src_addr + page_size, addrgen, page_id1, offset1);
    }
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);

    // Cap payload size to prevent invalid header initialization for large pages
    uint32_t capped_payload_size = (page_size * 2 > FABRIC_MAX_PACKET_SIZE) ? FABRIC_MAX_PACKET_SIZE : page_size * 2;

    auto noc_address0 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id0, offset0);
    auto noc_address1 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id1, offset1);

    fabric_unicast_noc_scatter_write_set_state<UpdateMask>(
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        tt::tt_fabric::NocUnicastScatterCommandHeader({noc_address0, noc_address1}, {static_cast<uint16_t>(page_size)}),
        capped_payload_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);

    // Set route once before sending packets
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);

    uint32_t remaining_size = page_size;
    uint32_t current_offset = offset;

    // Send full-size packets (loop skips for small pages)
    while (remaining_size > FABRIC_MAX_PACKET_SIZE) {
        auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Call basic function with SetRoute=false since we already set it
        fabric_multicast_noc_unicast_write<FabricSenderType, false>(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            ranges,
            src_addr,
            FABRIC_MAX_PACKET_SIZE,
            tt::tt_fabric::NocUnicastCommandHeader{noc_address});

        src_addr += FABRIC_MAX_PACKET_SIZE;
        current_offset += FABRIC_MAX_PACKET_SIZE;
        remaining_size -= FABRIC_MAX_PACKET_SIZE;
    }

    // Send remainder packet (for small pages, this is the only packet)
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Call basic function with SetRoute=false since we already set it (no barrier needed after last packet)
    fabric_multicast_noc_unicast_write<FabricSenderType, false>(
        client_interface,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        ranges,
        src_addr,
        remaining_size,
        tt::tt_fabric::NocUnicastCommandHeader{noc_address});
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);

    uint32_t remaining_size = page_size;
    uint32_t current_offset = offset;

    // Send full-size packets (loop skips for small pages)
    while (remaining_size > FABRIC_MAX_PACKET_SIZE) {
        auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Call basic _with_state function
        fabric_multicast_noc_unicast_write_with_state<UpdateMask>(
            client_interface,
            packet_header,
            src_addr,
            tt::tt_fabric::NocUnicastCommandHeader{noc_address},
            FABRIC_MAX_PACKET_SIZE);

        src_addr += FABRIC_MAX_PACKET_SIZE;
        current_offset += FABRIC_MAX_PACKET_SIZE;
        remaining_size -= FABRIC_MAX_PACKET_SIZE;
    }

    // Send remainder packet (for small pages, this is the only packet)
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Call basic _with_state function (no barrier needed after last packet)
    fabric_multicast_noc_unicast_write_with_state<UpdateMask>(
        client_interface, packet_header, src_addr, tt::tt_fabric::NocUnicastCommandHeader{noc_address}, remaining_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Cap initial payload size to hardware limit for large pages
    // The WithState calls in the loop will handle actual chunking
    uint32_t init_payload_size = (page_size > FABRIC_MAX_PACKET_SIZE) ? FABRIC_MAX_PACKET_SIZE : page_size;

    fabric_multicast_noc_unicast_write_set_state<UpdateMask>(
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        ranges,
        tt::tt_fabric::NocUnicastCommandHeader{noc_address},
        init_payload_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address0 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id0, offset0);
    auto noc_address1 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id1, offset1);

    // Set route once before sending packets
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);

    if (page_size * 2 <= FABRIC_MAX_PACKET_SIZE) {
        // Small pages: use single scatter operation
        fabric_multicast_noc_scatter_write<FabricSenderType>(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            ranges,
            src_addr,
            page_size * 2,
            tt::tt_fabric::NocUnicastScatterCommandHeader(
                {noc_address0, noc_address1}, {static_cast<uint16_t>(page_size)}));
    } else {
        // Large pages: fall back to separate multicast unicast writes
        // The fabric_multicast_noc_unicast_write calls will handle auto-packetization
        fabric_multicast_noc_unicast_write(
            client_interface, packet_header, dst_dev_id, dst_mesh_id, ranges, src_addr, addrgen, page_id0, offset0);

        // Ensure first call completes before starting second
        noc_async_writes_flushed();

        fabric_multicast_noc_unicast_write(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            ranges,
            src_addr + page_size,  // Offset for second page in CB
            addrgen,
            page_id1,
            offset1);
    }
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address0 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id0, offset0);
    auto noc_address1 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id1, offset1);

    if (page_size * 2 <= FABRIC_MAX_PACKET_SIZE) {
        // Small pages: use scatter operation
        fabric_multicast_noc_scatter_write_with_state<UpdateMask>(
            client_interface,
            packet_header,
            src_addr,
            tt::tt_fabric::NocUnicastScatterCommandHeader(
                {noc_address0, noc_address1}, {static_cast<uint16_t>(page_size)}),
            page_size * 2);
    } else {
        // Large pages: fall back to separate multicast unicast writes
        // Fix header state: kernel initialized noc_send_type as SCATTER_WRITE, but we need UNICAST_WRITE
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;

        // Send page0
        fabric_multicast_noc_unicast_write_with_state(
            client_interface, packet_header, dst_dev_id, dst_mesh_id, ranges, src_addr, addrgen, page_id0, offset0);

        // Ensure first call completes before starting second
        noc_async_writes_flushed();

        // Send page1
        fabric_multicast_noc_unicast_write_with_state(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            ranges,
            src_addr + page_size,
            addrgen,
            page_id1,
            offset1);
    }
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);

    // Cap payload size to prevent invalid header initialization for large pages
    uint32_t capped_payload_size = (page_size * 2 > FABRIC_MAX_PACKET_SIZE) ? FABRIC_MAX_PACKET_SIZE : page_size * 2;

    auto noc_address0 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id0, offset0);
    auto noc_address1 = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id1, offset1);

    fabric_multicast_noc_scatter_write_set_state<UpdateMask>(
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        ranges,
        tt::tt_fabric::NocUnicastScatterCommandHeader({noc_address0, noc_address1}, {static_cast<uint16_t>(page_size)}),
        capped_payload_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Set route once
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);

    uint32_t remaining = page_size;
    uint32_t current_src_addr = src_addr;
    uint64_t current_noc_addr = noc_address;

    // Send intermediate chunks as regular multicast writes (loop skips for small pages)
    while (remaining > FABRIC_MAX_PACKET_SIZE) {
        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Call basic multicast unicast write with SetRoute=false
        fabric_multicast_noc_unicast_write<FabricSenderType, false>(
            client_interface,
            packet_header,
            dst_dev_id,
            dst_mesh_id,
            ranges,
            current_src_addr,
            FABRIC_MAX_PACKET_SIZE,
            tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr});

        current_src_addr += FABRIC_MAX_PACKET_SIZE;
        current_noc_addr += FABRIC_MAX_PACKET_SIZE;
        remaining -= FABRIC_MAX_PACKET_SIZE;
    }

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Final chunk: fused write+atomic inc (for small pages, this is the only packet)
    fabric_multicast_noc_fused_unicast_with_atomic_inc<FabricSenderType, false>(
        client_interface,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        ranges,
        current_src_addr,
        remaining,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{current_noc_addr, semaphore_noc_address, val, flush});
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    uint32_t remaining_size = page_size;
    uint32_t current_offset = offset;
    uint32_t packet_src_addr = src_addr;

    // Send intermediate chunks as regular multicast writes (loop skips for small pages)
    while (remaining_size > FABRIC_MAX_PACKET_SIZE) {
        auto chunk_noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

        // Ensure hardware has finished reading packet_header before modifying it
        noc_async_writes_flushed();

        // Set noc_send_type to match the command fields
        packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;

        // Call basic multicast unicast write _with_state for intermediate packets
        fabric_multicast_noc_unicast_write_with_state<
            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
            client_interface,
            packet_header,
            packet_src_addr,
            tt::tt_fabric::NocUnicastCommandHeader{chunk_noc_address},
            FABRIC_MAX_PACKET_SIZE);

        packet_src_addr += FABRIC_MAX_PACKET_SIZE;
        current_offset += FABRIC_MAX_PACKET_SIZE;
        remaining_size -= FABRIC_MAX_PACKET_SIZE;
    }

    // Final chunk: fused write+atomic inc (for small pages, this is the only packet)
    auto final_noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);

    // Ensure hardware has finished reading packet_header before modifying it
    noc_async_writes_flushed();

    // Set noc_send_type back to fused for final chunk
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;

    // Call basic fused atomic inc _with_state for final packet
    fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask>(
        client_interface,
        packet_header,
        packet_src_addr,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{final_noc_address, semaphore_noc_address, val, flush},
        remaining_size);
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
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Cap initial payload size to hardware limit for large pages
    uint32_t init_payload_size = (page_size > FABRIC_MAX_PACKET_SIZE) ? FABRIC_MAX_PACKET_SIZE : page_size;

    fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        ranges,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{noc_address, semaphore_noc_address, val, flush},
        init_payload_size);
}

}  // namespace tt::tt_fabric::mesh::experimental
