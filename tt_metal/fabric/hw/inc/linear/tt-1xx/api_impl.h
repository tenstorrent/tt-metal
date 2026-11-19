// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Note: This file was moved from tt_metal/fabric/hw/inc/linear/api.h. For full commit history, run `git log --follow api_impl.h`.

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

static FORCE_INLINE void fabric_set_unicast_route(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t i) {
#if defined(FABRIC_2D)
    const auto& slot = connection_manager.get(i);
    fabric_set_unicast_route(packet_header, slot.dst_dev_id, slot.dst_mesh_id);
#else
    // 1D unicast, nop
#endif
}

static FORCE_INLINE void fabric_set_mcast_route(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t* range,
    uint8_t i) {
#if defined(FABRIC_2D)
    // 2D multicast
    const auto& slot = connection_manager.get(i);
    if (range[i] != 0) {
        auto hop = range[i];
        switch (static_cast<eth_chan_directions>(slot.tag)) {
            case eth_chan_directions::EAST: {
                fabric_set_mcast_route(packet_header, slot.dst_dev_id, slot.dst_mesh_id, hop, 0, 0, 0);
            } break;
            case eth_chan_directions::WEST: {
                fabric_set_mcast_route(packet_header, slot.dst_dev_id, slot.dst_mesh_id, 0, hop, 0, 0);
            } break;
            case eth_chan_directions::NORTH: {
                fabric_set_mcast_route(packet_header, slot.dst_dev_id, slot.dst_mesh_id, 0, 0, hop, 0);
            } break;
            case eth_chan_directions::SOUTH: {
                fabric_set_mcast_route(packet_header, slot.dst_dev_id, slot.dst_mesh_id, 0, 0, 0, hop);
            } break;
            default: ASSERT(false); break;
        }
    }
#else
    // 1D multicast, nop
#endif
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
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
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_write(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, num_hops[i]);
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
    uint8_t num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, num_hops[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t num_hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_atomic_inc(noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_atomic_inc(
            &slot.sender, packet_header, noc_unicast_atomic_inc_command_header, num_hops[i]);
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
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
    packet_header->payload_size_bytes = 0;
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_atomic_inc_set_state<UpdateMask>(packet_header, num_hops[i], command_header);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t num_hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_scatter_write(noc_unicast_scatter_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_scatter_write(
            &slot.sender, packet_header, src_addr, size, noc_unicast_scatter_command_header, num_hops[i]);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t num_hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_fused_unicast_scatter_write_atomic_inc(
        noc_unicast_scatter_atomic_inc_fused_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_fused_scatter_write_atomic_inc(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_unicast_scatter_atomic_inc_fused_command_header,
            num_hops[i]);
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
    uint8_t num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    static_assert(
        has_flag(UpdateMask, UnicastFusedScatterWriteAtomicIncUpdateMask::Flush),
        "When setting state, the Flush field must be updated to set the chunk encoding for the semaphore increment "
        "chunk");
    packet_header->to_chip_unicast(num_hops);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_fused_scatter_write_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state<UpdateMask>(
            packet_header, num_hops[i], command_header, packet_size_bytes);
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
    uint8_t num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_scatter_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, num_hops[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t num_hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_inline_write(noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_inline_write(
            &slot.sender, packet_header, noc_unicast_inline_write_command_header, num_hops[i]);
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
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
    packet_header->payload_size_bytes = 0;
    populate_unicast_inline_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_inline_write_set_state<UpdateMask>(packet_header, num_hops[i], command_header);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t num_hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_fused_unicast_write_atomic_inc(noc_fused_unicast_atomic_inc_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &slot.sender, packet_header, src_addr, size, noc_fused_unicast_atomic_inc_command_header, num_hops[i]);
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
    uint8_t num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    packet_header->to_chip_unicast(num_hops);
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
            packet_header, num_hops[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t start_distance,
    uint8_t range) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_write(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, start_distance[i], range[i]);
    });
}

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
    uint8_t start_distance,
    uint8_t range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_atomic_inc(noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_atomic_inc(
            &slot.sender, packet_header, noc_unicast_atomic_inc_command_header, start_distance[i], range[i]);
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
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t start_distance, uint8_t range, CommandHeaderT command_header) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
    packet_header->payload_size_bytes = 0;
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_atomic_inc_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t start_distance,
    uint8_t range) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_scatter_write(noc_unicast_scatter_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_scatter_write(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_unicast_scatter_command_header,
            start_distance[i],
            range[i]);
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
    uint8_t start_distance,
    uint8_t range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_scatter_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t start_distance,
    uint8_t range) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_unicast_inline_write(noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_inline_write(
            &slot.sender, packet_header, noc_unicast_inline_write_command_header, start_distance[i], range[i]);
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
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t start_distance, uint8_t range, CommandHeaderT command_header) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_INLINE_WRITE;
    packet_header->payload_size_bytes = 0;
    populate_unicast_inline_fields<UpdateMask>(packet_header, command_header);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_inline_write_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_fused_unicast_write_atomic_inc(noc_fused_unicast_atomic_inc_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_fused_unicast_atomic_inc_command_header,
            start_distance[i],
            range[i]);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t start_distance,
    uint8_t range) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->to_noc_fused_unicast_scatter_write_atomic_inc(
        noc_unicast_scatter_atomic_inc_fused_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t* start_distance,
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_unicast_scatter_atomic_inc_fused_command_header,
            start_distance[i],
            range[i]);
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
    uint8_t start_distance,
    uint8_t range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    static_assert(
        has_flag(UpdateMask, UnicastFusedScatterWriteAtomicIncUpdateMask::Flush),
        "When setting state, the Flush field must be updated to set the chunk encoding for the semaphore increment "
        "chunk");
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_fused_scatter_write_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header, packet_size_bytes);
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
    uint8_t start_distance,
    uint8_t range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header, packet_size_bytes);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint16_t hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_sparse_multicast(
        tt::tt_fabric::SparseMulticastRoutingCommandHeader<PACKET_HEADER_TYPE>{hops});
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint16_t* hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_sparse_multicast_noc_unicast_write(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, hops[i]);
    });
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_sparse_multicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocSparseMulticastWriteCommandHeader sparse_mcast_command_header,
    uint16_t hops) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;

    packet_header->to_chip_sparse_multicast(
        tt::tt_fabric::SparseMulticastRoutingCommandHeader<PACKET_HEADER_TYPE>{hops});
    packet_header->to_noc_sparse_mcast_write(sparse_mcast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

}  // namespace tt::tt_fabric::linear::experimental
