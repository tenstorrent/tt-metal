// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "tt_metal/fabric/hw/inc/mesh/addrgen_api.h"

using namespace tt::tt_fabric::common::experimental;
namespace tt::tt_fabric::mesh::experimental {

// Type trait to detect if a type is an addrgen (has get_noc_addr method)
template <typename T, typename = void>
struct is_addrgen : std::false_type {};

template <typename T>
struct is_addrgen<T, std::void_t<decltype(std::declval<const T&>().get_noc_addr(0))>> : std::true_type {};

// FABRIC_MAX_PACKET_SIZE is available via macro from addrgen_api.h

// hop info e/w/n/s
struct MeshMcastRange {
    uint8_t e;
    uint8_t w;
    uint8_t n;
    uint8_t s;
};

// clang-format off
/**
 * Issues a unicast write from local L1 memory to a destination NOC address.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | True     |
 */
// clang-format on
template <typename FabricSenderType, bool SetRoute = true>
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

// clang-format off
/**
 * Issues a unicast write for all headers in a route via a connection manager.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                       | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | True     |
 */
// clang-format on
template <bool SetRoute = true>
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

// clang-format off
/**
 * Unicast write (stateful): updates only fields selected by UpdateMask on the packet header then submits the payload.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <
    UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, noc_unicast_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Unicast write (stateful, route variant): updates only fields selected by UpdateMask for all headers in the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                       | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Unicast write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                        | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                       | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

// clang-format off
/**
 * Unicast write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                        | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                       | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Issues a unicast atomic increment to a destination semaphore address.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                        | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                       | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 */
// clang-format on
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

// clang-format off
/**
 * Issues a unicast atomic increment for all headers in a route via a connection manager.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                        | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 */
// clang-format on
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

// clang-format off
/**
 * Unicast atomic increment (stateful): updates only fields selected by UpdateMask, then submits the packet header.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Unicast atomic increment (stateful, route variant): updates only fields selected by UpdateMask for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                        | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_atomic_inc_command_header);
    });
}

// clang-format off
/**
 * Unicast atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                        | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
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

// clang-format off
/**
 * Unicast atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                        | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header);
    });
}

// clang-format off
/**
 * Issues a unicast scatter write from local L1 to destination chunks (split into up to four chunks).
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                        | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                       | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 */
// clang-format on
template <typename FabricSenderType, bool SetRoute = true>
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

// clang-format off
/**
 * Issues a unicast scatter write for all headers in a route via a connection manager.
 *
 * Return value: None
 *
 * | Argument                              | Description                              | Type                                           | Required |
 * |---------------------------------------|------------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager         | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route containing packet headers          | uint8_t                                        | True     |
 * | src_addr                              | Source L1 address                        | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                    | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header             | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 */
// clang-format on
template <bool SetRoute = true>
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

// clang-format off
/**
 * Unicast scatter write (stateful): updates only fields selected by UpdateMask, then submits the payload.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                       | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_scatter_write_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_unicast_scatter_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Unicast scatter write (stateful, route variant): updates only fields selected by UpdateMask for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                       | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_scatter_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_scatter_command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Unicast scatter write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                          | Required |
 * |---------------------------------------|-----------------------------------------|-------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                      | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                     | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                      | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_scatter_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

// clang-format off
/**
 * Unicast scatter write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                        | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                       | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Issues a unicast inline write (32-bit) to a destination NOC address.
 *
 * Return value: None
 *
 * | Argument                           | Description                      | Type                                          | Required |
 * |------------------------------------|----------------------------------|-----------------------------------------------|----------|
 * | client_interface                   | Fabric sender interface          | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                      | Packet header to use             | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                         | Destination device id            | uint8_t                                       | True     |
 * | dst_mesh_id                        | Destination mesh id              | uint16_t                                      | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True |
 */
// clang-format on
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

// clang-format off
/**
 * Issues a unicast inline write for all headers in a route via a connection manager.
 *
 * Return value: None
 *
 * | Argument                           | Description                         | Type                                          | Required |
 * |------------------------------------|-------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                 | Routing plane connection manager    | RoutingPlaneConnectionManager&                | True     |
 * | route_id                           | Route containing packet headers     | uint8_t                                       | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header    | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True |
 */
// clang-format on
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

// clang-format off
/**
 * Unicast inline write (stateful): updates only fields selected by UpdateMask, then submits the packet header.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*     　　　   | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*           　　　   | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header           | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | False |
 */
// clang-format on
template <
    UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_inline_fields<UpdateMask>(packet_header, noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Unicast inline write (stateful, route variant): updates only fields selected by UpdateMask for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                          | Type                                            | Required |
 * |---------------------------------------|--------------------------------------|-------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager     | RoutingPlaneConnectionManager&                  | True     |
 * | route_id                              | Route containing packet headers      | uint8_t                                         | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header        | tt::tt_fabric::NocUnicastInlineWriteCommandHeader  | False |
 */
// clang-format on
template <
    UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_inline_write_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_inline_write_command_header);
    });
}

// clang-format off
/**
 * Unicast inline write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                         | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                      | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                     | True     |
 * | command_header                        | Template command header                 | CommandHeaderT               | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
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

// clang-format off
/**
 * Unicast inline write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                        | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_inline_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header);
    });
}

// clang-format off
/**
 * Issues a fused unicast write + atomic increment to a destination and a semaphore address.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                              | Required |
 * |---------------------------------------|-----------------------------------------|---------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*                | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                      | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                           | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                          | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                          | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                          | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header              | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True  |
 */
// clang-format on
template <typename FabricSenderType, bool SetRoute = true>
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

// clang-format off
/**
 * Issues a fused unicast write + atomic increment for all headers in a route via a connection manager.
 *
 * Return value: None
 *
 * | Argument                                  | Description                             | Type                                              | Required |
 * |-------------------------------------------|-----------------------------------------|---------------------------------------------------|----------|
 * | connection_manager                        | Routing plane connection manager        | RoutingPlaneConnectionManager&                    | True     |
 * | route_id                                  | Route containing packet headers         | uint8_t                                           | True     |
 * | src_addr                                  | Source L1 address                       | uint32_t                                          | True     |
 * | size                                      | Payload size in bytes                   | uint32_t                                          | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header                  | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True  |
 */
// clang-format on
template <bool SetRoute = true>
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

// clang-format off
/**
 * Fused unicast write + atomic increment (stateful): updates only masked fields, then submits the payload.
 *
 * Return value: None
 *
 * | Argument                                  | Description                            | Type                                              | Required |
 * |-------------------------------------------|----------------------------------------|---------------------------------------------------|----------|
 * | client_interface                          | Fabric sender interface                | tt_l1_ptr WorkerToFabricEdmSender*                | True     |
 * | packet_header                             | Packet header to use                   | volatile PACKET_HEADER_TYPE*                      | True     |
 * | src_addr                                  | Source L1 address                      | uint32_t                                          | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header                 | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | False |
 * | packet_size_bytes                         | Payload size override if masked        | uint16_t                                          | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Fused unicast write + atomic increment (stateful, route variant): updates only masked fields for all headers.
 *
 * Return value: None
 *
 * | Argument                                  | Description                            | Type                                              | Required |
 * |-------------------------------------------|----------------------------------------|---------------------------------------------------|----------|
 * | connection_manager                        | Routing plane connection manager       | RoutingPlaneConnectionManager&                    | True     |
 * | route_id                                  | Route containing packet headers        | uint8_t                                           | True     |
 * | src_addr                                  | Source L1 address                      | uint32_t                                          | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header                 | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | False |
 * | packet_size_bytes                         | Payload size override if masked        | uint16_t                                          | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Fused unicast write + atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                       | Description                             | Type                             | Required |
 * |--------------------------------|-----------------------------------------|----------------------------------|----------|
 * | packet_header                  | Packet header to use                    | volatile PACKET_HEADER_TYPE*     | True     |
 * | dst_dev_id                     | Destination device id                   | uint8_t                          | True     |
 * | dst_mesh_id                    | Destination mesh id                     | uint16_t                         | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

// clang-format off
/**
 * Fused unicast write + atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                       | Description                             | Type                             | Required |
 * |--------------------------------|-----------------------------------------|----------------------------------|----------|
 * | connection_manager             | Routing plane connection manager        | RoutingPlaneConnectionManager&   | True     |
 * | route_id                       | Route whose headers will be updated     | uint8_t                          | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Multicast unicast write: issues a unicast write with chip-level multicast routing metadata.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface           | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header              | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id                 | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id                | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges                     | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 */
// clang-format on
template <typename FabricSenderType, bool SetRoute = true>
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

// clang-format off
/**
 * Multicast unicast write (route variant): issues writes for all headers in the route using multicast routing metadata.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | connection_manager         | Routing plane connection manager        | RoutingPlaneConnectionManager&             | True     |
 * | route_id                   | Route containing packet headers         | uint8_t                                    | True     |
 * | ranges                     | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                     | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast unicast write (stateful): updates only fields selected by UpdateMask, then submits the payload.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface           | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header              | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | False    |
 * | packet_size_bytes          | Payload size override if masked         | uint16_t                                   | False    |
 */
// clang-format on
template <
    UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, noc_unicast_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Multicast unicast write (stateful, route variant): updates only fields selected by UpdateMask for all headers.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                      | Required |
 * |----------------------------|-----------------------------------------|-------------------------------------------|----------|
 * | connection_manager         | Routing plane connection manager        | RoutingPlaneConnectionManager&            | True     |
 * | route_id                   | Route containing packet headers         | uint8_t                                   | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                  | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader    | False    |
 * | packet_size_bytes          | Payload size override if masked         | uint16_t                                  | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::None, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Multicast unicast write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                 | Required |
 * |---------------------------------------|-----------------------------------------|--------------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*         | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                              | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                             | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                       | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                             | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    populate_unicast_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

// clang-format off
/**
 * Multicast unicast write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                 | Required |
 * |---------------------------------------|-----------------------------------------|--------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&       | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                              | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*               | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                       | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                             | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Multicast unicast atomic increment: issues a unicast atomic inc with chip-level multicast routing metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                        | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                       | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                          | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast unicast atomic increment (route variant): issues atomic inc for all headers using multicast metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                        | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                          | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast unicast atomic inc (stateful): updates only fields selected by UpdateMask, then submits the packet header.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, noc_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Multicast unicast atomic inc (stateful, route variant): updates only masked fields for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                        | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_atomic_inc_command_header);
    });
}

// clang-format off
/**
 * Multicast unicast atomic inc (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                         | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header = nullptr) {
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
    packet_header->payload_size_bytes = 0;
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    populate_unicast_atomic_inc_fields<UpdateMask>(packet_header, command_header);
}

// clang-format off
/**
 * Multicast unicast atomic inc (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                       | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                         | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_atomic_inc_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header);
    });
}

// clang-format off
/**
 * Multicast unicast scatter write: issues a unicast scatter write with multicast routing metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                        | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                       | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                          | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast unicast scatter write (route variant): issues writes for all headers using multicast metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                       | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                         | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader | True     |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast unicast scatter write (stateful): updates only masked fields, then submits the payload.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_scatter_write_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_unicast_scatter_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Multicast unicast scatter write (stateful, route variant): updates only masked fields for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                       | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_scatter_write_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_unicast_scatter_command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Multicast unicast scatter write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                         | True     |
 * | command_header                        | Template command header                  | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    populate_unicast_scatter_write_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

// clang-format off
/**
 * Multicast unicast scatter write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                       | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                         | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges[i], command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Multicast unicast inline write: issues a 32-bit inline write with multicast routing metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                        | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                       | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                          | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header           | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True  |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast unicast inline write (route variant): issues inline writes for all headers using multicast metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                          | Type                                           | Required |
 * |---------------------------------------|--------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager     | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route containing packet headers      | uint8_t                                        | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                         | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header        | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True  |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast unicast inline write (stateful): updates only masked fields, then submits the packet header.
 *
 * Return value: None
 *
 * | Argument                              | Description                     | Type                                           | Required |
 * |---------------------------------------|---------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface         | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use            | volatile PACKET_HEADER_TYPE*                   | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header   | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | False |
 */
// clang-format on
template <
    UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_inline_fields<UpdateMask>(packet_header, noc_unicast_inline_write_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Multicast unicast inline write (stateful, route variant): updates only masked fields for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                        | True     |
 * | noc_unicast_inline_write_command_header | Inline write command header           | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | False |
 */
// clang-format on
template <
    UnicastInlineWriteUpdateMask UpdateMask = UnicastInlineWriteUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_inline_write_with_state<UpdateMask>(
            &slot.sender, packet_header, noc_unicast_inline_write_command_header);
    });
}

// clang-format off
/**
 * Multicast unicast inline write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                         | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
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

// clang-format off
/**
 * Multicast unicast inline write (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                        | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                          | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
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

// clang-format off
/**
 * Multicast fused unicast write + atomic increment: issues fused op with multicast routing metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                             | Required |
 * |---------------------------------------|-----------------------------------------|--------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*               | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                     | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                          | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                         | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                            | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                         | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                         | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header              | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 */
// clang-format on
template <typename FabricSenderType, bool SetRoute = true>
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

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (route variant): issues fused ops for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager     | RoutingPlaneConnectionManager&                   | True     |
 * | route_id                              | Route containing packet headers      | uint8_t                                          | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                           | True     |
 * | src_addr                              | Source L1 address                    | uint32_t                                         | True     |
 * | size                                  | Payload size in bytes                | uint32_t                                         | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header           | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 */
// clang-format on
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

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (stateful): updates only masked fields, then submits payload.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                       | Required |
 * |---------------------------------------|----------------------------------|---------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface          | tt_l1_ptr WorkerToFabricEdmSender*                | True     |
 * | packet_header                         | Packet header to use             | volatile PACKET_HEADER_TYPE*                      | True     |
 * | src_addr                              | Source L1 address                | uint32_t                                          | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header       | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | False |
 * | packet_size_bytes                     | Payload size override if masked  | uint16_t                                          | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    [[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (stateful, route variant): updates only masked fields for all headers.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                       | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header       | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | False    |
 * | packet_size_bytes                     | Payload size override if masked          | uint16_t                                     | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask>(
            &slot.sender, packet_header, src_addr, noc_fused_unicast_atomic_inc_command_header, packet_size_bytes);
    });
}

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | ranges                                | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                         | True     |
 * | command_header                        | Template fused command header           | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
    fabric_set_mcast_route(packet_header, dst_dev_id, dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
    packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(packet_header, packet_size_bytes, command_header);
}

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route whose headers will be updated     | uint8_t                                       | True     |
 * | ranges                                | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*                         | True     |
 * | command_header                        | Template fused command header           | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0) {
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
// clang-format off
/**
 * Unicast write (addrgen overload): sends payload to destination computed from address generator.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device ID                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh ID                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | addrgen                               | Address generator (e.g. TensorAccessor) | const AddrGenType&                            | True     |
 * | page_id                               | Page ID to compute NOC address          | uint32_t                                      | True     |
 * | offset                                | Offset within page                      | uint32_t                                      | False    |
 */
// clang-format on
template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset = 0) {
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

// clang-format off
/**
 * Unicast write (stateful, addrgen overload): updates only fields selected by UpdateMask on the packet header then submits the payload, with destination computed from address generator.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | UpdateMask                            | Template parameter: which fields to update | UnicastWriteUpdateMask                     | False    |
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device ID                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh ID                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | addrgen                               | Address generator (e.g. TensorAccessor) | const AddrGenType&                            | True     |
 * | page_id                               | Page ID to compute NOC address          | uint32_t                                      | True     |
 * | offset                                | Offset within page                      | uint32_t                                      | False    |
 */
// clang-format on
template <
    UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize,
    typename FabricSenderType,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset = 0) {
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

// clang-format off
/**
 * Unicast write (set-state, addrgen overload): pre-configures headers for repeated use, with destination computed from address generator.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | UpdateMask                            | Template parameter: which fields to update | UnicastWriteUpdateMask                     | False    |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | addrgen                               | Address generator (e.g. TensorAccessor) | const AddrGenType&                            | True     |
 * | page_id                               | Page ID to compute NOC address          | uint32_t                                      | True     |
 * | offset                                | Offset within page                      | uint32_t                                      | False    |
 */
// clang-format on
template <
    UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset = 0) {
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Cap initial payload size to hardware limit for large pages
    // The WithState calls in the loop will handle actual chunking
    uint32_t init_payload_size = (page_size > FABRIC_MAX_PACKET_SIZE) ? FABRIC_MAX_PACKET_SIZE : page_size;

    fabric_unicast_noc_unicast_write_set_state<UpdateMask>(
        packet_header, dst_dev_id, dst_mesh_id, tt::tt_fabric::NocUnicastCommandHeader{noc_address}, init_payload_size);
}

// Fused Unicast Write + Atomic Inc Addrgen Overloads
// clang-format off
/**
 * Fused unicast write with atomic increment (addrgen overload): sends payload and increments semaphore, with destination computed from address generator.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device ID                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh ID                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | addrgen                               | Address generator (e.g. TensorAccessor) | const AddrGenType&                            | True     |
 * | page_id                               | Page ID to compute NOC address          | uint32_t                                      | True     |
 * | semaphore_noc_address                 | NOC address of semaphore to increment   | uint64_t                                      | True     |
 * | val                                   | Increment value                         | uint16_t                                      | True     |
 * | offset                                | Offset within page                      | uint32_t                                      | False    |
 * | flush                                 | Flush data cache after write            | bool                                          | False    |
 */
// clang-format on
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
    uint32_t offset = 0,
    bool flush = true) {
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

// clang-format off
/**
 * Fused unicast write with atomic increment (stateful, addrgen overload): updates only fields selected by UpdateMask on the packet header then submits the payload and increments semaphore, with destination computed from address generator.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | UpdateMask                            | Template parameter: which fields to update | UnicastFusedAtomicIncUpdateMask            | False    |
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device ID                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh ID                     | uint16_t                                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | addrgen                               | Address generator (e.g. TensorAccessor) | const AddrGenType&                            | True     |
 * | page_id                               | Page ID to compute NOC address          | uint32_t                                      | True     |
 * | semaphore_noc_address                 | NOC address of semaphore to increment   | uint64_t                                      | True     |
 * | val                                   | Increment value                         | uint16_t                                      | True     |
 * | offset                                | Offset within page                      | uint32_t                                      | False    |
 * | flush                                 | Flush data cache after write            | bool                                          | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                                                 UnicastFusedAtomicIncUpdateMask::PayloadSize,
    typename FabricSenderType,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
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
    uint32_t offset = 0,
    bool flush = true) {
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

// clang-format off
/**
 * Fused unicast write with atomic increment (set-state, addrgen overload): pre-configures headers for repeated use, with destination computed from address generator.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | UpdateMask                            | Template parameter: which fields to update | UnicastFusedAtomicIncUpdateMask            | False    |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | dst_dev_id                            | Destination device id                   | uint8_t                                       | True     |
 * | dst_mesh_id                           | Destination mesh id                     | uint16_t                                      | True     |
 * | addrgen                               | Address generator (e.g. TensorAccessor) | const AddrGenType&                            | True     |
 * | page_id                               | Page ID to compute NOC address          | uint32_t                                      | True     |
 * | semaphore_noc_address                 | NOC address of semaphore to increment   | uint64_t                                      | True     |
 * | val                                   | Increment value                         | uint16_t                                      | True     |
 * | offset                                | Offset within page                      | uint32_t                                      | False    |
 * | flush                                 | Flush data cache after write            | bool                                          | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask =
        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
        UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush |
        UnicastFusedAtomicIncUpdateMask::PayloadSize,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset = 0,
    bool flush = true) {
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

// clang-format off
/**
 * Unicast scatter write (TensorAccessor overload): computes NOC addresses from addrgen for two pages.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface   | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device ID                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh ID                     | uint16_t                                   | True     |
 * | src_addr           | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id0           | First page index                        | uint32_t                                   | True     |
 * | page_id1           | Second page index                       | uint32_t                                   | True     |
 * | offset0            | Offset within first page                | uint32_t                                   | False    |
 * | offset1            | Offset within second page               | uint32_t                                   | False    |
 */
// clang-format on
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
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
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

// clang-format off
/**
 * Unicast scatter write (stateful, TensorAccessor overload): updates only masked fields, computes NOC addresses from addrgen for two pages.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask         | Template parameter: which fields to update | UnicastScatterWriteUpdateMask           | False    |
 * | client_interface   | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device ID                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh ID                     | uint16_t                                   | True     |
 * | src_addr           | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id0           | First page index                        | uint32_t                                   | True     |
 * | page_id1           | Second page index                       | uint32_t                                   | True     |
 * | offset0            | Offset within first page                | uint32_t                                   | False    |
 * | offset1            | Offset within second page               | uint32_t                                   | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::DstAddrs |
                                               UnicastScatterWriteUpdateMask::ChunkSizes |
                                               UnicastScatterWriteUpdateMask::PayloadSize,
    typename FabricSenderType,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
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

// clang-format off
/**
 * Unicast scatter write (set-state, TensorAccessor overload): pre-configures headers for repeated use, computes NOC addresses from addrgen for two pages.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask         | Template parameter: which fields to update | UnicastScatterWriteUpdateMask           | False    |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device ID                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh ID                     | uint16_t                                   | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id0           | First page index                        | uint32_t                                   | True     |
 * | page_id1           | Second page index                       | uint32_t                                   | True     |
 * | offset0            | Offset within first page                | uint32_t                                   | False    |
 * | offset1            | Offset within second page               | uint32_t                                   | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::DstAddrs |
                                               UnicastScatterWriteUpdateMask::ChunkSizes |
                                               UnicastScatterWriteUpdateMask::PayloadSize,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
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

// clang-format off
/**
 * Multicast unicast write (TensorAccessor overload): computes NOC address and page size from addrgen.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface   | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges             | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | src_addr           | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id            | Page index                              | uint32_t                                   | True     |
 * | offset             | Offset within page                      | uint32_t                                   | False    |
 */
// clang-format on
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
    uint32_t offset = 0) {
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

// clang-format off
/**
 * Multicast unicast write (stateful, TensorAccessor overload): updates only masked fields, computes NOC address from addrgen.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask         | Template parameter: which fields to update | UnicastWriteUpdateMask                  | False    |
 * | client_interface   | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges             | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | src_addr           | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id            | Page index                              | uint32_t                                   | True     |
 * | offset             | Offset within page                      | uint32_t                                   | False    |
 */
// clang-format on
template <
    UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize,
    typename FabricSenderType,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset = 0) {
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

// clang-format off
/**
 * Multicast unicast write (set-state, TensorAccessor overload): pre-configures headers for repeated use, computes NOC address from addrgen.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask         | Template parameter: which fields to update | UnicastWriteUpdateMask                  | False    |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges             | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id            | Page index                              | uint32_t                                   | True     |
 * | offset             | Offset within page                      | uint32_t                                   | False    |
 */
// clang-format on
template <
    UnicastWriteUpdateMask UpdateMask = UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset = 0) {
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

// clang-format off
/**
 * Multicast scatter write (TensorAccessor overload): computes NOC addresses from addrgen for two pages with multicast routing.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface   | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges             | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | src_addr           | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id0           | First page index                        | uint32_t                                   | True     |
 * | page_id1           | Second page index                       | uint32_t                                   | True     |
 * | offset0            | Offset within first page                | uint32_t                                   | False    |
 * | offset1            | Offset within second page               | uint32_t                                   | False    |
 */
// clang-format on
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
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
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

// clang-format off
/**
 * Multicast scatter write (stateful, TensorAccessor overload): updates only masked fields, computes NOC addresses from addrgen for two pages with multicast routing.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask         | Template parameter: which fields to update | UnicastScatterWriteUpdateMask           | False    |
 * | client_interface   | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges             | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | src_addr           | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id0           | First page index                        | uint32_t                                   | True     |
 * | page_id1           | Second page index                       | uint32_t                                   | True     |
 * | offset0            | Offset within first page                | uint32_t                                   | False    |
 * | offset1            | Offset within second page               | uint32_t                                   | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::DstAddrs |
                                               UnicastScatterWriteUpdateMask::ChunkSizes |
                                               UnicastScatterWriteUpdateMask::PayloadSize,
    typename FabricSenderType,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
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
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
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

// clang-format off
/**
 * Multicast scatter write (set-state, TensorAccessor overload): pre-configures headers for repeated use, computes NOC addresses from addrgen for two pages with multicast routing.
 *
 * Return value: None
 *
 * | Argument           | Description                             | Type                                       | Required |
 * |--------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask         | Template parameter: which fields to update | UnicastScatterWriteUpdateMask           | False    |
 * | packet_header      | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id         | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id        | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges             | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | addrgen            | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id0           | First page index                        | uint32_t                                   | True     |
 * | page_id1           | Second page index                       | uint32_t                                   | True     |
 * | offset0            | Offset within first page                | uint32_t                                   | False    |
 * | offset1            | Offset within second page               | uint32_t                                   | False    |
 */
// clang-format on
template <
    UnicastScatterWriteUpdateMask UpdateMask = UnicastScatterWriteUpdateMask::DstAddrs |
                                               UnicastScatterWriteUpdateMask::ChunkSizes |
                                               UnicastScatterWriteUpdateMask::PayloadSize,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    const AddrGenType& addrgen,
    uint32_t page_id0,
    uint32_t page_id1,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
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

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (TensorAccessor overload): computes NOC address from addrgen, sends payload with multicast routing and increments semaphore.
 *
 * Return value: None
 *
 * | Argument               | Description                             | Type                                       | Required |
 * |------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface       | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header          | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id             | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id            | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges                 | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | src_addr               | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen                | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id                | Page index                              | uint32_t                                   | True     |
 * | semaphore_noc_address  | NOC address of semaphore to increment   | uint64_t                                   | True     |
 * | val                    | Increment value                         | uint16_t                                   | True     |
 * | offset                 | Offset within page                      | uint32_t                                   | False    |
 * | flush                  | Flush data cache after write            | bool                                       | False    |
 */
// clang-format on
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
    uint32_t offset = 0,
    bool flush = true) {
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

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (stateful, TensorAccessor overload): updates only masked fields, computes NOC address from addrgen.
 *
 * Return value: None
 *
 * | Argument               | Description                             | Type                                       | Required |
 * |------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask             | Template parameter: which fields to update | UnicastFusedAtomicIncUpdateMask         | False    |
 * | client_interface       | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header          | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | src_addr               | Source L1 address                       | uint32_t                                   | True     |
 * | addrgen                | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id                | Page index                              | uint32_t                                   | True     |
 * | semaphore_noc_address  | NOC address of semaphore to increment   | uint64_t                                   | True     |
 * | val                    | Increment value                         | uint16_t                                   | True     |
 * | offset                 | Offset within page                      | uint32_t                                   | False    |
 * | flush                  | Flush data cache after write            | bool                                       | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                                                 UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                                                 UnicastFusedAtomicIncUpdateMask::PayloadSize,
    typename FabricSenderType,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset = 0,
    bool flush = true) {
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

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (set-state, TensorAccessor overload): pre-configures headers for repeated use, computes NOC address from addrgen.
 *
 * Return value: None
 *
 * | Argument               | Description                             | Type                                       | Required |
 * |------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | UpdateMask             | Template parameter: which fields to update | UnicastFusedAtomicIncUpdateMask         | False    |
 * | packet_header          | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | dst_dev_id             | Destination device id                   | uint8_t                                    | True     |
 * | dst_mesh_id            | Destination mesh id                     | uint16_t                                   | True     |
 * | ranges                 | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&                      | True     |
 * | addrgen                | Address generator (e.g., TensorAccessor)| AddrGenType                                | True     |
 * | page_id                | Page index                              | uint32_t                                   | True     |
 * | semaphore_noc_address  | NOC address of semaphore to increment   | uint64_t                                   | True     |
 * | val                    | Increment value                         | uint16_t                                   | True     |
 * | offset                 | Offset within page                      | uint32_t                                   | False    |
 * | flush                  | Flush data cache after write            | bool                                       | False    |
 */
// clang-format on
template <
    UnicastFusedAtomicIncUpdateMask UpdateMask = UnicastFusedAtomicIncUpdateMask::None,
    typename AddrGenType,
    typename = std::enable_if_t<is_addrgen<AddrGenType>::value>>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint32_t offset = 0,
    bool flush = true) {
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
