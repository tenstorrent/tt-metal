// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

namespace tt::tt_fabric::mesh::experimental::detail {

// Type trait to detect if a type is an addrgen (has get_noc_addr method)
template <typename T, typename = void>
struct is_addrgen : std::false_type {};

template <typename T>
struct is_addrgen<T, std::void_t<decltype(std::declval<const T&>().get_noc_addr(0))>> : std::true_type {};

// clang-format off
/**
 * Issues a single-packet unicast write from local L1 memory to a destination NOC address.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_unicast_noc_unicast_write which auto-packetizes.
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
FORCE_INLINE void fabric_unicast_noc_unicast_write_single_packet(
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
 * Issues a single-packet unicast write for all headers in a route via a connection manager.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_unicast_noc_unicast_write which auto-packetizes.
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
FORCE_INLINE void fabric_unicast_noc_unicast_write_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write_single_packet<decltype(slot.sender), SetRoute>(
            &slot.sender, packet_header, slot.dst_dev_id, slot.dst_mesh_id, src_addr, size, noc_unicast_command_header);
    });
}

// clang-format off
/**
 * Issues a single-packet unicast scatter write from local L1 to destination chunks (split into up to four chunks).
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
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
FORCE_INLINE void fabric_unicast_noc_scatter_write_single_packet(
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
 * Issues a single-packet unicast scatter write for all headers in a route via a connection manager.
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
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
FORCE_INLINE void fabric_unicast_noc_scatter_write_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_scatter_write_single_packet<decltype(slot.sender), SetRoute>(
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
 * Issues a single-packet unicast fused scatter write + atomic increment (2 scatter chunks + semaphore inc).
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | client_interface                                  | Fabric sender interface           | tt_l1_ptr WorkerToFabricEdmSender*                         | True     |
 * | packet_header                                     | Packet header to use              | volatile PACKET_HEADER_TYPE*                               | True     |
 * | dst_dev_id                                        | Destination device id             | uint8_t                                                    | True     |
 * | dst_mesh_id                                       | Destination mesh id               | uint16_t                                                   | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes              | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header  | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 */
// clang-format on
template <typename FabricSenderType, bool SetRoute = true>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet(
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

// clang-format off
/**
 * Issues a single-packet unicast fused scatter write + atomic increment for all headers in a route via a connection manager.
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | connection_manager                                | Routing plane connection manager | RoutingPlaneConnectionManager&                             | True     |
 * | route_id                                          | Route containing packet headers   | uint8_t                                                    | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes              | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 */
// clang-format on
template <bool SetRoute = true>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet<decltype(slot.sender), SetRoute>(
            &slot.sender,
            packet_header,
            slot.dst_dev_id,
            slot.dst_mesh_id,
            src_addr,
            size,
            noc_unicast_scatter_atomic_inc_fused_command_header);
    });
}

// clang-format off
/**
 * Issues a single-packet fused unicast write + atomic increment to a destination and a semaphore address.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_unicast_noc_fused_unicast_with_atomic_inc which auto-packetizes.
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
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet(
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
 * Issues a single-packet fused unicast write + atomic increment for all headers in a route via a connection manager.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_unicast_noc_fused_unicast_with_atomic_inc which auto-packetizes.
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
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet<decltype(slot.sender), SetRoute>(
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
 * Single-packet multicast unicast write: issues a unicast write with chip-level multicast routing metadata.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_multicast_noc_unicast_write which auto-packetizes.
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
FORCE_INLINE void fabric_multicast_noc_unicast_write_single_packet(
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
 * Single-packet multicast unicast write (connection manager variant): issues writes for all headers
 * in the route using multicast routing metadata.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_multicast_noc_unicast_write which auto-packetizes.
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
FORCE_INLINE void fabric_multicast_noc_unicast_write_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_unicast_write_single_packet(
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
 * Multicast single-packet scatter write: issues a unicast scatter write with multicast routing metadata.
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
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
FORCE_INLINE void fabric_multicast_noc_scatter_write_single_packet(
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
 * Multicast single-packet scatter write (connection manager variant): issues writes for all headers using multicast metadata.
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
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
FORCE_INLINE void fabric_multicast_noc_scatter_write_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_scatter_write_single_packet(
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
 * Single-packet multicast fused unicast write + atomic increment: issues fused op with multicast routing metadata.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_multicast_noc_fused_unicast_with_atomic_inc which auto-packetizes.
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
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet(
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
 * Single-packet multicast fused unicast write + atomic increment (connection manager variant): issues fused ops for all headers.
 * For payloads larger than FABRIC_MAX_PACKET_SIZE, use fabric_multicast_noc_fused_unicast_with_atomic_inc which auto-packetizes.
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
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet(
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
 * Multicast single-packet fused scatter write + atomic increment (2 scatter chunks + semaphore inc).
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | client_interface                                  | Fabric sender interface           | tt_l1_ptr WorkerToFabricEdmSender*                         | True     |
 * | packet_header                                     | Packet header to use              | volatile PACKET_HEADER_TYPE*                               | True     |
 * | dst_dev_id                                        | Destination device id             | uint8_t                                                    | True     |
 * | dst_mesh_id                                       | Destination mesh id              | uint16_t                                                   | True     |
 * | ranges                                            | Multicast hop counts (E/W/N/S)    | const MeshMcastRange&                                      | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes             | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 */
// clang-format on
template <typename FabricSenderType, bool SetRoute = true>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet(
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

// clang-format off
/**
 * Multicast single-packet fused scatter write + atomic increment (route variant): issues for all headers.
 * For large payloads, use addrgen overloads. This variant must not exceed FABRIC_MAX_PACKET_SIZE.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | connection_manager                                | Routing plane connection manager | RoutingPlaneConnectionManager&                             | True     |
 * | route_id                                          | Route containing packet headers   | uint8_t                                                    | True     |
 * | ranges                                            | Per-header multicast hop counts (E/W/N/S) | const MeshMcastRange*                            | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes             | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet(
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

}  // namespace tt::tt_fabric::mesh::experimental::detail
