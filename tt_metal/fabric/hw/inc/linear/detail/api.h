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
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"

using namespace tt::tt_fabric::common::experimental;

namespace tt::tt_fabric::linear::experimental::detail {

// Type trait to detect if a type is an addrgen (has get_noc_addr method)
template <typename T, typename = void>
struct is_addrgen : std::false_type {};

template <typename T>
struct is_addrgen<T, std::void_t<decltype(std::declval<const T&>().get_noc_addr(0))>> : std::true_type {};

// clang-format off
/**
 * Unicast write (single-packet, raw-size): sends exactly one packet; use fabric_unicast_noc_unicast_write
 * for auto-packetizing large payloads.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*            | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                  | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | True     |
 * | num_hops                              | Unicast hop count                       | uint8_t                                       | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write_single_packet(
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

// clang-format off
/**
 * Unicast write (single-packet, connection manager): sends exactly one packet per connection;
 * use fabric_unicast_noc_unicast_write for auto-packetizing large payloads.
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                      | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_write_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_write_single_packet(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, num_hops[i]);
    });
}

// clang-format off
/**
 * Issues a single-packet unicast scatter write from local L1 to destination chunks (split into up to four chunks).
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 * | num_hops                              | Unicast hop count                       | uint8_t                                        | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_scatter_write_single_packet(
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

// clang-format off
/**
 * Issues a single-packet unicast scatter write for all headers in a route via a connection manager.
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
 * | num_hops                              | Per-header unicast hop counts            | uint8_t*                                       | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_scatter_write_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_scatter_write_single_packet(
            &slot.sender, packet_header, src_addr, size, noc_unicast_scatter_command_header, num_hops[i]);
    });
}

// clang-format off
/**
 * Issues a single-packet unicast fused scatter write + atomic increment (2 scatter chunks + semaphore inc).
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | client_interface                                  | Fabric sender interface           | tt_l1_ptr WorkerToFabricEdmSender*                          | True     |
 * | packet_header                                     | Packet header to use              | volatile PACKET_HEADER_TYPE*                                | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes              | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header  | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 * | num_hops                                          | Unicast hop count                  | uint8_t                                                    | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet(
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

// clang-format off
/**
 * Issues a single-packet unicast fused scatter write + atomic increment for all headers in a route via a connection manager.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | connection_manager                                | Routing plane connection manager | RoutingPlaneConnectionManager&                            | True     |
 * | route_id                                          | Route containing packet headers   | uint8_t                                                    | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes              | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 * | num_hops                                          | Per-header unicast hop counts     | uint8_t*                                                  | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_unicast_scatter_atomic_inc_fused_command_header,
            num_hops[i]);
    });
}

// clang-format off
/**
 * Fused unicast write + atomic increment (single-packet, raw-size): sends exactly one packet;
 * use fabric_unicast_noc_fused_unicast_with_atomic_inc for auto-packetizing large payloads.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                              | Required |
 * |---------------------------------------|-----------------------------------------|---------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*                | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                      | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                          | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                          | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header              | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True  |
 * | num_hops                              | Unicast hop count                       | uint8_t                                           | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet(
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

// clang-format off
/**
 * Fused unicast write + atomic increment (single-packet, connection manager): sends exactly one packet
 * per connection; use fabric_unicast_noc_fused_unicast_with_atomic_inc for auto-packetizing large payloads.
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
 * | num_hops                                  | Per-header unicast hop counts           | uint8_t*                                          | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t* num_hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet(
            &slot.sender, packet_header, src_addr, size, noc_fused_unicast_atomic_inc_command_header, num_hops[i]);
    });
}

// clang-format off
/**
 * Multicast unicast write (single-packet, raw-size): sends exactly one packet with multicast routing;
 * use fabric_multicast_noc_unicast_write for auto-packetizing large payloads.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface           | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header              | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 * | start_distance             | Multicast start distance                | uint8_t                                    | True     |
 * | range                      | Multicast range                         | uint8_t                                    | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_write_single_packet(
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

// clang-format off
/**
 * Multicast unicast write (single-packet, connection manager): sends exactly one packet per connection
 * with multicast routing; use fabric_multicast_noc_unicast_write for auto-packetizing large payloads.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | connection_manager         | Routing plane connection manager        | RoutingPlaneConnectionManager&             | True     |
 * | route_id                   | Route containing packet headers         | uint8_t                                    | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 * | start_distance             | Per-header multicast start distance     | uint8_t*                                   | True     |
 * | range                      | Per-header multicast range              | uint8_t*                                   | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_unicast_write_single_packet(
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
        fabric_multicast_noc_unicast_write_single_packet(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, start_distance[i], range[i]);
    });
}

// clang-format off
/**
 * Single-packet multicast unicast scatter write: issues a unicast scatter write with multicast routing metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*             | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                   | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 * | start_distance                        | Multicast start distance                | uint8_t                                        | True     |
 * | range                                 | Multicast range                         | uint8_t                                        | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_scatter_write_single_packet(
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

// clang-format off
/**
 * Single-packet multicast unicast scatter write (route variant): issues writes for all headers using multicast metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                | True     |
 * | route_id                              | Route containing packet headers         | uint8_t                                       | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader | True     |
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                                      | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                                      | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_scatter_write_single_packet(
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
        fabric_multicast_noc_scatter_write_single_packet(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_unicast_scatter_command_header,
            start_distance[i],
            range[i]);
    });
}

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (single-packet, raw-size): sends exactly one packet
 * with multicast routing; use fabric_multicast_noc_fused_unicast_with_atomic_inc for auto-packetizing.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                             | Required |
 * |---------------------------------------|-----------------------------------------|--------------------------------------------------|----------|
 * | client_interface                      | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*               | True     |
 * | packet_header                         | Packet header to use                    | volatile PACKET_HEADER_TYPE*                     | True     |
 * | src_addr                              | Source L1 address                       | uint32_t                                         | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                         | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header              | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 * | start_distance                        | Multicast start distance                | uint8_t                                          | True     |
 * | range                                 | Multicast range                         | uint8_t                                          | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet(
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

// clang-format off
/**
 * Multicast fused unicast write + atomic increment (single-packet, connection manager): sends exactly
 * one packet per connection; use fabric_multicast_noc_fused_unicast_with_atomic_inc for auto-packetizing.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                          | Required |
 * |---------------------------------------|-----------------------------------------|-----------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager     | RoutingPlaneConnectionManager&                   | True     |
 * | route_id                              | Route containing packet headers      | uint8_t                                          | True     |
 * | src_addr                              | Source L1 address                    | uint32_t                                         | True     |
 * | size                                  | Payload size in bytes                | uint32_t                                         | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header           | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 * | start_distance                        | Per-header multicast start distance  | uint8_t*                                         | True     |
 * | range                                 | Per-header multicast range           | uint8_t*                                         | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet(
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
        fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_fused_unicast_atomic_inc_command_header,
            start_distance[i],
            range[i]);
    });
}

// clang-format off
/**
 * Single-packet multicast fused scatter write + atomic increment (2 scatter chunks + semaphore inc).
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | client_interface                                  | Fabric sender interface           | tt_l1_ptr WorkerToFabricEdmSender*                         | True     |
 * | packet_header                                     | Packet header to use              | volatile PACKET_HEADER_TYPE*                               | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes             | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 * | start_distance                                    | Multicast start distance          | uint8_t                                                    | True     |
 * | range                                             | Multicast range                   | uint8_t                                                    | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet(
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

// clang-format off
/**
 * Single-packet multicast fused scatter write + atomic increment (route variant): issues for all headers.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | connection_manager                                | Routing plane connection manager | RoutingPlaneConnectionManager&                             | True     |
 * | route_id                                          | Route containing packet headers   | uint8_t                                                    | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes             | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 * | start_distance                                    | Per-header multicast start distance | uint8_t*                                                | True     |
 * | range                                             | Per-header multicast range       | uint8_t*                                                   | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet(
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
        fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet(
            &slot.sender,
            packet_header,
            src_addr,
            size,
            noc_unicast_scatter_atomic_inc_fused_command_header,
            start_distance[i],
            range[i]);
    });
}

// clang-format off
/**
 * Sparse multicast unicast write (single-packet, raw-size): sends exactly one packet with sparse
 * multicast routing; use fabric_sparse_multicast_noc_unicast_write for the naming-convention wrapper.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | client_interface           | Fabric sender interface                 | tt_l1_ptr WorkerToFabricEdmSender*         | True     |
 * | packet_header              | Packet header to use                    | volatile PACKET_HEADER_TYPE*               | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 * | hops                       | Sparse multicast hop bitmask            | uint16_t                                   | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write_single_packet(
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

// clang-format off
/**
 * Sparse multicast unicast write (single-packet, connection manager): sends exactly one packet per
 * connection; use fabric_sparse_multicast_noc_unicast_write for the naming-convention wrapper.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | connection_manager         | Routing plane connection manager        | RoutingPlaneConnectionManager&             | True     |
 * | route_id                   | Route containing packet headers         | uint8_t                                    | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 * | hops                       | Per-header sparse multicast hop bitmask | uint16_t*                                  | True     |
 */
// clang-format on
FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write_single_packet(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint16_t* hops) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_sparse_multicast_noc_unicast_write_single_packet(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, hops[i]);
    });
}

}  // namespace tt::tt_fabric::linear::experimental::detail
