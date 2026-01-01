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
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header);

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
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header);

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
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header);

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
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr);

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
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr);

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
    CommandHeaderT command_header);

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
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header);

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
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header);

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
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header);

// clang-format off
/**
 * Issues a unicast fused scatter write + atomic increment (2 scatter chunks + semaphore inc).
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
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header);

// clang-format off
/**
 * Issues a unicast fused scatter write + atomic increment for all headers in a route via a connection manager.
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
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header);

// clang-format off
/**
 * Fused scatter write + atomic increment (stateful): updates only masked fields, then submits the payload.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | client_interface                                  | Fabric sender interface           | tt_l1_ptr WorkerToFabricEdmSender*                          | True     |
 * | packet_header                                     | Packet header to use              | volatile PACKET_HEADER_TYPE*                               | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header  | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | False  |
 * | packet_size_bytes                                 | Payload size override if masked   | uint16_t                                                   | False    |
 */
// clang-format on
template <
    UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
    uint16_t packet_size_bytes = 0);

// clang-format off
/**
 * Fused scatter write + atomic increment (stateful, route variant): updates only masked fields for all headers.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | connection_manager                                | Routing plane connection manager  | RoutingPlaneConnectionManager&                             | True     |
 * | route_id                                          | Route containing packet headers   | uint8_t                                                    | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | False  |
 * | packet_size_bytes                                 | Payload size override if masked   | uint16_t                                                   | False    |
 */
// clang-format on
template <
    UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
    uint16_t packet_size_bytes = 0);

// clang-format off
/**
 * Fused scatter write + atomic increment (set-state): pre-configures headers for repeated use across the route.
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
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0);

// clang-format off
/**
 * Fused scatter write + atomic increment (set-state): pre-configures headers for repeated use across the route.
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
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header);

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
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header);

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
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr);

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
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr);

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
    CommandHeaderT command_header);

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
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header);

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
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header);

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
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header);

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
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header);

// One call represents one branch of the existing client decomposition: E-only, W-only, N with
// optional E/W teeth, or S with optional E/W teeth. Z is never specified by the client; an express
// reverse tree may select it in addition to, or instead of, the branch's cardinal root output.

// Acquires one shared header and encodes one same-mesh multicast branch, anchored at the manager's
// source chip. Inter-mesh multicast retains its explicit anchor in the lower-level API.

// clang-format off
/**
 * Issues one branch of a client-split multicast. The branch tree is encoded once into one shared
 * header, then that immutable header is submitted through every root output selected by the reverse
 * tree, including an express Z output when needed. The operation is same-mesh and anchored at the
 * manager's source chip.
 *
 * Requires one direction-tagged connection per root output and exactly one header in the route.
 * Missing connections fail-stop rather than delivering to only part of the range.
 *
 * Source-local delivery is not part of this API: the operation bypasses the source RX, and only the
 * Ethernet outputs in the root action are used.
 *
 * Return value: None
 *
 * | Argument                   | Description                             | Type                                       | Required |
 * |----------------------------|-----------------------------------------|--------------------------------------------|----------|
 * | connection_manager         | Routing plane connection manager        | RoutingPlaneConnectionManager&             | True     |
 * | route_id                   | Route holding the shared packet header  | uint8_t                                    | True     |
 * | branch                     | One E, W, N-with-teeth, or S-with-teeth branch | const MeshMcastRange&                | True     |
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header);

// clang-format off
/**
 * Issues an atomic increment for one branch of a client-split multicast. The branch tree and command
 * are populated once in one shared header, then submitted through every root output selected by the
 * reverse tree, including an express Z output when needed. The operation is same-mesh and anchored at
 * the manager's source chip.
 *
 * Requires one direction-tagged connection per root output and exactly one header in the route.
 * Missing connections fail-stop rather than delivering to only part of the range.
 *
 * Source-local delivery is not part of this API: the operation bypasses the source RX, and only the
 * Ethernet outputs in the root action are used.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Routing plane connection manager        | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route holding the shared packet header  | uint8_t                                        | True     |
 * | branch                                | One E, W, N-with-teeth, or S-with-teeth branch | const MeshMcastRange&                    | True     |
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header);

// clang-format off
/**
 * Issues a fused write and atomic increment for one branch of a client-split multicast. The branch
 * tree and command are populated once in one shared header, then submitted through every root output
 * selected by the reverse tree, including an express Z output when needed. The operation is same-mesh
 * and anchored at the manager's source chip.
 *
 * Requires one direction-tagged connection per root output and exactly one header in the route.
 * Missing connections fail-stop rather than delivering to only part of the range.
 *
 * Source-local delivery is not part of this API: the operation bypasses the source RX, and only the
 * Ethernet outputs in the root action are used.
 *
 * Return value: None
 *
 * | Argument                                    | Description                            | Type                                                   | Required |
 * |---------------------------------------------|----------------------------------------|--------------------------------------------------------|----------|
 * | connection_manager                          | Routing plane connection manager       | RoutingPlaneConnectionManager&                         | True     |
 * | route_id                                    | Route holding the shared packet header | uint8_t                                                | True     |
 * | branch                                      | One E, W, N-with-teeth, or S-with-teeth branch | const MeshMcastRange&                            | True     |
 * | src_addr                                    | Source L1 address                      | uint32_t                                               | True     |
 * | size                                        | Payload size in bytes                  | uint32_t                                               | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header                   | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader  | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_source_inject_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header);

// clang-format off
/**
 * Issues a fused two-chunk scatter write and atomic increment for one branch of a client-split
 * multicast. The branch tree and command are populated once in one shared header, then submitted
 * through every root output selected by the reverse tree, including an express Z output when needed.
 * The operation is same-mesh and anchored at the manager's source chip.
 *
 * Requires one direction-tagged connection per root output and exactly one header in the route.
 * Missing connections fail-stop rather than delivering to only part of the range.
 *
 * Source-local delivery is not part of this API: the operation bypasses the source RX, and only the
 * Ethernet outputs in the root action are used.
 *
 * Return value: None
 *
 * | Argument                                          | Description                            | Type                                                          | Required |
 * |---------------------------------------------------|----------------------------------------|---------------------------------------------------------------|----------|
 * | connection_manager                                | Routing plane connection manager       | RoutingPlaneConnectionManager&                                | True     |
 * | route_id                                          | Route holding the shared packet header | uint8_t                                                       | True     |
 * | branch                                            | One E, W, N-with-teeth, or S-with-teeth branch | const MeshMcastRange&                                   | True     |
 * | src_addr                                          | Source L1 address                      | uint32_t                                                      | True     |
 * | size                                              | Payload size in bytes                  | uint32_t                                                      | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter and atomic header       | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader   | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_source_inject_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header);

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
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header);

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
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr);

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
    CommandHeaderT noc_unicast_atomic_inc_command_header = nullptr);

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
    CommandHeaderT command_header = nullptr);

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
    CommandHeaderT command_header = nullptr);

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
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header);

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
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header);

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
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header);

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
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr);

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
    CommandHeaderT noc_unicast_inline_write_command_header = nullptr);

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
    CommandHeaderT command_header);

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
    CommandHeaderT command_header);

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
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header);

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
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header);

// clang-format off
/**
 * Multicast fused scatter write + atomic increment (2 scatter chunks + semaphore inc).
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
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header);

// clang-format off
/**
 * Multicast fused scatter write + atomic increment (route variant): issues for all headers.
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
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header);

// clang-format off
/**
 * Multicast fused scatter write + atomic increment (stateful): updates only masked fields, then submits payload.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | client_interface                                  | Fabric sender interface           | tt_l1_ptr WorkerToFabricEdmSender*                         | True     |
 * | packet_header                                     | Packet header to use              | volatile PACKET_HEADER_TYPE*                               | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | False  |
 * | packet_size_bytes                                 | Payload size override if masked   | uint16_t                                                   | False    |
 */
// clang-format on
template <
    UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
    typename FabricSenderType,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
    uint16_t packet_size_bytes = 0);

// clang-format off
/**
 * Multicast fused scatter write + atomic increment (stateful, route variant): updates only masked fields for all headers.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | connection_manager                                | Routing plane connection manager  | RoutingPlaneConnectionManager&                             | True     |
 * | route_id                                          | Route containing packet headers   | uint8_t                                                    | True     |
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | False  |
 * | packet_size_bytes                                 | Payload size override if masked   | uint16_t                                                   | False    |
 */
// clang-format on
template <
    UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header = nullptr,
    uint16_t packet_size_bytes = 0);

// clang-format off
/**
 * Multicast fused scatter write + atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                       | Description                             | Type                             | Required |
 * |--------------------------------|-----------------------------------------|----------------------------------|----------|
 * | packet_header                  | Packet header to use                    | volatile PACKET_HEADER_TYPE*     | True     |
 * | dst_dev_id                     | Destination device id                   | uint8_t                          | True     |
 * | dst_mesh_id                    | Destination mesh id                     | uint16_t                         | True     |
 * | ranges                         | Multicast hop counts (E/W/N/S)          | const MeshMcastRange&            | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0);

// clang-format off
/**
 * Multicast fused scatter write + atomic increment (set-state): pre-configures headers for repeated use across the route.
 *
 * Return value: None
 *
 * | Argument                       | Description                             | Type                             | Required |
 * |--------------------------------|-----------------------------------------|----------------------------------|----------|
 * | connection_manager             | Routing plane connection manager        | RoutingPlaneConnectionManager&   | True     |
 * | route_id                       | Route whose headers will be updated     | uint8_t                          | True     |
 * | ranges                         | Per-header multicast hop counts (E/W/N/S)| const MeshMcastRange*          | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint16_t packet_size_bytes = 0);

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
    uint32_t offset = 0);

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
    uint32_t offset = 0);

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
    uint32_t offset = 0);

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
    bool flush = true);

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
    bool flush = true);

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
    bool flush = true);

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
    uint32_t offset1 = 0);

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
    uint32_t offset1 = 0);

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
    uint32_t offset1 = 0);

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
    uint32_t offset = 0);

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
    uint32_t offset = 0);

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
    uint32_t offset = 0);

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
    uint32_t offset1 = 0);

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
    uint32_t offset1 = 0);

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
    uint32_t offset1 = 0);

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
    bool flush = true);

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
    bool flush = true);

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
    bool flush = true);

}  // namespace tt::tt_fabric::mesh::experimental

// API IMPL files for new architectures will be included here as they are added
#include "tt_metal/fabric/hw/inc/mesh/tt-1xx/api_impl.h"
