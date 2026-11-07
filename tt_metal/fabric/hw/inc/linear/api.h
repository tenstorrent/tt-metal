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

static FORCE_INLINE void fabric_set_unicast_route(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t i);

static FORCE_INLINE void fabric_set_mcast_route(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t* range,
    uint8_t i);

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
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_command_header            | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader        | True     |
 * | num_hops                              | Unicast hop count                       | uint8_t                                       | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops);

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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                      | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* num_hops);

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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t                                        | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                       | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t num_hops,
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                       | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
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
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 * | num_hops                              | Unicast hop count                       | uint8_t                                        | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t num_hops);

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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                       | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t* num_hops);

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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header);

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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header);

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
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 * | num_hops                              | Unicast hop count                       | uint8_t                                        | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t num_hops);

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
 * | num_hops                              | Per-header unicast hop counts            | uint8_t*                                       | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t* num_hops);

// clang-format off
/**
 * Issues a unicast fused scatter write + atomic increment (2 scatter chunks + semaphore inc).
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
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t num_hops);

// clang-format off
/**
 * Issues a unicast fused scatter write + atomic increment for all headers in a route via a connection manager.
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
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t* num_hops);

// clang-format off
/**
 * Fused scatter write + atomic increment (stateful): updates only masked fields, then submits the payload.
 *
 * Return value: None
 *
 * | Argument                                          | Description                       | Type                                                       | Required |
 * |---------------------------------------------------|-----------------------------------|------------------------------------------------------------|----------|
 * | client_interface                                  | Fabric sender interface           | tt_l1_ptr WorkerToFabricEdmSender*                          | True     |
 * | packet_header                                     | Packet header to use              | volatile PACKET_HEADER_TYPE*                                | True     |
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
 * | Argument                       | Description                             | Type                                                       | Required |
 * |--------------------------------|-----------------------------------------|------------------------------------------------------------|----------|
 * | packet_header                  | Packet header to use                    | volatile PACKET_HEADER_TYPE*                               | True     |
 * | num_hops                       | Per-header unicast hop counts           | uint8_t                                                    | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                                             | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                                                   | False    |
 */
// clang-format on
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t num_hops,
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
 * | num_hops                       | Per-header unicast hop counts           | uint8_t*                         | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t                      | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                      | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t num_hops,
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                       | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
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
 * | noc_unicast_inline_write_command_header | Inline write command header | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True |
 * | num_hops                           | Unicast hop count                | uint8_t                                       | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t num_hops);

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
 * | num_hops                           | Per-header unicast hop counts       | uint8_t*                                      | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t* num_hops);

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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t                      | True     |
 * | command_header                        | Template command header                 | CommandHeaderT               | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header);

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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
    CommandHeaderT command_header);

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
 * | src_addr                              | Source L1 address                       | uint32_t                                          | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                          | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header              | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True  |
 * | num_hops                              | Unicast hop count                       | uint8_t                                           | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t num_hops);

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
 * | num_hops                                  | Per-header unicast hop counts           | uint8_t*                                          | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t* num_hops);

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
 * | num_hops                       | Per-header unicast hop counts           | uint8_t                          | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t num_hops,
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
 * | num_hops                       | Per-header unicast hop counts           | uint8_t*                         | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* num_hops,
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
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 * | start_distance             | Multicast start distance                | uint8_t                                    | True     |
 * | range                      | Multicast range                         | uint8_t                                    | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t start_distance,
    uint8_t range);

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
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 * | start_distance             | Per-header multicast start distance     | uint8_t*                                   | True     |
 * | range                      | Per-header multicast range              | uint8_t*                                   | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t* start_distance,
    uint8_t* range);

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
 * | start_distance                        | Per-header multicast start distance     | uint8_t                              | True     |
 * | range                                 | Per-header multicast range              | uint8_t                              | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                       | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                             | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t start_distance,
    uint8_t range,
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
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                             | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                             | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                       | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                             | False    |
 */
// clang-format on
template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
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
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 * | start_distance                        | Multicast start distance                | uint8_t                                        | True     |
 * | range                                 | Multicast range                         | uint8_t                                        | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range);

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
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                                       | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                                       | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t* start_distance,
    uint8_t* range);

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
 * | start_distance                        | Per-header multicast start distance     | uint8_t                                       | True     |
 * | range                                 | Per-header multicast range              | uint8_t                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t start_distance,
    uint8_t range,
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
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                                      | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                                      | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
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
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 * | start_distance                        | Multicast start distance                | uint8_t                                        | True     |
 * | range                                 | Multicast range                         | uint8_t                                        | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t start_distance,
    uint8_t range);

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
 * | src_addr                              | Source L1 address                       | uint32_t                                      | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                      | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader | True     |
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                                      | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                                      | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t* start_distance,
    uint8_t* range);

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
 * | start_distance                        | Per-header multicast start distance     | uint8_t                                       | True     |
 * | range                                 | Per-header multicast range              | uint8_t                                       | True     |
 * | command_header                       | Template command header                  | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t start_distance,
    uint8_t range,
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
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                                      | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                                      | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
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
 * | noc_unicast_inline_write_command_header | Inline write command header           | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True  |
 * | start_distance                        | Multicast start distance                | uint8_t                                        | True     |
 * | range                                 | Multicast range                         | uint8_t                                        | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t start_distance,
    uint8_t range);

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
 * | noc_unicast_inline_write_command_header | Inline write command header        | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True  |
 * | start_distance                        | Per-header multicast start distance  | uint8_t*                                       | True     |
 * | range                                 | Per-header multicast range           | uint8_t*                                       | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t* start_distance,
    uint8_t* range);

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
 * | start_distance                        | Per-header multicast start distance     | uint8_t                                       | True     |
 * | range                                 | Per-header multicast range              | uint8_t                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t start_distance, uint8_t range, CommandHeaderT command_header);

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
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                                       | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
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
 * | src_addr                              | Source L1 address                       | uint32_t                                         | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                         | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header              | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 * | start_distance                        | Multicast start distance                | uint8_t                                          | True     |
 * | range                                 | Multicast range                         | uint8_t                                          | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range);

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
 * | src_addr                              | Source L1 address                    | uint32_t                                         | True     |
 * | size                                  | Payload size in bytes                | uint32_t                                         | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header           | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 * | start_distance                        | Per-header multicast start distance  | uint8_t*                                         | True     |
 * | range                                 | Per-header multicast range           | uint8_t*                                         | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t* start_distance,
    uint8_t* range);

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
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes             | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 * | start_distance                                    | Multicast start distance          | uint8_t                                                    | True     |
 * | range                                             | Multicast range                   | uint8_t                                                    | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t start_distance,
    uint8_t range);

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
 * | src_addr                                          | Source L1 address                 | uint32_t                                                   | True     |
 * | size                                              | Payload size in bytes             | uint32_t                                                   | True     |
 * | noc_unicast_scatter_atomic_inc_fused_command_header | Fused scatter+atomic inc header | tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader | True   |
 * | start_distance                                    | Per-header multicast start distance | uint8_t*                                                | True     |
 * | range                                             | Per-header multicast range       | uint8_t*                                                   | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header,
    uint8_t* start_distance,
    uint8_t* range);

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
 * | start_distance                 | Per-header multicast start distance     | uint8_t                          | True     |
 * | range                          | Per-header multicast range              | uint8_t                          | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t start_distance,
    uint8_t range,
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
 * | start_distance                  | Per-header multicast start distance     | uint8_t*                         | True     |
 * | range                          | Per-header multicast range              | uint8_t*                         | True     |
 * | command_header                 | Template fused command header           | CommandHeaderT                   | False    |
 * | packet_size_bytes              | Payload size override if masked         | uint16_t                         | False    |
 */
// clang-format on
template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
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
 * | start_distance                        | Per-header multicast start distance     | uint8_t                                       | True     |
 * | range                                 | Per-header multicast range              | uint8_t                                       | True     |
 * | command_header                        | Template fused command header           | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t start_distance,
    uint8_t range,
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
 * | start_distance                        | Per-header multicast start distance     | uint8_t*                                      | True     |
 * | range                                 | Per-header multicast range              | uint8_t*                                      | True     |
 * | command_header                        | Template fused command header           | CommandHeaderT                                | False    |
 * | packet_size_bytes                     | Payload size override if masked         | uint16_t                                      | False    |
 */
// clang-format on
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint8_t* start_distance,
    uint8_t* range,
    CommandHeaderT command_header = nullptr,
    uint16_t packet_size_bytes = 0);

// clang-format off
/**
 * Sparse multicast unicast write: issues a unicast write with sparse multicast routing metadata.
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
FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint16_t hops);

// clang-format off
/**
 * Sparse multicast unicast write (route variant): issues writes for all headers in the route using sparse multicast routing metadata.
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
FORCE_INLINE void fabric_sparse_multicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint16_t* hops);

// clang-format off
/**
 * Sparse multicast delivering a single payload to a flat, hop-ordered list of destination addresses,
 * grouped per writing chip by sparse_mcast_command_header.counts[] (num_chips chips, sum == num_dests):
 * chip c receives its counts[c] pages. Unlike scatter (many addresses, one chip), addresses here span
 * the non-contiguous colinear chips selected by the hop bitmask, and one chip may take multiple pages.
 * The caller must set noc_address[0..num_dests), counts[0..num_chips), num_dests and num_chips. 1D only.
 *
 * Return value: None
 *
 * | Argument                   | Description                                      | Type                                          | Required |
 * |----------------------------|--------------------------------------------------|-----------------------------------------------|----------|
 * | client_interface           | Fabric sender interface                          | tt_l1_ptr FabricSenderType*                   | True     |
 * | packet_header              | Packet header to use                             | volatile PACKET_HEADER_TYPE*                  | True     |
 * | src_addr                   | Source L1 address                                | uint32_t                                      | True     |
 * | size                       | Payload size in bytes                            | uint32_t                                      | True     |
 * | sparse_mcast_command_header| Flat hop-ordered addresses + per-chip counts     | tt::tt_fabric::NocSparseMulticastWriteCommandHeader | True |
 * | hops                       | Sparse multicast hop bitmask                     | uint16_t                                      | True     |
 */
// clang-format on
template <typename FabricSenderType>
FORCE_INLINE void fabric_sparse_multicast_noc_scatter_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocSparseMulticastWriteCommandHeader sparse_mcast_command_header,
    uint16_t hops);

}  // namespace tt::tt_fabric::linear::experimental

// API IMPL files for new architectures will be included here as they are added
#include "tt_metal/fabric/hw/inc/linear/tt-1xx/api_impl.h"
