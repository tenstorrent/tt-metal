// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "fabric/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::linear::experimental {

// clang-format off
/**
 * Field update masks for Linear Fabric APIs (compile-time bitfields).
 * These flags are used with the *_set_state and *_with_state helpers to control
 * which fields of the packet header should be updated at submission time.
 */
// clang-format on
// Field update masks for linear fabric APIs (compile-time bitfields)
enum class UnicastWriteUpdateMask : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    PayloadSize = 1u << 1,
};

enum class UnicastInlineWriteUpdateMask : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    Value = 1u << 1,
};

enum class UnicastAtomicIncUpdateMask : uint32_t {
    None = 0,
    DstAddr = 1u << 0,
    Wrap = 1u << 1,
    Val = 1u << 2,
    Flush = 1u << 3,
};

// Scatter write dynamic mask (coarse-grained)
enum class UnicastScatterWriteUpdateMask : uint32_t {
    None = 0,
    DstAddrs = 1u << 0,    // update all noc_address[i]
    ChunkSizes = 1u << 1,  // update all chunk_size[i]
    PayloadSize = 1u << 2,
};

// Fused write+atomic inc dynamic mask
enum class UnicastFusedAtomicIncUpdateMask : uint32_t {
    None = 0,
    WriteDstAddr = 1u << 0,
    SemaphoreAddr = 1u << 1,
    Wrap = 1u << 2,
    Val = 1u << 3,
    Flush = 1u << 4,
    PayloadSize = 1u << 5,
};

// Bitwise helpers for enum class flags
constexpr inline UnicastWriteUpdateMask operator|(UnicastWriteUpdateMask a, UnicastWriteUpdateMask b) {
    return static_cast<UnicastWriteUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastWriteUpdateMask operator&(UnicastWriteUpdateMask a, UnicastWriteUpdateMask b) {
    return static_cast<UnicastWriteUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastWriteUpdateMask mask, UnicastWriteUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastInlineWriteUpdateMask operator|(
    UnicastInlineWriteUpdateMask a, UnicastInlineWriteUpdateMask b) {
    return static_cast<UnicastInlineWriteUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastInlineWriteUpdateMask operator&(
    UnicastInlineWriteUpdateMask a, UnicastInlineWriteUpdateMask b) {
    return static_cast<UnicastInlineWriteUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastInlineWriteUpdateMask mask, UnicastInlineWriteUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastAtomicIncUpdateMask operator|(UnicastAtomicIncUpdateMask a, UnicastAtomicIncUpdateMask b) {
    return static_cast<UnicastAtomicIncUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastAtomicIncUpdateMask operator&(UnicastAtomicIncUpdateMask a, UnicastAtomicIncUpdateMask b) {
    return static_cast<UnicastAtomicIncUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastAtomicIncUpdateMask mask, UnicastAtomicIncUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastScatterWriteUpdateMask operator|(
    UnicastScatterWriteUpdateMask a, UnicastScatterWriteUpdateMask b) {
    return static_cast<UnicastScatterWriteUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastScatterWriteUpdateMask operator&(
    UnicastScatterWriteUpdateMask a, UnicastScatterWriteUpdateMask b) {
    return static_cast<UnicastScatterWriteUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastScatterWriteUpdateMask mask, UnicastScatterWriteUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

constexpr inline UnicastFusedAtomicIncUpdateMask operator|(
    UnicastFusedAtomicIncUpdateMask a, UnicastFusedAtomicIncUpdateMask b) {
    return static_cast<UnicastFusedAtomicIncUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr inline UnicastFusedAtomicIncUpdateMask operator&(
    UnicastFusedAtomicIncUpdateMask a, UnicastFusedAtomicIncUpdateMask b) {
    return static_cast<UnicastFusedAtomicIncUpdateMask>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr inline bool has_flag(UnicastFusedAtomicIncUpdateMask mask, UnicastFusedAtomicIncUpdateMask bit) {
    return static_cast<uint32_t>(mask & bit) != 0u;
}

// ========================
// Common populate helpers
// ========================

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, uint16_t packet_size_bytes, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_inline_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastInlineWriteUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_inline_write.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastInlineWriteUpdateMask::Value)) {
        packet_header->command_fields.unicast_inline_write.value = command_header.value;
    }
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::DstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Wrap)) {
        packet_header->command_fields.unicast_seminc.wrap = command_header.wrap;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Val)) {
        packet_header->command_fields.unicast_seminc.val = command_header.val;
    }
    if constexpr (has_flag(UpdateMask, UnicastAtomicIncUpdateMask::Flush)) {
        packet_header->command_fields.unicast_seminc.flush = command_header.flush;
    }
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_scatter_write_fields(
    volatile PACKET_HEADER_TYPE* packet_header, uint16_t packet_size_bytes, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::DstAddrs)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; i++) {
            auto comps = get_noc_address_components(command_header.noc_address[i]);
            auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
            packet_header->command_fields.unicast_scatter_write.noc_address[i] = noc_addr;
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::ChunkSizes)) {
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; i++) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] = command_header.chunk_size[i];
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
static FORCE_INLINE void populate_unicast_fused_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, uint16_t packet_size_bytes, const CommandHeaderT& command_header) {
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::WriteDstAddr)) {
        auto comps = get_noc_address_components(command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(comps.first.x, comps.first.y, comps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.noc_address = noc_addr;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::SemaphoreAddr)) {
        auto scomps = get_noc_address_components(command_header.semaphore_noc_address);
        auto snoc = safe_get_noc_addr(scomps.first.x, scomps.first.y, scomps.second, edm_to_local_chip_noc);
        packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address = snoc;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Wrap)) {
        packet_header->command_fields.unicast_seminc_fused.wrap = command_header.wrap;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Val)) {
        packet_header->command_fields.unicast_seminc_fused.val = command_header.val;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::Flush)) {
        packet_header->command_fields.unicast_seminc_fused.flush = command_header.flush;
    }
    if constexpr (has_flag(UpdateMask, UnicastFusedAtomicIncUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

static FORCE_INLINE void fabric_set_unicast_route(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t i) {
#if defined(FABRIC_2D)
    const auto& slot = connection_manager.get(i);
#if defined(DYNAMIC_ROUTING_ENABLED)
    fabric_set_unicast_route(
        packet_header, connection_manager.my_chip_id, slot.dst_dev_id, slot.dst_mesh_id, connection_manager.ew_dim);
#else
    fabric_set_unicast_route(slot.dst_dev_id, packet_header);
#endif
#else
    // 1D unicast, nop
#endif
}

static FORCE_INLINE void fabric_set_mcast_route(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t* range,
    uint8_t i) {
#if defined(FABRIC_2D)  // for both DYNAMIC
    // 2D multicast
    const auto& slot = connection_manager.get(i);
    if (range[i] != 0) {
        auto hop = range[i];
#if defined(DYNAMIC_ROUTING_ENABLED)
        hop -= 1;
#endif
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
            default: ASSERT(FALSE);
        }
    }
#else
    // 1D multicast, nop
#endif
}

// clang-format off
/**
 * Opens fabric routing-plane connections for all headers associated with the given route.
 * Reads connection parameters from runtime arguments and initializes headers with routing metadata.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Connection manager to build and open    | RoutingPlaneConnectionManager&                 | True     |
 * | route_id                              | Route identifier used to fetch headers  | uint8_t                                        | True     |
 * | rt_arg_idx                            | Runtime-args cursor (advanced as parsed)| size_t&                                        | True     |
 */
// clang-format on
FORCE_INLINE void open_connections(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint32_t num_connections_to_build,
    size_t& rt_arg_idx) {
    connection_manager = tt::tt_fabric::RoutingPlaneConnectionManager::template build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION>(rt_arg_idx, num_connections_to_build);
}

// clang-format off
/**
 * Closes all connections owned by the provided connection manager.
 *
 * Return value: None
 *
 * | Argument                              | Description                             | Type                                           | Required |
 * |---------------------------------------|-----------------------------------------|------------------------------------------------|----------|
 * | connection_manager                    | Connection manager to be closed         | RoutingPlaneConnectionManager&                 | True     |
 */
// clang-format on
FORCE_INLINE void close_connections(tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager) {
    connection_manager.close();
}

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
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                      | True     |
 */
// clang-format on
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
template <UnicastWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header = {},
    uint16_t packet_size_bytes = 0) {
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
template <UnicastWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header = {},
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
    uint16_t packet_size_bytes = 0) {
    packet_header->to_chip_unicast(num_hops);
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
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, num_hops[i], command_header, packet_size_bytes);
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
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 * | num_hops                              | Unicast hop count                       | uint8_t                                        | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                       | True     |
 */
// clang-format on
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
template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
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
template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <
    UnicastAtomicIncUpdateMask UpdateMask = UnicastAtomicIncUpdateMask::None,
    typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header) {
    packet_header->to_chip_unicast(num_hops);
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
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_unicast_atomic_inc_set_state<UpdateMask>(packet_header, num_hops[i], command_header);
    });
}

// clang-format off
/**
 * Issues a unicast scatter write from local L1 to destination chunks (split into up to two chunks).
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
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
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
 * | num_hops                              | Per-header unicast hop counts            | uint8_t*                                       | True     |
 */
// clang-format on
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
template <UnicastScatterWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header = {},
    uint16_t packet_size_bytes = 0) {
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
template <UnicastScatterWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header = {},
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
    uint16_t packet_size_bytes = 0) {
    packet_header->to_chip_unicast(num_hops);
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
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, num_hops[i], command_header, packet_size_bytes);
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
 * | noc_unicast_inline_write_command_header | Inline write command header | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True |
 * | num_hops                           | Unicast hop count                | uint8_t                                       | True     |
 */
// clang-format on
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
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
 * | num_hops                           | Per-header unicast hop counts       | uint8_t*                                      | True     |
 */
// clang-format on
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
template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
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
template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t                      | True     |
 * | command_header                        | Template command header                 | CommandHeaderT               | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t num_hops, CommandHeaderT command_header) {
    packet_header->to_chip_unicast(num_hops);
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
 * | num_hops                              | Per-header unicast hop counts           | uint8_t*                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
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
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t num_hops) {
    packet_header->to_chip_unicast(num_hops);
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
 * | num_hops                                  | Per-header unicast hop counts           | uint8_t*                                          | True     |
 */
// clang-format on
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
template <UnicastFusedAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header = {},
    uint16_t packet_size_bytes = 0) {
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
template <UnicastFusedAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header = {},
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
    uint16_t packet_size_bytes = 0) {
    packet_header->to_chip_unicast(num_hops);
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
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_unicast_route(connection_manager, packet_header, i);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
            packet_header, num_hops[i], command_header, packet_size_bytes);
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
 * | src_addr                   | Source L1 address                       | uint32_t                                   | True     |
 * | size                       | Payload size in bytes                   | uint32_t                                   | True     |
 * | noc_unicast_command_header | Destination NOC command header          | tt::tt_fabric::NocUnicastCommandHeader     | True     |
 * | start_distance             | Multicast start distance                | uint8_t                                    | True     |
 * | range                      | Multicast range                         | uint8_t                                    | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_write(
            &slot.sender, packet_header, src_addr, size, noc_unicast_command_header, start_distance[i], range[i]);
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
template <UnicastWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header = {},
    uint16_t packet_size_bytes = 0) {
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
template <UnicastWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header = {},
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
    uint16_t packet_size_bytes = 0) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
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
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_write_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header, packet_size_bytes);
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
 * | noc_unicast_atomic_inc_command_header | Atomic increment command header         | tt::tt_fabric::NocUnicastAtomicIncCommandHeader| True     |
 * | start_distance                        | Multicast start distance                | uint8_t                                        | True     |
 * | range                                 | Multicast range                         | uint8_t                                        | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_atomic_inc(
            &slot.sender, packet_header, noc_unicast_atomic_inc_command_header, start_distance[i], range[i]);
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
template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
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
template <UnicastAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
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
    CommandHeaderT command_header = nullptr) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_ATOMIC_INC;
    packet_header->payload_size_bytes = 0;
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
    CommandHeaderT command_header = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_atomic_inc_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header);
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
 * | src_addr                              | Source L1 address                       | uint32_t                                       | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                       | True     |
 * | noc_unicast_scatter_command_header    | Scatter write command header            | tt::tt_fabric::NocUnicastScatterCommandHeader  | True     |
 * | start_distance                        | Multicast start distance                | uint8_t                                        | True     |
 * | range                                 | Multicast range                         | uint8_t                                        | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
template <UnicastScatterWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header = {},
    uint16_t packet_size_bytes = 0) {
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
template <UnicastScatterWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header = {},
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
    uint16_t packet_size_bytes = 0) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_scatter_write_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header, packet_size_bytes);
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
 * | noc_unicast_inline_write_command_header | Inline write command header           | tt::tt_fabric::NocUnicastInlineWriteCommandHeader | True  |
 * | start_distance                        | Multicast start distance                | uint8_t                                        | True     |
 * | range                                 | Multicast range                         | uint8_t                                        | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
    uint8_t* range) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_inline_write(
            &slot.sender, packet_header, noc_unicast_inline_write_command_header, start_distance[i], range[i]);
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
template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
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
template <UnicastInlineWriteUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
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
 * | start_distance                        | Per-header multicast start distance     | uint8_t                                       | True     |
 * | range                                 | Per-header multicast range              | uint8_t                                       | True     |
 * | command_header                        | Template command header                 | CommandHeaderT                                 | False    |
 */
// clang-format on
template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT = std::nullptr_t>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header, uint8_t start_distance, uint8_t range, CommandHeaderT command_header) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
    CommandHeaderT command_header) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_unicast_inline_write_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header);
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
 * | src_addr                              | Source L1 address                       | uint32_t                                         | True     |
 * | size                                  | Payload size in bytes                   | uint32_t                                         | True     |
 * | noc_fused_unicast_atomic_inc_command_header | Fused command header              | tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader | True |
 * | start_distance                        | Multicast start distance                | uint8_t                                          | True     |
 * | range                                 | Multicast range                         | uint8_t                                          | True     |
 */
// clang-format on
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    uint8_t start_distance,
    uint8_t range) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
template <UnicastFusedAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header = {},
    uint16_t packet_size_bytes = 0) {
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
template <UnicastFusedAtomicIncUpdateMask UpdateMask>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header = {},
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
    uint16_t packet_size_bytes = 0) {
    packet_header->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{start_distance, range});
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
    uint16_t packet_size_bytes = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        fabric_set_mcast_route(connection_manager, packet_header, range, i);
        fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<UpdateMask>(
            packet_header, start_distance[i], range[i], command_header, packet_size_bytes);
    });
}

}  // namespace tt::tt_fabric::linear::experimental
