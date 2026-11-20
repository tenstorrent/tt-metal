// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Horizon (Quasar 2xx) mesh API implementation.
// Uses Horizon crossbar routing via fabric_flat_address_routing_l1_info_t::flat_addr + noc_async_write.
// Implemented: fabric_unicast_noc_unicast_write (stateless + ConnectionManager overload),
//              fabric_unicast_noc_unicast_write_{set,with}_state,
//              fabric_unicast_noc_scatter_write (stateless),
//              fabric_unicast_noc_scatter_write_{set,with}_state.
// All other functions are TODO stubs.

#pragma once

#ifndef NOC_ATT_ENABLED
#error "quasar_horizon/api_impl.h requires ATT-enabled builds (TT_METAL_NOC_ATT must be set)"
#endif

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

// ATT endpoint parameters for Horizon x2 (values from horizon_2x3_att_config.h).
// Defined here as constexpr to avoid pulling in the config header into kernel code.
static constexpr uint32_t HORIZON_ATT_WORKER_GRID_X = 2;
static constexpr uint32_t HORIZON_ATT_ENDPOINT_SHIFT = 26;
static constexpr uint64_t HORIZON_ATT_OFFSET_MASK = (uint64_t(1) << HORIZON_ATT_ENDPOINT_SHIFT) - 1;

namespace horizon_detail {
// Decodes a Horizon ATT-encoded noc_address (bits [25:0] = local offset, bits [31:26] = ATT selector)
// and looks up the flat crossbar address for the given destination device and mesh.
static FORCE_INLINE uint64_t
resolve_destination_flat_address(uint64_t att_noc_addr, uint8_t dst_dev_id, uint16_t dst_mesh_id) {
    const uint64_t local_offset = att_noc_addr & HORIZON_ATT_OFFSET_MASK;
    const uint32_t selector = static_cast<uint32_t>((att_noc_addr >> HORIZON_ATT_ENDPOINT_SHIFT) & 0x3F);
    const uint32_t tile_x = selector % HORIZON_ATT_WORKER_GRID_X;
    const uint32_t tile_y = selector / HORIZON_ATT_WORKER_GRID_X;
    ASSERT(dst_mesh_id < tt::tt_fabric::MAX_HORIZON_MESHES);
    ASSERT(dst_dev_id < tt::tt_fabric::MAX_HORIZON_DEVS_PER_MESH);
    ASSERT(tile_x < HORIZON_ATT_WORKER_GRID_X);
    ASSERT(tile_y < tt::tt_fabric::MAX_HORIZON_NOC_Y);
    const auto* rt =
        reinterpret_cast<const tt_l1_ptr tt::tt_fabric::fabric_flat_address_routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint64_t base = rt->flat_addr[dst_mesh_id][dst_dev_id][tile_x][tile_y];
    ASSERT(base != 0);  // unfilled routing table slot
    return base + local_offset;
}
}  // namespace horizon_detail

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
    [[maybe_unused]] auto* _ci = client_interface;
    [[maybe_unused]] auto* _ph = packet_header;

    const uint64_t noc_addr = noc_unicast_command_header.noc_address;
    const uint64_t local_offset = noc_addr & HORIZON_ATT_OFFSET_MASK;
    // Horizon NOC2AXI crossbar window is 1 MB per tile slot.
    ASSERT(local_offset + size <= (1u << 20));
    const uint64_t remote_addr = horizon_detail::resolve_destination_flat_address(noc_addr, dst_dev_id, dst_mesh_id);
    noc_async_write(src_addr, remote_addr, size);
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
#ifdef FABRIC_2D
    auto& slot = connection_manager.get(route_id);
    fabric_unicast_noc_unicast_write<decltype(slot.sender), SetRoute>(
        &slot.sender,
        (volatile PACKET_HEADER_TYPE*)nullptr,
        static_cast<uint8_t>(slot.dst_dev_id),
        slot.dst_mesh_id,
        src_addr,
        size,
        noc_unicast_command_header);
#else
    static_assert(!SetRoute, "Horizon ConnectionManager unicast write requires FABRIC_2D");
#endif
}

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] auto* _ci = client_interface;
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::DstAddr)) {
        packet_header->command_fields.unicast_write.noc_address = noc_unicast_command_header.noc_address;
    }
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
    const uint64_t flat = horizon_detail::resolve_destination_flat_address(
        packet_header->command_fields.unicast_write.noc_address,
        static_cast<uint8_t>(packet_header->dst_start_chip_id),
        packet_header->dst_start_mesh_id);
    noc_async_write(src_addr, flat, packet_header->payload_size_bytes);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    // HybridMeshPacketHeader::dst_start_chip_id/dst_start_mesh_id repurposed as
    // dst_dev_id/dst_mesh_id for Horizon crossbar routing.
    packet_header->dst_start_chip_id = dst_dev_id;
    packet_header->dst_start_mesh_id = dst_mesh_id;
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::DstAddr)) {
        packet_header->command_fields.unicast_write.noc_address = command_header.noc_address;
    }
    if constexpr (has_flag(UpdateMask, UnicastWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    [[maybe_unused]] auto* _ci = client_interface;
    [[maybe_unused]] auto* _ph = packet_header;
    uint32_t src_offset = 0;
    const uint8_t chunk_count = noc_unicast_scatter_command_header.chunk_count;
    for (uint8_t i = 0; i < chunk_count; ++i) {
        const uint16_t chunk_size = (i < static_cast<uint8_t>(chunk_count - 1))
                                        ? noc_unicast_scatter_command_header.chunk_size[i]
                                        : static_cast<uint16_t>(size - src_offset);
        const uint64_t flat = horizon_detail::resolve_destination_flat_address(
            noc_unicast_scatter_command_header.noc_address[i], dst_dev_id, dst_mesh_id);
        noc_async_write(src_addr + src_offset, flat, chunk_size);
        src_offset += chunk_size;
    }
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    [[maybe_unused]] auto* _ci = client_interface;
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::DstAddrs)) {
        packet_header->command_fields.unicast_scatter_write.chunk_count =
            noc_unicast_scatter_command_header.chunk_count;
        for (uint8_t i = 0; i < noc_unicast_scatter_command_header.chunk_count; ++i) {
            packet_header->command_fields.unicast_scatter_write.noc_address[i] =
                noc_unicast_scatter_command_header.noc_address[i];
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::ChunkSizes)) {
        for (uint8_t i = 0; i < static_cast<uint8_t>(noc_unicast_scatter_command_header.chunk_count - 1); ++i) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] =
                noc_unicast_scatter_command_header.chunk_size[i];
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
    const uint8_t n = packet_header->command_fields.unicast_scatter_write.chunk_count;
    const uint8_t dev_id = static_cast<uint8_t>(packet_header->dst_start_chip_id);
    const uint16_t mesh_id = packet_header->dst_start_mesh_id;
    const uint16_t total = packet_header->payload_size_bytes;
    uint32_t src_offset = 0;
    for (uint8_t i = 0; i < n; ++i) {
        const uint16_t chunk_size = (i < static_cast<uint8_t>(n - 1))
                                        ? packet_header->command_fields.unicast_scatter_write.chunk_size[i]
                                        : static_cast<uint16_t>(total - src_offset);
        const uint64_t flat = horizon_detail::resolve_destination_flat_address(
            packet_header->command_fields.unicast_scatter_write.noc_address[i], dev_id, mesh_id);
        noc_async_write(src_addr + src_offset, flat, chunk_size);
        src_offset += chunk_size;
    }
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    // HybridMeshPacketHeader::dst_start_chip_id/dst_start_mesh_id repurposed as
    // dst_dev_id/dst_mesh_id for Horizon crossbar routing.
    packet_header->dst_start_chip_id = dst_dev_id;
    packet_header->dst_start_mesh_id = dst_mesh_id;
    packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::DstAddrs)) {
        packet_header->command_fields.unicast_scatter_write.chunk_count = command_header.chunk_count;
        for (uint8_t i = 0; i < command_header.chunk_count; ++i) {
            packet_header->command_fields.unicast_scatter_write.noc_address[i] = command_header.noc_address[i];
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::ChunkSizes)) {
        for (uint8_t i = 0; i < static_cast<uint8_t>(command_header.chunk_count - 1); ++i) {
            packet_header->command_fields.unicast_scatter_write.chunk_size[i] = command_header.chunk_size[i];
        }
    }
    if constexpr (has_flag(UpdateMask, UnicastScatterWriteUpdateMask::PayloadSize)) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, uint8_t route_id, CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <bool SetRoute>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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

FORCE_INLINE uint8_t multicast_branch_allowed_root_outputs(const MeshMcastRange& branch) {
    if (branch.n != 0) {
        return Routing2DCodec::ACTION_NORTH | Routing2DCodec::ACTION_Z;
    }
    if (branch.s != 0) {
        return Routing2DCodec::ACTION_SOUTH | Routing2DCodec::ACTION_Z;
    }
    return branch.e != 0 ? Routing2DCodec::ACTION_EAST : Routing2DCodec::ACTION_WEST;
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
    [[maybe_unused]] uint32_t src_addr,
    [[maybe_unused]] uint32_t size) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void submit_multicast_branch_fanout_with_payload(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    const PreparedMulticastBranchFanout& fanout,
    uint32_t src_addr,
    uint32_t size) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void submit_multicast_branch_fanout_header_only(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager, const PreparedMulticastBranchFanout& fanout) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

}  // namespace mcast_detail

FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_source_inject_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange& branch,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}
#endif

template <UnicastWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastScatterWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <typename FabricSenderType>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    CommandHeaderT noc_unicast_inline_write_command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastInlineWriteUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_unicast_scatter_atomic_inc_fused_command_header) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_unicast_scatter_atomic_inc_fused_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedScatterWriteAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    CommandHeaderT noc_fused_unicast_atomic_inc_command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const MeshMcastRange& ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    CommandHeaderT command_header,
    uint16_t packet_size_bytes) {
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
}

template <UnicastWriteUpdateMask UpdateMask, typename AddrGenType, typename>
FORCE_INLINE void fabric_unicast_noc_unicast_write_set_state(
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint32_t offset) {
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #61516) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
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
    /* TODO: To be implemented (tracked in issue #63264) */
    ASSERT(false);
}

}  // namespace tt::tt_fabric::mesh::experimental
