// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Horizon (Quasar 2xx) a2a API implementation.
// Destinations resolve to flat crossbar addresses through fabric_over_noc_routing_l1_info_t::flat_addr; the
// transaction is then a plain NoC write/atomic from the calling core. Included from a2a/api.h only.

#pragma once

#include <cstdint>

#include "api/debug/assert.h"
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::a2a::experimental {

namespace detail {

// noc_address from get_noc_addr under the ATT is REMOTE_WINDOW.compare | (selector << endpoint_shift) | offset,
// where selector is the tile index within the device (row-major over the ATT worker grid).
static constexpr uint32_t HORIZON_ATT_SELECTOR_MASK = 0x3F;
static constexpr uint64_t HORIZON_ATT_OFFSET_MASK = (uint64_t(1) << HORIZON_REMOTE_ENDPOINT_SHIFT) - 1;

// Horizon NoC communication can currently only reach the lower 1 MB of each Tensix core's L1; the NOC2AXI
// crossbar window is 1 MB per tile slot.
static constexpr uint64_t HORIZON_TILE_WINDOW_BYTES = uint64_t(1) << 20;

FORCE_INLINE void assert_in_tile_window(uint64_t noc_address, uint32_t size) {
    ASSERT((noc_address & HORIZON_ATT_OFFSET_MASK) + size <= HORIZON_TILE_WINDOW_BYTES);
}

// Writes go through the crossbar, so NoC ordering alone does not keep a later atomic behind them. A flushed
// atomic waits for every outstanding write from this core to be acked first.
FORCE_INLINE void wait_for_writes_acked_if(bool flush) {
    if (flush) {
        noc_async_write_barrier();
    }
}

FORCE_INLINE void scatter_write(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    const uint64_t* noc_addresses,
    const uint16_t* chunk_sizes,
    uint8_t chunk_count) {
    ASSERT(chunk_count >= NOC_SCATTER_WRITE_MIN_CHUNKS && chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS);
    uint32_t offset = 0;
    for (uint8_t i = 0; i < chunk_count; i++) {
        // The last chunk's size is implicit: whatever remains of the payload.
        const bool is_last = i + 1 == chunk_count;
        ASSERT(offset < size);
        const uint32_t chunk_size = is_last ? size - offset : chunk_sizes[i];
        assert_in_tile_window(noc_addresses[i], chunk_size);
        noc_async_write(src_addr + offset, get_fabric_noc_addr(dst_dev_id, dst_mesh_id, noc_addresses[i]), chunk_size);
        offset += chunk_size;
    }
}

}  // namespace detail

FORCE_INLINE uint64_t get_fabric_noc_addr(uint8_t dst_dev_id, uint16_t dst_mesh_id, uint64_t noc_address) {
    const uint64_t local_offset = noc_address & detail::HORIZON_ATT_OFFSET_MASK;
    const uint32_t selector =
        static_cast<uint32_t>((noc_address >> HORIZON_REMOTE_ENDPOINT_SHIFT) & detail::HORIZON_ATT_SELECTOR_MASK);
    const uint32_t tile_x = selector % HORIZON_ATT_WORKER_GRID_X;
    const uint32_t tile_y = selector / HORIZON_ATT_WORKER_GRID_X;

    ASSERT(dst_mesh_id < MAX_HORIZON_MESHES);
    ASSERT(dst_dev_id < MAX_HORIZON_DEVS_PER_MESH);
    ASSERT(tile_x < MAX_HORIZON_NOC_X);
    ASSERT(tile_y < MAX_HORIZON_NOC_Y);

    const auto* routing_table =
        reinterpret_cast<const tt_l1_ptr fabric_over_noc_routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint64_t base = routing_table->flat_addr[dst_mesh_id][dst_dev_id][tile_x][tile_y];
    // Zero base means self-entry or an unfilled routing table slot.
    ASSERT(base != 0);
    return base + local_offset;
}

// ============================================================================
// Unicast
// ============================================================================

FORCE_INLINE void fabric_unicast_noc_unicast_write(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    const uint64_t noc_address = noc_unicast_command_header.noc_address;
    detail::assert_in_tile_window(noc_address, size);
    noc_async_write(src_addr, get_fabric_noc_addr(dst_dev_id, dst_mesh_id, noc_address), size);
}

FORCE_INLINE void fabric_unicast_noc_unicast_inline_write(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header) {
    const uint64_t noc_address = noc_unicast_inline_write_command_header.noc_address;
    detail::assert_in_tile_window(noc_address, sizeof(uint32_t));
    noc_inline_dw_write(
        get_fabric_noc_addr(dst_dev_id, dst_mesh_id, noc_address), noc_unicast_inline_write_command_header.value);
}

FORCE_INLINE void fabric_unicast_noc_unicast_atomic_inc(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header) {
    const uint64_t noc_address = noc_unicast_atomic_inc_command_header.noc_address;
    detail::assert_in_tile_window(noc_address, sizeof(uint32_t));
    detail::wait_for_writes_acked_if(noc_unicast_atomic_inc_command_header.flush);
    noc_semaphore_inc(
        get_fabric_noc_addr(dst_dev_id, dst_mesh_id, noc_address), noc_unicast_atomic_inc_command_header.val);
}

FORCE_INLINE void fabric_unicast_noc_scatter_write(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header) {
    detail::scatter_write(
        dst_dev_id,
        dst_mesh_id,
        src_addr,
        size,
        noc_unicast_scatter_command_header.noc_address,
        noc_unicast_scatter_command_header.chunk_size,
        noc_unicast_scatter_command_header.chunk_count);
}

FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header) {
    const auto& header = noc_fused_unicast_atomic_inc_command_header;
    fabric_unicast_noc_unicast_write(
        dst_dev_id, dst_mesh_id, src_addr, size, tt::tt_fabric::NocUnicastCommandHeader{header.noc_address});
    fabric_unicast_noc_unicast_atomic_inc(
        dst_dev_id,
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(header.semaphore_noc_address, header.val, header.flush));
}

FORCE_INLINE void fabric_unicast_noc_fused_scatter_write_atomic_inc(
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_fused_scatter_write_atomic_inc_command_header) {
    const auto& header = noc_fused_scatter_write_atomic_inc_command_header;
    detail::scatter_write(
        dst_dev_id,
        dst_mesh_id,
        src_addr,
        size,
        header.noc_address,
        header.chunk_size,
        NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS);
    fabric_unicast_noc_unicast_atomic_inc(
        dst_dev_id,
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(header.semaphore_noc_address, header.val, header.flush));
}

// ============================================================================
// Multicast
//
// The DstDevIds overloads hold the implementation. The variadic overloads pack their ids into a local array
// and forward; with a compile-time count the loop unrolls the same as a hand-written sequence.
// Fused ops issue every write first, ack them once (if flush), then issue every increment, rather than
// paying one ack round-trip per device.
// ============================================================================

FORCE_INLINE void fabric_multicast_noc_unicast_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    DstDevIds dst_dev_ids) {
    ASSERT(dst_dev_ids.size() > 0);
    for (uint8_t dst_dev_id : dst_dev_ids) {
        fabric_unicast_noc_unicast_write(dst_dev_id, dst_mesh_id, src_addr, size, noc_unicast_command_header);
    }
}

template <typename... DevIdTs, typename>
FORCE_INLINE void fabric_multicast_noc_unicast_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    DevIdTs... dst_dev_ids) {
    const uint8_t ids[] = {static_cast<uint8_t>(dst_dev_ids)...};
    fabric_multicast_noc_unicast_write(dst_mesh_id, src_addr, size, noc_unicast_command_header, DstDevIds(ids));
}

FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    DstDevIds dst_dev_ids) {
    ASSERT(dst_dev_ids.size() > 0);
    for (uint8_t dst_dev_id : dst_dev_ids) {
        fabric_unicast_noc_unicast_inline_write(dst_dev_id, dst_mesh_id, noc_unicast_inline_write_command_header);
    }
}

template <typename... DevIdTs, typename>
FORCE_INLINE void fabric_multicast_noc_unicast_inline_write(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastInlineWriteCommandHeader noc_unicast_inline_write_command_header,
    DevIdTs... dst_dev_ids) {
    const uint8_t ids[] = {static_cast<uint8_t>(dst_dev_ids)...};
    fabric_multicast_noc_unicast_inline_write(dst_mesh_id, noc_unicast_inline_write_command_header, DstDevIds(ids));
}

FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    DstDevIds dst_dev_ids) {
    ASSERT(dst_dev_ids.size() > 0);
    const uint64_t noc_address = noc_unicast_atomic_inc_command_header.noc_address;
    detail::assert_in_tile_window(noc_address, sizeof(uint32_t));
    detail::wait_for_writes_acked_if(noc_unicast_atomic_inc_command_header.flush);
    for (uint8_t dst_dev_id : dst_dev_ids) {
        noc_semaphore_inc(
            get_fabric_noc_addr(dst_dev_id, dst_mesh_id, noc_address), noc_unicast_atomic_inc_command_header.val);
    }
}

template <typename... DevIdTs, typename>
FORCE_INLINE void fabric_multicast_noc_unicast_atomic_inc(
    uint16_t dst_mesh_id,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader noc_unicast_atomic_inc_command_header,
    DevIdTs... dst_dev_ids) {
    const uint8_t ids[] = {static_cast<uint8_t>(dst_dev_ids)...};
    fabric_multicast_noc_unicast_atomic_inc(dst_mesh_id, noc_unicast_atomic_inc_command_header, DstDevIds(ids));
}

FORCE_INLINE void fabric_multicast_noc_scatter_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    DstDevIds dst_dev_ids) {
    ASSERT(dst_dev_ids.size() > 0);
    for (uint8_t dst_dev_id : dst_dev_ids) {
        fabric_unicast_noc_scatter_write(dst_dev_id, dst_mesh_id, src_addr, size, noc_unicast_scatter_command_header);
    }
}

template <typename... DevIdTs, typename>
FORCE_INLINE void fabric_multicast_noc_scatter_write(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterCommandHeader noc_unicast_scatter_command_header,
    DevIdTs... dst_dev_ids) {
    const uint8_t ids[] = {static_cast<uint8_t>(dst_dev_ids)...};
    fabric_multicast_noc_scatter_write(dst_mesh_id, src_addr, size, noc_unicast_scatter_command_header, DstDevIds(ids));
}

FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    DstDevIds dst_dev_ids) {
    const auto& header = noc_fused_unicast_atomic_inc_command_header;
    fabric_multicast_noc_unicast_write(
        dst_mesh_id, src_addr, size, tt::tt_fabric::NocUnicastCommandHeader{header.noc_address}, dst_dev_ids);
    fabric_multicast_noc_unicast_atomic_inc(
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(header.semaphore_noc_address, header.val, header.flush),
        dst_dev_ids);
}

template <typename... DevIdTs, typename>
FORCE_INLINE void fabric_multicast_noc_fused_unicast_with_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header,
    DevIdTs... dst_dev_ids) {
    const uint8_t ids[] = {static_cast<uint8_t>(dst_dev_ids)...};
    fabric_multicast_noc_fused_unicast_with_atomic_inc(
        dst_mesh_id, src_addr, size, noc_fused_unicast_atomic_inc_command_header, DstDevIds(ids));
}

FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_fused_scatter_write_atomic_inc_command_header,
    DstDevIds dst_dev_ids) {
    const auto& header = noc_fused_scatter_write_atomic_inc_command_header;
    ASSERT(dst_dev_ids.size() > 0);
    for (uint8_t dst_dev_id : dst_dev_ids) {
        detail::scatter_write(
            dst_dev_id,
            dst_mesh_id,
            src_addr,
            size,
            header.noc_address,
            header.chunk_size,
            NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS);
    }
    fabric_multicast_noc_unicast_atomic_inc(
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(header.semaphore_noc_address, header.val, header.flush),
        dst_dev_ids);
}

template <typename... DevIdTs, typename>
FORCE_INLINE void fabric_multicast_noc_fused_scatter_write_atomic_inc(
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader noc_fused_scatter_write_atomic_inc_command_header,
    DevIdTs... dst_dev_ids) {
    const uint8_t ids[] = {static_cast<uint8_t>(dst_dev_ids)...};
    fabric_multicast_noc_fused_scatter_write_atomic_inc(
        dst_mesh_id, src_addr, size, noc_fused_scatter_write_atomic_inc_command_header, DstDevIds(ids));
}

}  // namespace tt::tt_fabric::a2a::experimental
