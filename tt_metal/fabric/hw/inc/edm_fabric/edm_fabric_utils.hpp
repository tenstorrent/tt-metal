// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

static constexpr uint8_t edm_to_local_chip_noc = 1;

// When UDM_MODE and dynamic NOC are both enabled, all fabric transactions must use the same NOC
// to preserve packet ordering when multiple data movement kernels write to fabric.
// Using Noc0 for now until we get perf benchmark to indicate otherwise.
static constexpr uint8_t udm_dynamic_mode_worker_noc = 0;

// Helper to get the appropriate NOC for fabric worker operations.
// When UDM_MODE and dynamic NOC are both enabled, all fabric transactions must use the same NOC
// to preserve packet ordering when multiple data movement kernels write to fabric.
FORCE_INLINE constexpr uint8_t get_fabric_worker_noc() {
#if defined(UDM_MODE)
    if constexpr (NOC_MODE == DM_DYNAMIC_NOC) {
        return udm_dynamic_mode_worker_noc;
    } else {
        return noc_index;
    }
#else
    return noc_index;
#endif
}

enum EDM_IO_BLOCKING_MODE { FLUSH_BLOCKING, BLOCKING, NON_BLOCKING };

template <bool stateful_api>
FORCE_INLINE void send_chunk_from_address_with_trid(
    const uint32_t& local_l1_address,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint32_t remote_l1_write_addr_h,
    uint32_t remote_l1_write_addr_l,
    uint8_t trid,
    uint8_t noc,
    uint8_t cmd_buf) {
#ifdef ARCH_BLACKHOLE
    // forced true
    constexpr bool update_counter = true;
#else
    constexpr bool update_counter = false;
#endif
    if constexpr (stateful_api) {
        noc_async_write_one_packet_with_trid_with_state<update_counter, true>(
            local_l1_address, remote_l1_write_addr_l, page_size * num_pages, trid, cmd_buf, noc);
    } else {
        noc_async_write_one_packet_with_trid<update_counter, true>(
            local_l1_address,
            get_noc_addr_helper(remote_l1_write_addr_h, remote_l1_write_addr_l),
            page_size * num_pages,
            trid,
            cmd_buf,
            noc);
    }
}

template <EDM_IO_BLOCKING_MODE blocking_mode = EDM_IO_BLOCKING_MODE::BLOCKING>
FORCE_INLINE void send_chunk_from_address(
    const uint32_t& local_l1_address,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint64_t remote_l1_write_addr) {
    const uint8_t noc = get_fabric_worker_noc();
    noc_async_write(local_l1_address, remote_l1_write_addr, page_size * num_pages, noc);
    if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING) {
        noc_async_writes_flushed(noc);
    } else if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::BLOCKING) {
        noc_async_write_barrier(noc);
    }
}

template <EDM_IO_BLOCKING_MODE blocking_mode = EDM_IO_BLOCKING_MODE::BLOCKING>
FORCE_INLINE void send_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    const uint8_t noc = get_fabric_worker_noc();
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages, noc);
    if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING) {
        noc_async_writes_flushed(noc);
        cb_pop_front(cb_id, num_pages);
    } else if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::BLOCKING) {
        noc_async_write_barrier(noc);
        cb_pop_front(cb_id, num_pages);
    }
}

}  // namespace tt::tt_fabric
