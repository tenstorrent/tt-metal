// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"

namespace tt::tt_fabric {

static constexpr uint8_t edm_to_local_chip_noc = 1;

enum EDM_IO_BLOCKING_MODE { FLUSH_BLOCKING, BLOCKING, NON_BLOCKING };

template <EDM_IO_BLOCKING_MODE blocking_mode = EDM_IO_BLOCKING_MODE::BLOCKING, bool stateful_api>
FORCE_INLINE void send_chunk_from_address_with_trid(
    const uint32_t& local_l1_address,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint32_t remote_l1_write_addr_h,
    uint32_t remote_l1_write_addr_l,
    uint8_t trid,
    uint8_t noc,
    uint8_t cmd_buf) {
    if constexpr (stateful_api) {
        noc_async_write_one_packet_with_trid_with_state<false, true>(
            local_l1_address, remote_l1_write_addr_l, page_size * num_pages, trid, cmd_buf, noc);
    } else {
        noc_async_write_one_packet_with_trid<false, true>(
            local_l1_address,
            get_noc_addr_helper(remote_l1_write_addr_h, remote_l1_write_addr_l),
            page_size * num_pages,
            trid,
            cmd_buf,
            noc);
    }
    // TODO: this barrier will no longer be functional since we are not incrementing noc counters, remove
    if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING) {
        noc_async_writes_flushed();
    } else if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::BLOCKING) {
        noc_async_write_barrier();
    }
}

template <EDM_IO_BLOCKING_MODE blocking_mode = EDM_IO_BLOCKING_MODE::BLOCKING>
FORCE_INLINE void send_chunk_from_address(
    const uint32_t& local_l1_address,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint64_t remote_l1_write_addr) {
    noc_async_write(local_l1_address, remote_l1_write_addr, page_size * num_pages);
    if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING) {
        noc_async_writes_flushed();
    } else if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::BLOCKING) {
        noc_async_write_barrier();
    }
}

template <EDM_IO_BLOCKING_MODE blocking_mode = EDM_IO_BLOCKING_MODE::BLOCKING>
FORCE_INLINE void send_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING) {
        noc_async_writes_flushed();
        cb_pop_front(cb_id, num_pages);
    } else if constexpr (blocking_mode == EDM_IO_BLOCKING_MODE::BLOCKING) {
        noc_async_write_barrier();
        cb_pop_front(cb_id, num_pages);
    }
}

}  // namespace tt::tt_fabric
