// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

FORCE_INLINE void fetch_chunk(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(remote_l1_read_addr, l1_write_addr, page_size);
        remote_l1_read_addr += page_size;
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}

FORCE_INLINE void send_chunk(const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, remote_l1_write_addr, page_size);
        remote_l1_write_addr += page_size;
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}

template<typename AddrGen>
FORCE_INLINE void write_and_send_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t num_cols, const uint32_t num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, remote_l1_write_addr, page_size);
        remote_l1_write_addr += page_size;
        #ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
        #endif
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}

template<typename AddrGen>
FORCE_INLINE void write_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        #ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
        #endif
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}

template<typename AddrGen>
FORCE_INLINE void read_chunk(uint32_t& input_page_idx, const uint32_t& cb_id, const AddrGen& s, const uint32_t& num_pages, const uint32_t& page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    for (; input_page_idx < end_read_idx; ++input_page_idx) {
        cb_reserve_back(cb_id, 1);
        uint32_t local_l1_read_addr = get_write_ptr(cb_id);
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        #endif
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}

template<typename AddrGen>
FORCE_INLINE void read_chunk(uint32_t& input_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t &cb_id, const AddrGen& s, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
     for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id, 1);
        uint32_t local_l1_read_addr = get_write_ptr(cb_id);
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        input_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            input_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        input_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            input_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                input_page_idx += row_offset;
            }
        }
        #endif
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
