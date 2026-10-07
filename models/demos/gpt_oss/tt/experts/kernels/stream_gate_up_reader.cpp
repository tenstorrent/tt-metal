// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert gate|up weight streamer (NCRISC, NOC0) for the gpt-oss decode MoE (experts/stream.py).
//
// Each worker core streams the weight columns it owns from one DRAM bank. Per bank the weights are stored
// column-major (all K tiles of a column contiguous), grouped per expert and per reader core, so a routed
// expert's columns for this core are one contiguous byte range: expert e starts at e * expert_stride +
// reader_offset. The K dimension carries one extra tile holding the bias (see stream.py).
// Reads use transaction ids so up to two columns are in flight while compute consumes the previous one.
//
// runtime args: [w_addr, idx_addr, bank_id, vc, reader_offset_bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t reader_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t num_sel = get_compile_time_arg_val(4);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t idx_page_bytes = get_compile_time_arg_val(8);
    constexpr auto idx_args = TensorAccessorArgs<9>();

    constexpr uint32_t col_bytes = kt * tile_bytes;
    constexpr uint32_t pages_per_col = col_bytes / page_bytes;
    static_assert(pages_per_col * page_bytes == col_bytes, "page size must divide the column");

    // Routed expert ids (first num_sel uint16 entries of the indices stick).
    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, idx_page_bytes);
    noc_async_read(s_idx.get_noc_addr(0), idx_l1, idx_page_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
    uint32_t offsets[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        offsets[e] = static_cast<uint32_t>(ids[e]) * expert_stride + reader_offset;
    }

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    // Triple-buffered columns, one extra column in flight (transaction ids 1..3).
    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;

    for (uint32_t e = 0; e < num_sel; ++e) {
        uint32_t src = offsets[e];
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < pages_per_col; ++p) {
                noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                src += page_bytes;
                l1_write += page_bytes;
            }
            if (free_blocks == num_buffers - extra_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_w, kt);
                trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == num_buffers ? 1 : trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_base;
            }
        }
    }
    for (uint32_t i = 0; i < extra_in_flight; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, kt);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
}
