// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Dense weight streamer (NCRISC, NOC0) for the decode streamed linear op (fused_decode/stream.py: LinearStream).
// This core's weight columns are one contiguous range of its DRAM bank (groups of G columns, K-major); it is read
// as `num_blocks` blocks of `block_tiles` tiles (one K block of one column group) in `page_bytes` packets, with one
// block in flight ahead of compute, into a ring of `num_buffers` blocks (the circular buffer's depth).
//
// runtime args: [w_addr, bank_id, vc, reader_offset_bytes(, KV_SPLIT: 1 if this core's first output column is K / V)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "bfp8_row.hpp"
#ifdef MOE_SPLIT_ROWS
#include "router_topk.hpp"
#endif

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t reader_offset = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);    // tiles per block
    constexpr uint32_t cols = get_compile_time_arg_val(2);  // blocks
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(5);

    constexpr uint32_t col_bytes = kt * tile_bytes;
    constexpr uint32_t pages_per_col = col_bytes / page_bytes;
    static_assert(pages_per_col * page_bytes == col_bytes, "page size must divide the block");

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
#ifdef LIN_R_WAIT_SEM
    // a fused program's earlier stage reads first (the drafter's attention + o_proj, dflash/fused_draft.py): start
    // streaming after LIN_R_WAIT_N increments of program semaphore LIN_R_WAIT_SEM
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(LIN_R_WAIT_SEM)), LIN_R_WAIT_N);
#endif
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;

    uint32_t src = reader_offset;
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
#ifdef KV_SPLIT
    auto kv_split = [&]() {
        // Multi-row QKV with the K / V cache write (stream_linear_writer.cpp): when this core's first output column
        // (runtime arg 4) is a K / V column, convert rows [rows / 2, rows) of it to BFP8 (shared exponent + 4 mantissa
        // words per 16-value half, 32 B apart in KV_SPLIT_CB_STAGE) while the writer converts the others; the writer
        // pops the tile. Cores holding only Q columns skip it.
        if (get_arg_val<uint32_t>(4)) {
            constexpr uint32_t rows = ROWS;
            constexpr uint32_t half_row = 32;  // one row of a [rows, 16] face
            constexpr uint32_t row_face = 32 * rows;
            cb_wait_front(KV_SPLIT_CB_OUT, 1);
            invalidate_l1_cache();
            const uint32_t src = get_read_ptr(KV_SPLIT_CB_OUT);
            volatile tt_l1_ptr uint32_t* st =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(KV_SPLIT_CB_STAGE));
            for (uint32_t i = rows / 2; i < rows; ++i) {
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    uint32_t shared, man[4];
                    bfp8_row::half_row(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + hh * row_face + i * half_row),
                        shared,
                        man);
                    volatile tt_l1_ptr uint32_t* e = st + 8 * (2 * i + hh);
                    e[0] = shared;
                    for (uint32_t q = 0; q < 4; ++q) {
                        e[1 + q] = man[q];
                    }
                }
            }
            asm volatile("fence" ::: "memory");
            noc_semaphore_inc(get_noc_addr(get_semaphore(KV_SPLIT_SEM)), 1);
            noc_async_atomic_barrier();
        }
    };
    if constexpr (cols > extra_in_flight) {
        kv_split();  // before the last columns' read barrier: the conversion overlaps their reads
    }
#endif
    for (uint32_t i = 0; i < extra_in_flight && i < cols; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, kt);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
#ifdef KV_SPLIT
    if constexpr (cols <= extra_in_flight) {
        kv_split();  // the K / V column is among the deferred pushes: convert once it is pushed (no deadlock)
    }
#endif
#ifdef MOE_SPLIT_ROWS
    // Router of the fused MoE (dflash/moe_fused.py, with MOE_EARLY_UNION): the second half of the rows' top-k
    // (stream_linear_writer.cpp runs the first half and the union), from the logits tile in cb_out (the writer pops it
    // after the union): ids first (the union goes out before the softmax), then the weights.
    {
        constexpr uint32_t rows = ROWS;
        constexpr uint32_t topk = MOE_SPLIT_TOPK;
        cb_wait_front(MOE_SPLIT_CB_OUT, 1);
        invalidate_l1_cache();
        const uint32_t src = get_read_ptr(MOE_SPLIT_CB_OUT);
        const uint32_t scr = get_write_ptr(MOE_SPLIT_CB_SCR);
        volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128 + 64 * rows + 128);
        volatile tt_l1_ptr uint32_t* wq = ids + rows * topk;
        volatile tt_l1_ptr uint32_t* keys = wq + rows * topk;
        router_topk::rows_ids<topk, rows>(reinterpret_cast<const uint32_t*>(src), rows / 2, rows, ids, keys);
        asm volatile("fence" ::: "memory");
        noc_semaphore_inc(get_noc_addr(get_semaphore(MOE_SPLIT_SEM)), 1);
        noc_async_atomic_barrier();
        router_topk::rows_softmax<topk>(rows / 2, rows, keys, wq);
        asm volatile("fence" ::: "memory");
        noc_semaphore_inc(get_noc_addr(get_semaphore(MOE_SPLIT_SEM)), 1);
        noc_async_atomic_barrier();
    }
#endif
}
