// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash residual add + RMSNorm, writer (BRISC) (resnorm_reader.cpp): this core's partial row sums of squares
// (cb_part, one FP32 tile, column 0) into slot nb of cb_all on each of the row's GX cores (itself included), then their
// semaphore 0 incremented; then the outputs: RN_OUT_BF16 the normed tiles (cb_n) to the L1-interleaved BF16 output
// (tile m, nb NB + j), RN_X8 their BFP8 copies (cb_n8) to the BFP8 output; RN_ROUTER this core's partial logits (cb_lp)
// into slot nb of the row's core 0 (semaphore 1), and on core 0 the row tile's logits (cb_lo) to the logits output
// (tile m).
//
// common runtime args: [n_addr, n8_addr, logits_addr, noc (x << 8 | y) of the grid's cores (y GX + x, 2 per word)]
// compile-time args: [cb_part, cb_all, cb_n, NB, GX, NT (column tiles of the output), cb_n8, cb_lp, cb_lall, cb_lo,
// n_tile (bytes of a cb_n tile: BF16 or FP32), accessors ((n), (n8), (logits))]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_all = get_compile_time_arg_val(1);
    constexpr uint32_t cb_n = get_compile_time_arg_val(2);
    constexpr uint32_t NB = get_compile_time_arg_val(3);
    constexpr uint32_t GX = get_compile_time_arg_val(4);
    constexpr uint32_t NT = get_compile_time_arg_val(5);
    constexpr uint32_t cb_n8 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_lp = get_compile_time_arg_val(7);
    constexpr uint32_t cb_lall = get_compile_time_arg_val(8);
    constexpr uint32_t cb_lo = get_compile_time_arg_val(9);
    constexpr uint32_t n_tile = get_compile_time_arg_val(10);
    constexpr uint32_t tile = 2048, tile8 = 1088, f32_tile = 4096;
    constexpr uint32_t noc_arg = 3;

    const uint32_t nb = get_absolute_logical_x(), m = get_absolute_logical_y();
    auto peer = [&](uint32_t x) -> uint64_t {  // core (x, m)'s NOC address base (add an L1 address)
        const uint32_t i = m * GX + x;
        const uint32_t w = get_common_arg_val<uint32_t>(noc_arg + i / 2) >> (16 * (i % 2));
        return get_noc_addr((w >> 8) & 0xFF, w & 0xFF, 0);
    };
    const uint32_t all_base = get_write_ptr(cb_all);  // (fresh CB: the same address on every core)
#ifndef RN_MPC
#define RN_MPC 1
#endif

    constexpr uint32_t acc_base = 11;
#ifdef RN_OUT_BF16
    constexpr auto n_args = TensorAccessorArgs<acc_base>();
    constexpr uint32_t acc_n8 = n_args.next_compile_time_args_offset();
    const auto s_n = TensorAccessor(n_args, get_common_arg_val<uint32_t>(0), n_tile);
#else
    constexpr uint32_t acc_n8 = acc_base;
#endif
#ifdef RN_X8
    constexpr auto n8_args = TensorAccessorArgs<acc_n8>();
    constexpr uint32_t acc_lo = n8_args.next_compile_time_args_offset();
    const auto s_n8 = TensorAccessor(n8_args, get_common_arg_val<uint32_t>(1), tile8);
#else
    constexpr uint32_t acc_lo = acc_n8;
#endif
#ifdef RN_ROUTER
    constexpr auto lo_args = TensorAccessorArgs<acc_lo>();
    const auto s_lo = TensorAccessor(lo_args, get_common_arg_val<uint32_t>(2), tile);
    const uint32_t lall_base = get_write_ptr(cb_lall);
#endif
    for (uint32_t mi = 0; mi < RN_MPC; ++mi) {
        const uint32_t rt = m * RN_MPC + mi;
        // this row tile's partial into slot mi GX + nb of every core of the row, then their semaphore 0
        cb_wait_front(cb_part, 1);
        const uint32_t src = get_read_ptr(cb_part);
        for (uint32_t k = 0; k < GX; ++k) {
            noc_async_write(src, peer(k) + all_base + (mi * GX + nb) * f32_tile, f32_tile);
        }
        noc_async_write_barrier();
        for (uint32_t k = 0; k < GX; ++k) {
            noc_semaphore_inc(peer(k) + get_semaphore(0), 1);
        }
        noc_async_atomic_barrier();
        cb_pop_front(cb_part, 1);
        for (uint32_t j = 0; j < NB; ++j) {
            const uint32_t page = rt * NT + nb * NB + j;
#ifdef RN_OUT_BF16
        cb_wait_front(cb_n, 1);
        noc_async_write(get_read_ptr(cb_n), s_n.get_noc_addr(page), n_tile);
#endif
#ifdef RN_X8
        cb_wait_front(cb_n8, 1);
        noc_async_write(get_read_ptr(cb_n8), s_n8.get_noc_addr(page), tile8);
#endif
        noc_async_writes_flushed();
#ifdef RN_OUT_BF16
        cb_pop_front(cb_n, 1);
#endif
#ifdef RN_X8
        cb_pop_front(cb_n8, 1);
#endif
        }
#ifdef RN_ROUTER
        {
            cb_wait_front(cb_lp, 1);
            noc_async_write(get_read_ptr(cb_lp), peer(0) + lall_base + (mi * GX + nb) * f32_tile, f32_tile);
            noc_async_write_barrier();
            noc_semaphore_inc(peer(0) + get_semaphore(1), 1);
            noc_async_atomic_barrier();
            cb_pop_front(cb_lp, 1);
            if (nb == 0) {
                cb_wait_front(cb_lo, 1);
                noc_async_write(get_read_ptr(cb_lo), s_lo.get_noc_addr(rt), tile);
                noc_async_write_barrier();
                cb_pop_front(cb_lo, 1);
            }
        }
#endif
    }
    noc_async_write_barrier();
}
