// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill (residual add +) RMSNorm + rotation (norm_rot.py), reader (NCRISC). The const tiles into c_2 once; this
// core's token row tile: HAS_ADD: a and b tile by tile into c_0 / c_1 (the compute adds them); else x straight into
// c_3, the row the compute keeps.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t Kt = get_common_arg_val<uint32_t>(3);
    const uint32_t rows = get_common_arg_val<uint32_t>(4);
    const uint32_t nconst = get_common_arg_val<uint32_t>(5);
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1), b1 = get_arg_val<uint32_t>(2), paired = get_arg_val<uint32_t>(3);
    const uint32_t px = get_arg_val<uint32_t>(4), py = get_arg_val<uint32_t>(5);  // partner (virtual coordinates)
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 2, cb_h = 3, cb_r = 5, cb_rp = 8;
    if (me >= rows) {
        return;
    }
    const uint32_t TB = get_tile_size(cb_h);
    const uint32_t T0 = 32 * b0, Kown = 32 * (b1 - b0);
    if (paired) {
        cb_reserve_back(cb_rp, 1);  // the partner writes its partial sums here (the same L1 address on both cores)
    }
    const auto as = TensorAccessor(a_args, a_addr, TB);
    const auto cs = TensorAccessor(c_args, c_addr, get_tile_size(cb_c));
    cb_reserve_back(cb_c, nconst);
    const uint32_t lc = get_write_ptr(cb_c);
    // only the const tiles this core uses: ONES, +-H32/32 and its blocks' vector tiles (w * d, then w); every core
    // reading all of them crowded the same DRAM pages
    const uint32_t ct = get_tile_size(cb_c), NV = Kt / 8;
    auto rd_const = [&](uint32_t t) { noc_async_read(cs.get_noc_addr(t), lc + t * ct, ct); };
    for (uint32_t t = 0; t < 3; ++t) {
        rd_const(t);
    }
    for (uint32_t v = (32 * b0) / 8; v < (32 * b1) / 8; ++v) {
        rd_const(3 + v);
#ifdef WRITE_N
        rd_const(3 + NV + v);
#endif
    }
#ifdef HAS_ADD
    const auto bs = TensorAccessor(b_args, b_addr, TB);
    constexpr uint32_t BATCH = 8;  // tiles per read barrier (c_0 / c_1 hold two batches)
    for (uint32_t t0 = 0; t0 < Kown; t0 += BATCH) {
        const uint32_t n = Kown - t0 < BATCH ? Kown - t0 : BATCH;
        cb_reserve_back(cb_a, n);
        cb_reserve_back(cb_b, n);
        const uint32_t la = get_write_ptr(cb_a), lb = get_write_ptr(cb_b);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(as.get_noc_addr(me * Kt + T0 + t0 + i), la + i * TB, TB);
            noc_async_read(bs.get_noc_addr(me * Kt + T0 + t0 + i), lb + i * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, n);
        cb_push_back(cb_b, n);
        if (t0 == 0) {
            cb_push_back(cb_c, nconst);
        }
    }
#else
    cb_reserve_back(cb_h, Kown);
    const uint32_t lh = get_write_ptr(cb_h);
    for (uint32_t t = 0; t < Kown; ++t) {
        noc_async_read(as.get_noc_addr(me * Kt + T0 + t), lh + t * TB, TB);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, nconst);
    cb_push_back(cb_h, Kown);
#endif
    if (paired) {
        // send this core's partial sums of squares (the compute's c_5) to the partner's c_8, then wait for the
        // partner's to arrive in this core's c_8
        const uint32_t sem = get_semaphore(0);
        volatile tt_l1_ptr uint32_t* sem_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        cb_wait_front(cb_r, 1);
        noc_async_write(get_read_ptr(cb_r), get_noc_addr(px, py, get_write_ptr(cb_rp)), get_tile_size(cb_r));
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, sem), 1);
        noc_semaphore_wait(sem_p, 1);
        noc_semaphore_set(sem_p, 0);
        cb_push_back(cb_rp, 1);
        noc_async_atomic_barrier();
    }
}
