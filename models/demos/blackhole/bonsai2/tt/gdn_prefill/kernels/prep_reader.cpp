// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet chunk preprocessing, reader (NCRISC). One item = (head row bh, chunk c) of C = 128 tokens:
// q, k, v [128, 128] (4 x 4 fp32 tiles each, from [BH, T, 128] TILE), beta (4 column tiles from [BH, T, 1]) and
// g as 4 row tiles (row bh % 32 of the [BH, T] TILE tensor's tiles, rows 1-31 zero). Constant tiles once.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: q, k, v, beta, g, consts, Nitems, ncores, grid_x, NC (chunks per head), Tt (T / 32), BHt (rows
    // of g's tile grid = ceil(BH / 32))
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(5);
    uint32_t first, n;
    core_span(
        get_common_arg_val<uint32_t>(6), get_common_arg_val<uint32_t>(7), get_common_arg_val<uint32_t>(8), 1, first, n);
    const uint32_t NC = get_common_arg_val<uint32_t>(9);
    const uint32_t Tt = get_common_arg_val<uint32_t>(10);
    constexpr uint32_t NCONST = get_compile_time_arg_val(0);
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 4096;  // fp32 tile
    const auto qs = TensorAccessor(q_args, q_addr, TB);
    const auto ks = TensorAccessor(k_args, k_addr, TB);
    const auto vs = TensorAccessor(v_args, v_addr, TB);
    const auto bs = TensorAccessor(b_args, b_addr, TB);
    const auto gs = TensorAccessor(g_args, g_addr, TB);
    const auto cs = TensorAccessor(c_args, c_addr, TB);
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_b = 3, cb_g = 4, cb_c = 5;

    cb_reserve_back(cb_c, NCONST);
    for (uint32_t i = 0; i < NCONST; ++i) {
        noc_async_read(cs.get_noc_addr(i), get_write_ptr(cb_c) + i * TB, TB);
    }
    noc_async_read_barrier();
    cb_push_back(cb_c, NCONST);

    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t it = first; it < first + n; ++it) {
        const uint32_t bh = it / NC, c = it % NC;
        const uint32_t row0 = bh * Tt + 4 * c;  // first tile row of this chunk in [BH, T, *]
        cb_reserve_back(cb_g, 4);
        const uint32_t G = get_write_ptr(cb_g);
        for (uint32_t o = 0; o < 4 * TB; o += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, G + o, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        // g row bh % 32 of tiles (bh / 32, 4c + i): faces (0, 1) for rows < 16, else (2, 3); 64 B per face row.
        const uint32_t r = bh % 32, fbase = (r < 16 ? 0 : 2) * 1024 + (r % 16) * 64;
        for (uint32_t i = 0; i < 4; ++i) {
            const uint64_t src = gs.get_noc_addr((bh / 32) * Tt + 4 * c + i);
            noc_async_read(src + fbase, G + i * TB, 64);                // -> face 0 row 0
            noc_async_read(src + fbase + 1024, G + i * TB + 1024, 64);  // -> face 1 row 0
        }
        cb_reserve_back(cb_b, 4);
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_read(bs.get_noc_addr(row0 + i), get_write_ptr(cb_b) + i * TB, TB);
        }
        cb_reserve_back(cb_q, 16);
        cb_reserve_back(cb_k, 16);
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t d = 0; d < 4; ++d) {
                const uint32_t t = (row0 + i) * 4 + d, o = (i * 4 + d) * TB;
                noc_async_read(qs.get_noc_addr(t), get_write_ptr(cb_q) + o, TB);
                noc_async_read(ks.get_noc_addr(t), get_write_ptr(cb_k) + o, TB);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_g, 4);
        cb_push_back(cb_b, 4);
        cb_push_back(cb_k, 16);
        cb_push_back(cb_q, 16);
        for (uint32_t h = 0; h < 2; ++h) {  // v in two halves (the compute takes 8 tiles at a time)
            cb_reserve_back(cb_v, 8);
            for (uint32_t x = 0; x < 8; ++x) {
                const uint32_t i = (h * 8 + x) / 4, d = (h * 8 + x) % 4;
                noc_async_read(vs.get_noc_addr((row0 + i) * 4 + d), get_write_ptr(cb_v) + x * TB, TB);
            }
            noc_async_read_barrier();
            cb_push_back(cb_v, 8);
        }
    }
}
