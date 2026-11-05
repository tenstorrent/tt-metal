// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode rotation x' = H D x, one 1024-block per core, reader (NCRISC). Gathers row 0 of activation tiles
// 32b .. 32b+31 (face 0 row 0 then face 1 row 0, 32 B each) into a row-major 32x32 block X[jh, jl] = x[1024b + 32jh +
// jl] and loads the block's sign tile (D, +-1, tiled: the compute multiplies) and the two Hadamard factor tiles. The
// gather is 64-B DRAM reads into scratch, then 32-B local NoC copies (a RISC word loop took 5 us per 2 KB).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"

void kernel_main() {
#ifdef ROT_COMMON
    // ROT_COMMON: the same args on every core: x, m, h, grid_x, u; block b = core index, row jh reads tile 32 b + jh.
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(3));
#define SRC_TILE(jh) (32 * b + (jh))
#define U_ADDR get_common_arg_val<uint32_t>(4)
#else
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t b = get_arg_val<uint32_t>(3);
#define SRC_TILE(jh) get_arg_val<uint32_t>(4 + (jh))
#define U_ADDR get_arg_val<uint32_t>(36)
#endif
    // args 4 .. 35: the activation tile each of the block's 32 rows reads (32b + jh unless the input is permuted)
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
#ifdef ROT_GLU
    // ROT_GLU: a second activation u (arg 36); the compute rotates silu(x) * u. The signs D go onto u only:
    // D (silu(x) * u) = silu(x) * (D u).
    constexpr auto u_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto us = TensorAccessor(u_args, U_ADDR, 2048);
#endif
    constexpr uint32_t cb_x = tt::CBIndex::c_0, cb_m = tt::CBIndex::c_1, cb_h = tt::CBIndex::c_2;
    const auto xs = TensorAccessor(x_args, x_addr, 2048);
    const auto ms = TensorAccessor(m_args, m_addr, 2048);
    const auto hs = TensorAccessor(h_args, h_addr, 2048);

    cb_reserve_back(cb_h, 2);
    noc_async_read(hs.get_noc_addr(0), get_write_ptr(cb_h), 2048);
    noc_async_read(hs.get_noc_addr(1), get_write_ptr(cb_h) + 2048, 2048);
    cb_reserve_back(cb_m, 1);
    noc_async_read(ms.get_noc_addr(b), get_write_ptr(cb_m), 2048);
    noc_async_read_barrier();
    cb_push_back(cb_m, 1);
    // 64-B reads (face row 0 and row 1) into scratch: DRAM reads must be 64-B aligned and sized on Blackhole, and a
    // 32-B read of a DRAM-resident activation returned other data. Row 1 is dropped by the local copies.
    constexpr uint32_t cb_scratch = tt::CBIndex::c_3;
    cb_reserve_back(cb_scratch, 1);
    const uint32_t sc = get_write_ptr(cb_scratch);
    auto gather = [&](const auto& src, uint32_t dst) {
        for (uint32_t jh = 0; jh < 32; ++jh) {
            const uint64_t t = src.get_noc_addr(SRC_TILE(jh));
            noc_async_read(t, sc + jh * 128, 64);
            noc_async_read(t + 512, sc + jh * 128 + 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t jh = 0; jh < 32; ++jh) {  // row jh: face 0 row 0, then face 1 row 0
            noc_async_read(get_noc_addr(sc + jh * 128), dst + jh * 64, 32);
            noc_async_read(get_noc_addr(sc + jh * 128 + 64), dst + jh * 64 + 32, 32);
        }
        noc_async_read_barrier();
    };
    cb_reserve_back(cb_x, 1);
    gather(xs, get_write_ptr(cb_x));
#ifdef ROT_GLU
    constexpr uint32_t cb_u = tt::CBIndex::c_4;
    cb_reserve_back(cb_u, 1);
    gather(us, get_write_ptr(cb_u));
    cb_push_back(cb_u, 1);
#endif
    cb_push_back(cb_h, 2);
    cb_push_back(cb_x, 1);
}
