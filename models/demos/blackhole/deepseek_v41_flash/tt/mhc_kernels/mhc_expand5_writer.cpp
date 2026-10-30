// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand v5 writer (token-group partition: one core = one token group of 4 x NG column tiles; runtime arg 1 =
// group) of v4 (DSV41_MHC_EXPAND_FAST): ONE block-diagonal matmul per (token group of 4, column tile) instead of one
// matmul per (token, column tile): A_g[4*tl + j][4*tl + i] = comb[t][i][j], A_g[4*tl + j][16 + tl] = post[t][j] (+ y2
// term at 20 + tl): every output row is the same K-ordered sum as before (the other K slots are exact zeros), only its
// row position in the tile moves. (v3 text:) mHC expand, VALID-ROW writer (BRISC): the output tiles are written as
// their two valid 256 B pieces only (face 0 rows 0..3 at byte 0, face 1 rows 0..3 at byte 1024); the destination buffer
// is pre-zeroed (mhc_stream_pool) and its padding is never touched. mHC expand, writer (BRISC).  First builds the
// per-token A tiles (the reader does the B tiles, in parallel):
//   A_t[j][4*tl + i] = comb[t][i][j],  A_t[j][16 + tl] = post[t][j],  A_t[j][20 + tl] = post[t][j] (y2 term)
// so that  out_t (rows j = new streams) = A_t @ B_(group, col tile).  Then streams the output tiles to DRAM.

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sa = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t NG = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_Y2 = get_compile_time_arg_val(6);
    constexpr uint32_t G = 1;  // this core's token group
    constexpr auto o_args = TensorAccessorArgs<7>();
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t tb = 4 * get_arg_val<uint32_t>(1);  // first token of the group
    const uint32_t nv = (T - tb) < 4 ? (T - tb) : 4;   // valid tokens of the group
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, TILE);
    const auto c_acc = TensorAccessor(c_args, c_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB out(cb_out), cba(cb_a), csa(cb_sa);

    DeviceZoneScopedN("EW_ALL");
    csa.reserve_back(1);
    cba.reserve_back(G);
    for (uint32_t tl = 0; tl < nv; ++tl) {
        noc.async_read(c_acc, csa, 256, {.page_id = tb + tl, .offset_bytes = 0}, {.offset_bytes = tl * 512});
        noc.async_read(p_acc, csa, 256, {.page_id = tb + tl, .offset_bytes = 0}, {.offset_bytes = tl * 512 + 256});
    }
    noc.async_write_zeros(cba, G * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    volatile tt_l1_ptr uint32_t* SC = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(csa.get_write_ptr());
    for (uint32_t tl = 0; tl < nv; ++tl) {
        volatile tt_l1_ptr uint32_t* At = A;  // block tile of the token group: rows 4*tl + j (new stream j of token tl)
        volatile tt_l1_ptr uint32_t* sc = SC + tl * 128;
        volatile tt_l1_ptr uint32_t* sp = sc + 64;
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t r = 4 * tl + j;
            for (uint32_t i = 0; i < 4; ++i) {
                At[r * 16 + 4 * tl + i] = sc[i * 16 + j];
            }
            At[256 + r * 16 + tl] = sp[j * 16];
            if (HAS_Y2) {
                At[256 + r * 16 + 4 + tl] = sp[j * 16];
            }
        }
    }
    cba.push_back(G);
    {
        DeviceZoneScopedN("EW_Adone");
    }

    const uint32_t nb = NG;  // one output tile per column tile: rows 4*tl + j hold the new streams of token tb + tl
    constexpr uint32_t CH = 4;
    for (uint32_t k = 0; k < nb; k += CH) {
        const uint32_t n = (nb - k) < CH ? (nb - k) : CH;
        out.wait_front(n);
        for (uint32_t q = 0; q < n; ++q) {
            const uint32_t j = j0 + k + q;
            for (uint32_t tl = 0; tl < nv; ++tl) {
                const uint32_t t = tb + tl;
                noc.async_write(
                    out, o_acc, 256, {.offset_bytes = q * TILE + tl * 256}, {.page_id = t * NT + j, .offset_bytes = 0});
                noc.async_write(
                    out,
                    o_acc,
                    256,
                    {.offset_bytes = q * TILE + 1024 + tl * 256},
                    {.page_id = t * NT + j, .offset_bytes = 1024});
            }
        }
        noc.async_writes_flushed();
        out.pop_front(n);
    }
    noc.async_write_barrier();
}
