// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC collapse (+ norm statistics), reader.  h[t, cols] = sum_i pre[t,i] x[t,i,cols].  In the token-row output layout
// one output tile (rows = tokens, 32 columns) is  sum_t A_t @ X_(t,j)  with A_t = zero tile whose row t holds pre[t,
// 0..3] and X_(t,j) the x tile (rows = streams) of token t: four accumulating tile matmuls, row t of the result is
// token t. The reader builds the T A tiles and a ones tile, and gathers the x tiles (rows 0..3 only; padding rows stay
// zero).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32) and word index
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))
#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t NT = get_compile_time_arg_val(5);
    constexpr uint32_t JB = get_compile_time_arg_val(6);  // column tiles per streamed batch
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto p_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t j1 = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB cba(cb_a), cbx(cb_x), cbo(cb_ones), cbs(cb_s);

    // ones tile
    cbo.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
        for (uint32_t k = 0; k < TILE / 4; ++k) {
            o[k] = 0x3F800000u;
        }
    }
    cbo.push_back(1);

    // A_t tiles
    cbs.reserve_back(1);
    cba.reserve_back(T);
    noc.async_write_zeros(cba, T * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
    volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(p_acc, cbs, 64, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* At = A + t * (TILE / 4);
        for (uint32_t i = 0; i < 4; ++i) {
            At[RW(t) + i] = sc[i];
        }
    }
    cba.push_back(T);

    // X tiles in (j, t) order, streamed in batches of JB column tiles (CB holds one batch)
    const uint32_t ng = j1 - j0;
    cbx.reserve_back(JB * T);
    noc.async_write_zeros(cbx, JB * T * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    for (uint32_t b0 = 0; b0 < ng; b0 += JB) {
        const uint32_t nbj = (ng - b0 < JB) ? (ng - b0) : JB;
        cbx.reserve_back(nbj * T);
        for (uint32_t gi = 0; gi < nbj; ++gi) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t dst = (gi * T + t) * TILE;
                const uint32_t page = t * NT + j0 + b0 + gi;
                noc.async_read(x_acc, cbx, 256, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = dst});
                noc.async_read(x_acc, cbx, 256, {.page_id = page, .offset_bytes = 1024}, {.offset_bytes = dst + 1024});
            }
        }
        noc.async_read_barrier();
        cbx.push_back(nbj * T);
    }
}
