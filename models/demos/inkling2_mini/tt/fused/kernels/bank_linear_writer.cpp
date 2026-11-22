// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Bank-streamed decode linear, writer (BRISC, NOC1) (tt/fused/bank_linear.py): row 0 of each fp32 output tile
// (face 0 row 0 = columns 0..15, face 1 row 0 = 16..31) into the dense fp32 output row at column 32 * n.
// tiles_out: instead the whole tile to page n of a TILE DRAM tensor. Columns >= out_cols are not written.
// First it reads the activation (kt tiles, the token rows) into cb_x on this NOC, block by block in the compute's
// walk order (block j = (p + rot) % nblk to position p, as the reference's lookahead), so compute starts after the
// first block while the weight stream runs on the reader's NOC. Runtime args: x address, rot, out address, number
// of passes, then (ncq, first column tile) per pass.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_out = 16;
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t nblk = get_named_compile_time_arg_val("nblk");
constexpr uint32_t out_cols = get_named_compile_time_arg_val("out_cols");
constexpr uint32_t out_page = get_named_compile_time_arg_val("out_page");
constexpr uint32_t tiles_out = get_named_compile_time_arg_val("tiles_out");
constexpr uint32_t out_tile = get_named_compile_time_arg_val("out_tile");
constexpr uint32_t DST_TILES = 8;

void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rot = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t npass = get_arg_val<uint32_t>(a++);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto out = TensorAccessor(out_args, out_addr, out_page);

    const uint32_t xtb = get_tile_size(cb_x);
    cb_reserve_back(cb_x, kt);
    const uint32_t xl = get_write_ptr(cb_x);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t p = 0; p < nblk; ++p) {
        const uint32_t j = (p + rot) % nblk;
        noc_async_read_set_trid(p % 4 + 1);
        for (uint32_t r = 0; r < kb; ++r) {
            noc_async_read(xs.get_noc_addr(j * kb + r), xl + (p * kb + r) * xtb, xtb);
        }
        if (p > 0) {
            noc_async_read_barrier_with_trid((p - 1) % 4 + 1);
            cb_push_back(cb_x, kb);
        }
    }
    noc_async_read_barrier_with_trid((nblk - 1) % 4 + 1);
    cb_push_back(cb_x, kb);
    noc_async_read_set_trid(0);
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t ncq = get_arg_val<uint32_t>(a++);
        const uint32_t n0 = get_arg_val<uint32_t>(a++);
        cb_wait_front(cb_out, DST_TILES);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < ncq; ++t) {
            const uint32_t col = (n0 + t) * 32;
            if (col >= out_cols) {
                continue;
            }
            if constexpr (tiles_out) {
                noc_async_write(src + t * out_tile, out.get_noc_addr(n0 + t), out_tile);
            } else {
                const uint64_t dst = out.get_noc_addr(0, col * 4);
                noc_async_write(src + t * 4096, dst, 64);
                noc_async_write(src + t * 4096 + 1024, dst + 64, 64);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, DST_TILES);
    }
}
