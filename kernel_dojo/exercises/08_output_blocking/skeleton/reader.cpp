// SPDX-License-Identifier: Apache-2.0
//
// Exercise 08 — 2-D blocked matmul reader.
//
// Instead of one row of A at a time (lesson 07), hold an Mb x Kt sub-block of A
// resident. Each B column you then read feeds Mb output tiles instead of one,
// so B's DRAM traffic drops by a factor of Mb.
//
// This core owns row-blocks [start_block, start_block + n_blocks). Row-block
// `blk` covers rows [blk * Mb, blk * Mb + Mb).

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Kt = get_arg_val<uint32_t>(2);
    const uint32_t Nt = get_arg_val<uint32_t>(3);
    const uint32_t start_block = get_arg_val<uint32_t>(4);
    const uint32_t n_blocks = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t Mb = get_compile_time_arg_val(2);

    constexpr auto a_args = TensorAccessorArgs<3>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t tile_bytes = get_tile_size(cb_a);
    const uint32_t end_block = start_block + n_blocks;

    for (uint32_t blk = start_block; blk < end_block; blk++) {
        const uint32_t row0 = blk * Mb;

        // TODO: reserve Mb * Kt pages in cb_a and fill them with A's sub-block.
        //       Tile (m, kt) of the sub-block is page (row0 + m) * Kt + kt of
        //       A, and must land at window slot m * Kt + kt so the compute
        //       kernel can find it. One barrier for the whole sub-block.

        for (uint32_t nt = 0; nt < Nt; nt++) {
            // TODO: same as lesson 07 — reserve Kt pages in cb_b, read B tiles
            //       (0..Kt-1, nt) at page kt * Nt + nt, barrier, push.
        }
    }
}
