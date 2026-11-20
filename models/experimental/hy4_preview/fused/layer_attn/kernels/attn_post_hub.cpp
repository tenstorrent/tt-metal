// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The hub of the post group in program P2 of the attention half (fused/layer_attn/program.py): the projections
// piece's hub (proj_hub.cpp, included as it is) between the merge X4 and the reduce X5.
//
//   1. wait (bounded) until the chip's merge cores have written their rows of ctx (the merged attention output of
//      the chip's 2 heads in latent space) into this core's x buffer: word A_CTX, one count per merge core.
//   2. the gate of the chip's 2 heads, which the pre group of P1 left in the pre hub core's output buffer, is read
//      behind ctx in the x buffer.
//   3. the piece's hub: x and the parameter page to the group's cores, release.
//   4. wait (bounded) for the o_proj units; the chip's partial of the sublayer output goes on (hy4l::feed_reduce):
//      to this chip's sender core, or into its own reduce cores' stage-1 landing when the chip heads its mesh row.
// A wait that runs out leaves its code in this core's status word and the kernel carries on, so the program ends.
//
// Stamps (HY4L_TIMING): 0 start, 1 ctx complete, 2 units released, 3 partial complete, 4 partial handed on.
//
// Runtime args: 0..3 proj_hub.cpp's (x, parameter page, status, go), then: output units to wait for, the address
// of their done word, output buffer address, merge cores expected, pre hub core NoC x | y << 16, the gate's address
// there, its bytes, its first tile in the x buffer, then hy4l::feed_reduce's arguments.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#define kernel_main hy4_proj_hub_main
#include "proj_hub.cpp"
#undef kernel_main

#ifdef HY4A_ROWS16
#include "attn_rows_dm.hpp"
#else
#include "attn_common.hpp"
#endif

void kernel_main() {
    constexpr uint32_t base = 4;
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t meta_in = get_arg_val<uint32_t>(1);
    const uint32_t n_out = get_arg_val<uint32_t>(base);
    const uint32_t done_addr = get_arg_val<uint32_t>(base + 1);
    const uint32_t out_addr = get_arg_val<uint32_t>(base + 2);
#ifdef HY4A_ROWS16
    // 16 token rows: a merge core counts once per sub-pass in which its row is live (attn_rows.hpp)
    const uint32_t merges = hy4r::live_rows(hy4a::row_mask());
#else
    const uint32_t merges = get_arg_val<uint32_t>(base + 3);
#endif
    const uint32_t pre_hub = get_arg_val<uint32_t>(base + 4);
    const uint32_t gate_addr = get_arg_val<uint32_t>(base + 5);
    const uint32_t gate_bytes = get_arg_val<uint32_t>(base + 6);
    const uint32_t gate_tile = get_arg_val<uint32_t>(base + 7);

    hy4l::stamp(0);
    volatile tt_l1_ptr uint32_t* runs = hy4a::word(hy4a::A_RUNS);
    runs[0] = runs[0] + 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_in)[0] = 1;
    if (hy4a::wait_word(hy4a::A_CTX, merges, hy4a::ST_CTX)) {
        hy4a::take_word(hy4a::A_CTX, merges);
    }
    hy4l::stamp(1);
    noc_async_read(
        get_noc_addr(pre_hub & 0xFFFF, pre_hub >> 16, gate_addr),
        x_addr + gate_tile * hy4proj::ROW_TILE_BYTES,
        gate_bytes);
    noc_async_read_barrier();
    hy4_proj_hub_main();
    hy4l::stamp(2);
    hy4l::wait_address(done_addr, n_out, hy4a::ST_UNITS);
    hy4l::stamp(3);
    hy4l::feed_reduce(base + 8, out_addr);
    volatile tt_l1_ptr uint32_t* fed = hy4l::word(hy4l::W_FED);
    fed[0] = fed[0] + 1;
    hy4l::stamp(4);
}
