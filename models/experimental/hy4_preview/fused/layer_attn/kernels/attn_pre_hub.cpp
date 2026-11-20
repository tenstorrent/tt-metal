// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The hub of the pre group in program P1 of the attention half (fused/layer_attn/program.py): the projections
// piece's hub (fused/projections/kernels/proj_hub.cpp, included as it is: its kernel_main runs as a function here)
// between the broadcast X1 and the small all-gather Xa.
//
//   1. the parameter page's row scales are set to 1 (x arrives normalised: the root applies the norm), and the
//      last word of the statistic tile to a sentinel.
//   2. wait (bounded) for x in this core's x buffer: word W_X, counted by the root chip's sender; take it off.
//   3. the piece's hub: x and the parameter page to the group's cores, release.
//   4. wait (bounded) until every output unit has written its tiles into this core's output buffer (word 1 of the
//      piece's go tensor), and until the statistic tile has landed: a unit counts itself BEFORE it writes its
//      statistic tile (proj_writer.cpp), so the count alone does not say the tile is there; the tile's last word
//      is zero as the unit writes it and stays the sentinel until then.
//   5. Xa: q_a's two tiles, the pool tile and the statistic tile go to this chip's sender core, then the ready
//      bits of their ops in one add.
// A wait that runs out leaves its code in this core's status word and the kernel carries on, so the program ends.
//
// Stamps (HY4L_TIMING): 0 start, 1 x arrived, 2 units released, 3 units done, 4 Xa handed over.
//
// Runtime args: 0..3 proj_hub.cpp's (x, parameter page, status, go), then: output units to wait for, the address of
// their done word, output buffer address, statistic tile address, sender core NoC x | y << 16, Xa staging address,
// ready word address, ready bits, whether the chip has a pool tile. Compile-time args and defines: proj_hub.cpp's,
// HY4L_*, HY4A_WORDS.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#define kernel_main hy4_proj_hub_main
#include "proj_hub.cpp"
#undef kernel_main

#include "attn_common.hpp"

void kernel_main() {
    constexpr uint32_t base = 4;
    const uint32_t meta_in = get_arg_val<uint32_t>(1);
    const uint32_t n_out = get_arg_val<uint32_t>(base);
    const uint32_t done_addr = get_arg_val<uint32_t>(base + 1);
    const uint32_t out_addr = get_arg_val<uint32_t>(base + 2);
    const uint32_t stat_addr = get_arg_val<uint32_t>(base + 3);
    const uint32_t sender = get_arg_val<uint32_t>(base + 4);
    const uint32_t staging = get_arg_val<uint32_t>(base + 5);
    const uint32_t ready = get_arg_val<uint32_t>(base + 6);
    const uint32_t bits = get_arg_val<uint32_t>(base + 7);
    const uint32_t has_pool = get_arg_val<uint32_t>(base + 8);

    hy4l::stamp(0);
    volatile tt_l1_ptr uint32_t* runs = hy4a::word(hy4a::A_RUNS);
    runs[0] = runs[0] + 1;
#ifdef HY4A_PRE_IDX
    // HY4A_PRE_IDX (a full-indexer layer whose P1 writes the scan's query; PLAN.md section 21.3; its value is the
    // first of the group's 4 index-query output tiles): the hub reads the step's cos | sin row tiles from the query
    // hub core's RoPE tensor into the two tiles behind x in its x buffer (the idx.rope unit's aux tiles; X1 writes x
    // only), so the index query's rope dims are rotated; in step 5 the 4 index-query tiles go to the sender's
    // staging too, at row tile HY4A_XA_IDX_TILE of it (the caller's ready bits hold the op's). Three more runtime
    // args: the query hub core's NoC x | y << 16, the RoPE tensor's address, the row tile of the x buffer the cos
    // tile goes to.
    {
        const uint32_t rope_core = get_arg_val<uint32_t>(base + 9);
        const uint32_t rope_addr = get_arg_val<uint32_t>(base + 10);
        const uint32_t rope_tile = get_arg_val<uint32_t>(base + 11);
        noc_async_read(
            get_noc_addr(rope_core & 0xFFFF, rope_core >> 16, rope_addr),
            get_arg_val<uint32_t>(0) + rope_tile * hy4proj::ROW_TILE_BYTES,
            2 * hy4proj::ROW_TILE_BYTES);
        noc_async_read_barrier();
    }
#endif
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_in);
    meta[0] = 1;
    for (uint32_t r = 0; r < ROWS; ++r) {
        meta[hy4proj::PARAM_SCALE / 4 + r] = 0x3F800000u;  // 1.0f
    }
    volatile tt_l1_ptr uint32_t* stat = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stat_addr);
    constexpr uint32_t stat_last = hy4proj::ROW_TILE_BYTES_F32 / 4 - 1;
    stat[stat_last] = 0xFFFFFFFFu;

    if (hy4l::wait_word(hy4l::W_X, 1, hy4l::ST_X)) {
        hy4l::take_word(hy4l::W_X, 1);
    }
    hy4l::stamp(1);
    hy4_proj_hub_main();
    hy4l::stamp(2);
    hy4l::wait_address(done_addr, n_out, hy4a::ST_UNITS);
    {
        uint32_t seen = 0xFFFFFFFFu;
        for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT && seen != 0; ++polls) {
            invalidate_l1_cache();
            seen = stat[stat_last];
        }
        if (seen != 0) {
            hy4a::fail(hy4a::ST_STAT, seen);
        }
    }
    hy4l::stamp(3);

    const uint32_t sx = sender & 0xFFFF;
    const uint32_t sy = sender >> 16;
    constexpr uint32_t tile = hy4proj::ROW_TILE_BYTES;
    noc_async_write(out_addr, get_noc_addr(sx, sy, staging), 2 * tile);
    if (has_pool) {
        noc_async_write(out_addr + 2 * tile, get_noc_addr(sx, sy, staging + 2 * tile), tile);
    }
    noc_async_write(stat_addr, get_noc_addr(sx, sy, staging + 4 * tile), hy4proj::ROW_TILE_BYTES_F32);
#ifdef HY4A_PRE_IDX
    noc_async_write(out_addr + HY4A_PRE_IDX * tile, get_noc_addr(sx, sy, staging + HY4A_XA_IDX_TILE * tile), 4 * tile);
#endif
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(sx, sy, ready), bits);
    noc_async_atomic_barrier();
    hy4l::stamp(4);
}
