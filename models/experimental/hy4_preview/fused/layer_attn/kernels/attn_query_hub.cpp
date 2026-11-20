// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The hub of the query group in program P1 of the attention half (fused/layer_attn/program.py): the projections
// piece's hub (proj_hub.cpp, included as it is) between the small all-gather Xa and the all-to-all X3.
//
//   1. wait (bounded) until every chip's two q_a tiles are in this core's x buffer (word A_QA, counted by the
//      chips' senders) and every chip's statistic tile is in the lane landing (A_LANES); take the counts off.
//   2. the lane landing goes to this core's compute kernel (attn_lane_sum.cpp), which adds the chips' tiles in
//      Float32; the 8 x 8 lane sums of the result go into the parameter page (the query units take
//      rsqrt(sum / 2048 + eps) of them), the row scales are 1.
//   3. the RoPE tiles of the rows' positions (cos | sin, a host-written tensor on this core) go behind q_a in the x
//      buffer.
//   4. the piece's hub: x and the parameter page to the group's cores, release.
//   5. wait (bounded) for the output units; X3: the absorbed queries of the chip's 2 heads go to the sender core,
//      then the op's ready bit.
// A wait that runs out leaves its code in this core's status word and the kernel carries on, so the program ends.
//
// Stamps (HY4L_TIMING): 0 start, 1 Xa complete, 2 units released, 3 units done, 4 X3 handed over.
//
// Runtime args: 0..3 proj_hub.cpp's (x, parameter page, status, go), then: output units to wait for, the address of
// their done word, output buffer address, output bytes, chips expected in Xa, RoPE tensor address, RoPE tile index in
// the x buffer, sender core NoC x | y << 16, X3 staging address, ready word address, ready bit. Compile-time args:
// proj_hub.cpp's; named: cb_land (the lane landing's circular buffer), cb_sum, land_pages.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#define kernel_main hy4_proj_hub_main
#include "proj_hub.cpp"
#undef kernel_main

#include "attn_common.hpp"

void kernel_main() {
    constexpr uint32_t base = 4;
    constexpr uint32_t cb_land = get_named_compile_time_arg_val("cb_land");
    constexpr uint32_t cb_sum = get_named_compile_time_arg_val("cb_sum");
    constexpr uint32_t land_pages = get_named_compile_time_arg_val("land_pages");
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t meta_in = get_arg_val<uint32_t>(1);
    const uint32_t n_out = get_arg_val<uint32_t>(base);
    const uint32_t done_addr = get_arg_val<uint32_t>(base + 1);
    const uint32_t out_addr = get_arg_val<uint32_t>(base + 2);
    const uint32_t out_bytes = get_arg_val<uint32_t>(base + 3);
    const uint32_t chips = get_arg_val<uint32_t>(base + 4);
    const uint32_t rope_addr = get_arg_val<uint32_t>(base + 5);
    const uint32_t rope_tile = get_arg_val<uint32_t>(base + 6);
    const uint32_t sender = get_arg_val<uint32_t>(base + 7);
    const uint32_t staging = get_arg_val<uint32_t>(base + 8);
    const uint32_t ready = get_arg_val<uint32_t>(base + 9);
    const uint32_t bit = get_arg_val<uint32_t>(base + 10);

    hy4l::stamp(0);
    volatile tt_l1_ptr uint32_t* runs = hy4a::word(hy4a::A_RUNS);
    runs[0] = runs[0] + 1;
    if (hy4a::wait_word(hy4a::A_QA, chips, hy4a::ST_QA)) {
        hy4a::take_word(hy4a::A_QA, chips);
    }
    if (hy4a::wait_word(hy4a::A_LANES, chips, hy4a::ST_LANES)) {
        hy4a::take_word(hy4a::A_LANES, chips);
    }
    hy4l::stamp(1);

    cb_reserve_back(cb_land, land_pages);
    cb_push_back(cb_land, land_pages);
    cb_wait_front(cb_sum, 1);
    {
        // the sum is a Float32 [32, 32] tile whose first face holds the chips' [8, 16] face 0: row r, even columns
        volatile tt_l1_ptr uint32_t* sum = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_sum));
        volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_in);
        meta[0] = 1;
        for (uint32_t r = 0; r < ROWS; ++r) {
            meta[hy4proj::PARAM_SCALE / 4 + r] = 0x3F800000u;  // 1.0f
            for (uint32_t l = 0; l < hy4proj::LANES; ++l) {
                meta[hy4proj::PARAM_LANES / 4 + r * hy4proj::LANES + l] = sum[r * 16 + 2 * l];
            }
        }
    }
    cb_pop_front(cb_sum, 1);
    {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rope_addr);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_addr + rope_tile * hy4proj::ROW_TILE_BYTES);
        for (uint32_t i = 0; i < 2 * hy4proj::ROW_TILE_BYTES / 4; ++i) {
            dst[i] = src[i];
        }
    }
    hy4l::stamp(2);
    hy4_proj_hub_main();
    hy4l::wait_address(done_addr, n_out, hy4a::ST_UNITS);
    hy4l::stamp(3);

    const uint32_t sx = sender & 0xFFFF;
    const uint32_t sy = sender >> 16;
    noc_async_write(out_addr, get_noc_addr(sx, sy, staging), out_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(sx, sy, ready), bit);
    noc_async_atomic_barrier();
    hy4l::stamp(4);
}
