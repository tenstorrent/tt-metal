// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for gate_hm v2: packed row t of head h's output tile becomes row 0 of tile h*Dt + t of the [1, 1, H*D]
// output (the out-projection's input; bf16 or fp32, from the output's dtype), two 16-element runs per row (faces 0
// and 1). A bf16 row t sits at t*32 B while its destination is 64 B aligned, so every run is first staged at a 64 B
// slot. Only row 0 is written: the output is a persistent zeroed buffer, so its padding rows stay zero.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 4, cb_stage = 7;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t esz = get_compile_time_arg_val(1);  // output element bytes (2 or 4)
    constexpr auto out_a = TensorAccessorArgs<2>();
    const uint32_t h = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_a, get_arg_val<uint32_t>(0), tb);
    constexpr uint32_t run = 16 * esz, face = 256 * esz;
    Noc noc;
    CircularBuffer co(cb_out), st(cb_stage);
    co.wait_front(1);
    const CoreLocalMem<volatile uint32_t> src(co.get_read_ptr());
    const CoreLocalMem<volatile uint32_t> stg(st.get_write_ptr());
    for (uint32_t t = 0; t < Dt; t++) {
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t s_w = (f * face + t * run) / 4, d_w = (t * 2 + f) * 16;
            for (uint32_t x = 0; x < run / 4; x++) {
                stg[d_w + x] = src[s_w + x];
            }
            noc.async_write(st, out, run, {.offset_bytes = (t * 2 + f) * 64},
                            {.page_id = h * Dt + t, .offset_bytes = f * face});
        }
    }
    noc.async_write_barrier();
    co.pop_front(1);
}
