// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for gate_hm: row 0 of head h's Dt tiles into tiles h*Dt .. of the [1, 1, H*D] output row (the
// out-projection's input; bf16 or fp32, from the output's dtype). Only row 0 is written (two 16-element runs per
// tile, faces 0 and 1): the output is a persistent zeroed buffer, so its padding rows stay zero whatever z's padding
// held.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 4;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t esz = get_compile_time_arg_val(1);  // output element bytes (2 or 4)
    constexpr auto out_a = TensorAccessorArgs<2>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t h = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_a, out_addr, tb);
    Noc noc;
    CircularBuffer co(cb_out);
    co.wait_front(Dt);
    for (uint32_t t = 0; t < Dt; t++) {
        constexpr uint32_t run = 16 * esz, face = 256 * esz;
        noc.async_write(co, out, run, {.offset_bytes = t * tb}, {.page_id = h * Dt + t, .offset_bytes = 0});
        noc.async_write(co, out, run, {.offset_bytes = t * tb + face}, {.page_id = h * Dt + t, .offset_bytes = face});
    }
    noc.async_write_barrier();
    co.pop_front(Dt);
}
