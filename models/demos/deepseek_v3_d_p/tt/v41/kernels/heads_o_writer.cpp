// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the V4.1 attention-output head layout (heads_o_compute.cpp): the tiled [1, G, S, HPG * D] output, group g
// holding heads g * HPG .. g * HPG + HPG - 1 side by side (the grouped wo_a input). Per unit (tile row r, head h =
// g * HPG + j): the NT no-rope tiles and the RT inverse-rotated tail tiles of head j's columns in tile row r of g.
//
// compile_time_args = [NT, RT, H, HPG, ST, TensorAccessorArgs(out)]  (ST = S / 32)
// runtime args      = [out_addr, unit_start, unit_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t HPG = get_compile_time_arg_val(3);
    constexpr uint32_t ST = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr uint32_t DT = NT + RT;
    constexpr uint32_t cb_nope = 16, cb_roped = 17;

    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    DataflowBuffer nope(cb_nope);
    DataflowBuffer roped(cb_roped);
    const uint32_t page = get_local_cb_interface(cb_nope).fifo_page_size;  // bf16 tile
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t r = u / H;
        const uint32_t h = u - r * H;
        const uint32_t g = h / HPG;
        const uint32_t first = ((g * ST + r) * HPG + (h - g * HPG)) * DT;
        nope.wait_front(NT);
        for (uint32_t t = 0; t < NT; ++t) {
            noc.async_write(nope, out, page, {.offset_bytes = t * page}, {.page_id = first + t});
        }
        roped.wait_front(RT);
        for (uint32_t t = 0; t < RT; ++t) {
            noc.async_write(roped, out, page, {.offset_bytes = t * page}, {.page_id = first + NT + t});
        }
        noc.async_writes_flushed();
        nope.pop_front(NT);
        roped.pop_front(RT);
    }
    noc.async_write_barrier();
}
