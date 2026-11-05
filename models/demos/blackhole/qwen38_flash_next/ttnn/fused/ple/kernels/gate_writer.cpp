// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gate writer, one core: the local_tiles gated tiles (CB 16, bf16) -> pages 0..local_tiles-1 of the output; with
// debug, the fp32 sum tile (CB 17) and coefficient tile (CB 18) -> page 0 of two fp32 tile tensors.
// Compile-time args: 0 local_tiles (the builder's LOCAL_TILES), 1 debug, then TensorAccessorArgs(out), (debug sum),
// (debug coefficient).
// Runtime args: 0 out addr, 1 debug sum addr, 2 debug coefficient addr, 3 the first output tile (the lane's
// local_tiles block; 0 for the 1-row form).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t local_tiles = get_compile_time_arg_val(0);
constexpr uint32_t debug = get_compile_time_arg_val(1);
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096;
constexpr uint32_t cb_out = 16, cb_debug_sum = 17, cb_debug_coefficient = 18;

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr auto debug_sum_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto debug_coefficient_args = TensorAccessorArgs<debug_sum_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(3);
    const Noc noc;
    if constexpr (debug) {
        const auto debug_sum = TensorAccessor(debug_sum_args, get_arg_val<uint32_t>(1));
        const auto debug_coefficient = TensorAccessor(debug_coefficient_args, get_arg_val<uint32_t>(2));
        DataflowBuffer sum_tile(cb_debug_sum), coefficient_tile(cb_debug_coefficient);
        sum_tile.wait_front(1);
        noc.async_write(sum_tile, debug_sum, fp32_tile_bytes, {.offset_bytes = 0}, {.page_id = 0, .offset_bytes = 0});
        coefficient_tile.wait_front(1);
        noc.async_write(
            coefficient_tile,
            debug_coefficient,
            fp32_tile_bytes,
            {.offset_bytes = 0},
            {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
        sum_tile.pop_front(1);
        coefficient_tile.pop_front(1);
    }
    {
        DataflowBuffer gated(cb_out);
        gated.wait_front(local_tiles);
        for (uint32_t t = 0; t < local_tiles; ++t) {
            noc.async_write(
                gated,
                out,
                bf16_tile_bytes,
                {.offset_bytes = t * bf16_tile_bytes},
                {.page_id = first + t, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        gated.pop_front(local_tiles);
    }
}
