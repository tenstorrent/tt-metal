// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, rope core reader: the -1 scalar tile, the cos/sin tiles, then the hand-off from its norm core.
// Compile-time args: TensorAccessorArgs cos, sin.  Runtime args: 0 cos, 1 sin addresses.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "main_tail_cbs.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t cos_addr = get_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_arg_val<uint32_t>(1);
    constexpr auto cos_args = TensorAccessorArgs<0>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    const auto cos = TensorAccessor(cos_args, cos_addr);
    const auto sin = TensorAccessor(sin_args, sin_addr);
    const Noc noc;
    Semaphore<> rope_handoff(rope_semaphore_id);

    {
        generate_bcast_col_scalar(CircularBuffer(cb_scalar), 0xBF800000u);
        CircularBuffer(cb_cos).reserve_back(rope_tiles);
        CircularBuffer(cb_sin).reserve_back(rope_tiles);
        for (uint32_t t = 0; t < rope_tiles; ++t) {
            noc.async_read(
                cos,
                CircularBuffer(cb_cos),
                cos.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * tile_bytes});
            noc.async_read(
                sin,
                CircularBuffer(cb_sin),
                sin.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * tile_bytes});
        }
        noc.async_read_barrier();
        CircularBuffer(cb_cos).push_back(rope_tiles);
        CircularBuffer(cb_sin).push_back(rope_tiles);
    }

    {
        CircularBuffer(cb_in).reserve_back(rope_tiles);
        CircularBuffer(cb_rot).reserve_back(rope_tiles);
        rope_handoff.wait_min(1);
        CircularBuffer(cb_in).push_back(rope_tiles);
        CircularBuffer(cb_rot).push_back(rope_tiles);
        rope_handoff.set(0);
    }
}
