// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(0);
    constexpr auto input_args = TensorAccessorArgs<1>();
    constexpr auto bias_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0));
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(1));
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_tile_size(0);
    Noc noc;
    DataflowBuffer in_cb(0);
    DataflowBuffer bias_cb(1);
    for (uint32_t i = 0; i < count; i += 4) {
        if (i == 0 || (start + i) % tiles_per_head == 0) {
            const uint32_t head = (start + i) / tiles_per_head;
            // Single four-tile CB caches this head until math pops at the
            // boundary; reserve prevents overwrite of a still-live bias.
            bias_cb.reserve_back(4);
            for (uint32_t j = 0; j < 4; ++j) {
                noc.async_read(bias, bias_cb, bytes, {.page_id = head * 4 + j}, {.offset_bytes = j * bytes});
            }
            noc.async_read_barrier();
            bias_cb.push_back(4);
        }
        in_cb.reserve_back(4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc.async_read(input, in_cb, bytes, {.page_id = start + i + j}, {.offset_bytes = j * bytes});
        }
        noc.async_read_barrier();
        in_cb.push_back(4);
    }
}

