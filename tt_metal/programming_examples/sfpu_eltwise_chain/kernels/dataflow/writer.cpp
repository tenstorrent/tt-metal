// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t result_cb_index = get_compile_time_arg_val(0);

    // Input data config
    const uint32_t output_data_tile_size_bytes = get_tile_size(result_cb_index);
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<1>();
    const auto interleaved_accessor =
        TensorAccessor(interleaved_accessor_args, output_buffer_addr, output_data_tile_size_bytes);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Save output data
    cb_wait_front(result_cb_index, one_tile);
    const uint32_t l1_read_addr = get_read_ptr(result_cb_index);
    noc_async_write_tile(0, interleaved_accessor, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(result_cb_index, one_tile);
}
