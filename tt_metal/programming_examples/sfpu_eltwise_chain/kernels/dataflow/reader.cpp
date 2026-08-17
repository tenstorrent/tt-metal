// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-metalium/constants.hpp"
#include "api/numeric/bfloat16.h"

#include <cstdint>
#include <cstring>

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t src_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(1);

    // Input data config
    constexpr auto interleaved_accessor_args = TensorAccessorArgs<2>();
    const auto interleaved_accessor = TensorAccessor(interleaved_accessor_args, input_buffer_addr);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    cb_reserve_back(src_cb_index, one_tile);
    const uint32_t l1_write_addr = get_write_ptr(src_cb_index);
    noc_async_read_page(0, interleaved_accessor, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(src_cb_index, one_tile);

    // Create tile with ones
    cb_reserve_back(ones_cb_index, one_tile);
    const uint32_t ones_l1_write_addr = get_write_ptr(ones_cb_index);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = fp32_to_bf16_truncate(1.0f);
    }
    cb_push_back(ones_cb_index, one_tile);
}
