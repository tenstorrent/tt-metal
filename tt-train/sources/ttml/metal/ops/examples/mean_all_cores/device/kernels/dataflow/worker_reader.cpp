// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api_addrgen.h>

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    // Runtime arguments
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

    // Circular buffer indices
    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduction_scaler_idx = tt::CBIndex::c_1;

    // Compile time arguments
    constexpr uint32_t Wt = get_compile_time_arg_val(0);

    // Get tile size
    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    // Setup tensor accessor for reading input
    constexpr auto input_args = TensorAccessorArgs<1>();
    const auto input_address_generator = TensorAccessor(input_args, input_address, tile_bytes);

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    generate_tile_with_bfloat16_value(
        cb_reduction_scaler_idx, one);  // generate tile with bfloat16 value 1.0 for reduction scaler

    for (uint32_t row = start_row; row < start_row + num_rows_to_process; ++row) {
        cb_reserve_back(cb_input_idx, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_idx);
        for (uint32_t col = 0; col < Wt; ++col) {
            noc_async_read_tile(col, input_address_generator, l1_write_addr);
            l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_idx, Wt);
    }
}
