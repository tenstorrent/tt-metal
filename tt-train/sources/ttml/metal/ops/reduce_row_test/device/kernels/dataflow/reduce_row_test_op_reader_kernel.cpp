// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api_addrgen.h>
#include <hostdevcommon/kernel_structs.h>

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_reduction_scaler = tt::CBIndex::c_1;
    constexpr auto cb_matmul_reduce = tt::CBIndex::c_2;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);

    const uint32_t tile_bytes = get_tile_size(cb_input);
    const DataFormat data_format = get_dataformat(cb_input);

    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    // generate scaler and mask tile
    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t

    generate_tile_with_bfloat16_value(
        cb_reduction_scaler, one);                      // generate tile with bfloat16 value 1.0 for reduction scaler
    generate_matmul_row_reduce_tile(cb_matmul_reduce);  // generate tile for matmul row reduce

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        // calculate the address of the first tile in the row
        // start_row is the number of rows already processed in other cores
        uint32_t idx = (start_row + i) * Wt;  // (take already processed rows + current row)*Wt(number of tiles in row)

        cb_reserve_back(cb_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(idx + j, input_address_generator, l1_write_addr);
            l1_write_addr += tile_bytes;
            noc_async_read_barrier();
        }
        cb_push_back(cb_input, Wt);
    }
}
