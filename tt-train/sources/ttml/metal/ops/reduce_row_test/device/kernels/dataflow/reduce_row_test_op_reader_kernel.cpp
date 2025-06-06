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
    uint32_t first_input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t second_input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

    constexpr uint32_t cb_first_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_second_input = tt::CBIndex::c_1;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);

    const uint32_t tile_bytes = get_tile_size(cb_first_input);
    const DataFormat data_format = get_dataformat(cb_first_input);

    const InterleavedAddrGenFast</* is_dram */ true> first_input_address_generator = {
        .bank_base_address = first_input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> second_input_address_generator = {
        .bank_base_address = second_input_address, .page_size = tile_bytes, .data_format = data_format};

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        // calculate the address of the first tile in the row
        // start_row is the number of rows already processed in other cores
        uint32_t idx = (start_row + i) * Wt;  // (take already processed rows + current row)*Wt(number of tiles in row)

        cb_reserve_back(cb_first_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_first_input);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(idx + j, first_input_address_generator, l1_write_addr);
            l1_write_addr += tile_bytes;
            noc_async_read_barrier();
        }
        cb_push_back(cb_first_input, Wt);

        cb_reserve_back(cb_second_input, Wt);
        uint32_t second_l1_write_addr = get_write_ptr(cb_second_input);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(idx + j, second_input_address_generator, second_l1_write_addr);
            second_l1_write_addr += tile_bytes;
            noc_async_read_barrier();
        }
        cb_push_back(cb_second_input, Wt);
    }
}
