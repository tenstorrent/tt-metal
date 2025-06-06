// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_output = tt::CBIndex::c_2;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);  // number of tiles in inner dimension

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const DataFormat data_format = get_dataformat(cb_output);

    const InterleavedAddrGenFast</* is dram */ true> output_addr_generator = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t end_row = start_row + num_rows_to_process;

    cb_wait_front(cb_output, Wt);
    uint32_t l1_read_addr = get_read_ptr(cb_output);
    for (uint32_t r = start_row; r < end_row; r++) {
        for (uint32_t c = 0, idx = r * Wt; c < Wt; c++) {
            noc_async_write_tile(idx, output_addr_generator, l1_read_addr);
            l1_read_addr += tile_bytes;
            noc_async_write_barrier();
        }
    }
    cb_pop_front(cb_output, Wt);
}
