// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t src0_cb_index = get_compile_time_arg_val(0);
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;

    // Input data config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src0_cb_index);
    const DataFormat input_data_format = get_dataformat(src0_cb_index);
    const InterleavedAddrGenFast<input_is_dram> interleaved_accessor = {
        .bank_base_address = input_buffer_addr,
        .page_size = input_data_tile_size_bytes,
        .data_format = input_data_format};

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    DPRINT << "1. READER 0: Reading input data to L1 src0 CB" << ENDL();
    cb_reserve_back(src0_cb_index, one_tile);
    const uint32_t l1_write_addr = get_write_ptr(src0_cb_index);
    noc_async_read_tile(0, interleaved_accessor, l1_write_addr);
    noc_async_read_barrier();

    // Print data in buffer
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
    DPRINT << " > Data in buffer: " << U32(ptr[0]) << ENDL();

    cb_push_back(src0_cb_index, one_tile);
    DPRINT << "2. READER 0: Data in src0 CB pushed from reader0" << ENDL();
}
