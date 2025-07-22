// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // Runtime args
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t src1_cb_index = get_compile_time_arg_val(0);
    constexpr bool output_is_dram = get_compile_time_arg_val(1) == 1;

    // Constants
    constexpr uint32_t one_tile = 1;

    // Output data config
    constexpr uint32_t output_tensor_tile_size_bytes = get_tile_size(src1_cb_index);
    constexpr DataFormat output_tensor_data_format = get_dataformat(src1_cb_index);
    const InterleavedAddrGenFast<output_is_dram> output_tensor_dram = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = output_tensor_tile_size_bytes,
        .data_format = output_tensor_data_format};

    // Wait for incoming data from reader1
    cb_wait_front(src1_cb_index, one_tile);
    DPRINT << "7. WRITER 1: Received data" << ENDL();
    const uint32_t l1_write_addr_output = get_read_ptr(src1_cb_index);

    // Print data in buffer
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_output);
    DPRINT << " > Received data from src1: " << U32(ptr[0]) << ENDL();

    // Save output data
    noc_async_write_tile(0, output_tensor_dram, l1_write_addr_output);
    noc_async_write_barrier();

    cb_pop_front(src1_cb_index, one_tile);
    DPRINT << "8. WRITER 1: Data saved" << ENDL();
}
