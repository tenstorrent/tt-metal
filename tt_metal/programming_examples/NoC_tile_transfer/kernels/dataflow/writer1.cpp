// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

void kernel_main() {
    // Dataflow buffer bindings
    DataflowBuffer src1_dfb(dfb::src1);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Output data config
    const auto output_tensor_dram = TensorAccessor(tensor::output);

    // Wait for incoming data from reader1
    src1_dfb.wait_front(one_tile);
    DPRINT("7. WRITER 1: Received data\n");
    const uint32_t l1_write_addr_output = src1_dfb.get_read_ptr();

    // Print data in buffer
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_output);
    DPRINT(" > Received data from src1: {}\n", ptr[0]);

    // Save output data
    noc_async_write_page(0, output_tensor_dram, l1_write_addr_output);
    noc_async_write_barrier();

    src1_dfb.pop_front(one_tile);
    DPRINT("8. WRITER 1: Data saved\n");
}
