// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

#include <cstdint>

void kernel_main() {
    // Dataflow buffer bindings
    DataflowBuffer src0_dfb(dfb::src0);

    // Input data config
    const auto interleaved_accessor = TensorAccessor(tensor::input);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Read input value data
    DPRINT("1. READER 0: Reading input data to L1 src0 CB\n");
    src0_dfb.reserve_back(one_tile);
    const uint32_t l1_write_addr = src0_dfb.get_write_ptr();
    noc_async_read_page(0, interleaved_accessor, l1_write_addr);
    noc_async_read_barrier();

    // Print data in buffer
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
    DPRINT(" > Data in buffer: {}\n", ptr[0]);

    src0_dfb.push_back(one_tile);
    DPRINT("2. READER 0: Data in src0 CB pushed from reader0\n");
}
