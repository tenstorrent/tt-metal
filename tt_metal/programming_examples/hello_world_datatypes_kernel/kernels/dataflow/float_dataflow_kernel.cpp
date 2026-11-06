// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Fetch the data from on-device memory and debug print.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    // Copy float from device DRAM into the core's SRAM
    uint32_t dram_addr = get_arg(args::dram_addr);
    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(0, dram_addr);
    DataflowBuffer dfb_src0(dfb::src0);  // bound by name in the host code
    uint32_t size = dfb_src0.get_entry_size();
    uint32_t l1_addr = dfb_src0.get_write_ptr();
    dfb_src0.reserve_back(0);
    noc_async_read(noc_addr, l1_addr, size);
    noc_async_read_barrier();

    // Read floating point value from SRAM and print it
    float* data = (float*)l1_addr;
    DPRINT("Master, I have retrieved the value stored on Device 0 DRAM. Here we go.  It is: {}\n", *data);

    dfb_src0.push_back(0);
}
