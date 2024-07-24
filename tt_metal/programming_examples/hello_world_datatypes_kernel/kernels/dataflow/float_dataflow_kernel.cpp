// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {

        // Fetch the data from on-device memory and debug print.
        // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

        // Copy float from device DRAM into Core 0,0's L1
        uint32_t dram_addr  = get_arg_val<uint32_t>(0);
        uint64_t noc_addr = get_noc_addr(1, 0, dram_addr);
        constexpr uint32_t cb_id = tt::CB::c_in0; // index=0
        uint32_t size = get_tile_size(cb_id);
        uint32_t l1_addr= get_write_ptr(cb_id);
        cb_reserve_back(cb_id, 0);
        noc_async_read(noc_addr, l1_addr, size);
        noc_async_read_barrier();

        float* data = (float*) l1_addr;
        DPRINT << "Master, I have retrieved the value stored on Device 0 DRAM. Here we go.  It is: " << F32(*data) << ENDL();

        cb_push_back(cb_id, 0);

}
