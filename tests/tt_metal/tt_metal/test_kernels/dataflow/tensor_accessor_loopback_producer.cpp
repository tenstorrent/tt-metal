// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: TensorAccessor loopback producer using a Metal 2.0 TensorBinding.
// Reads pages from an input tensor via TensorAccessor(tensor::input_tensor) and pushes them
// entry by entry into a DFB bound via dfb::input_dfb. The base address comes from the binding's
// slot in the kernel's TensorBinding address section, filled by SetProgramRunArgs from
// TensorArgument.
//
// Runtime args:
//   arg 0: number of pages to transfer

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);

    TensorAccessor accessor(tensor::input_tensor);
    DataflowBuffer buf(dfb::input_dfb);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {
        buf.reserve_back(1);
        uint64_t src_noc_addr = accessor.get_noc_addr(page_id);
        noc_async_read(src_noc_addr, buf.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        buf.push_back(1);
    }
}
