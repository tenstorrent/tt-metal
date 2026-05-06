// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: TensorAccessor loopback consumer using a Metal 2.0 TensorAccessor binding.
// Pops entries from a DFB bound via dfb::input_dfb and writes them as pages to an output tensor
// via make_tensor_accessor(ta::output_tensor). The base address comes from the binding's
// implicit __ta_addr_output_tensor CRTA, filled by SetProgramRunParameters from TensorRunParams.
//
// Runtime args:
//   arg 0: number of pages to transfer

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);

    auto accessor = make_tensor_accessor(ta::output_tensor);
    experimental::DataflowBuffer buf(dfb::input_dfb);
    uint32_t entry_size = buf.get_entry_size();

    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {
        buf.wait_front(1);
        uint64_t dst_noc_addr = accessor.get_noc_addr(page_id);
        noc_async_write(buf.get_read_ptr(), dst_noc_addr, entry_size);
        noc_async_write_barrier();
        buf.pop_front(1);
    }
}
