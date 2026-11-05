// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Runtime arguments to write data back into the output tensor.
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Nt = get_arg(args::Nt);

    DataflowBuffer dfb_out0(dfb::out0);

    // Create the address generator for the output tensor. Due to us sharing tensor and dataflow buffer
    // configuration parameters (e.g. same data type and same page size) in the host code, we can grab
    // the same parameters from the dataflow buffer as we would from the DRAM tensor.
    // The output tensor is bound to this kernel by name in the host code.
    const auto s = TensorAccessor(tensor::out0);

    // Loop through the matrix dimensions Mt and Nt. bmm will generate C's tiles C=A*B, MN=MK*KN,
    // in row major order, we just read them from the DFB and write out to DRAM
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n = 0; n < Nt; ++n) {
            // Wait for the matrix multiplication kernel to produce an output
            dfb_out0.wait_front(1);
            // Write the output tile to DRAM.
            uint32_t l1_read_addr = dfb_out0.get_read_ptr();
            noc_async_write_page(m * Nt + n, s, l1_read_addr);
            noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                        // noc_async_write_flushed() can be faster because it waits
                                        // until the write request is sent. In that case, you have to
                                        // use noc_async_write_barrier() at least once at the end of
                                        // data movement kernel to make sure all writes are done.
            dfb_out0.pop_front(1);
        }
    }
}
