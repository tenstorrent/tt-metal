// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments to write data back into the output buffer.
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = 16;

    // Create the address generator for the output buffer. Due to us sharing buffer and circular buffer
    // configuration parameters (e.g. same data type and same page size) in the host code, we can grab
    // the same parameters from the circular buffer as we would from the DRAM buffer.
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr, get_tile_size(cb_id_out0));

    // Loop through the matrix dimensions Mt and Nt. bmm will generate C's tiles C=A*B, MN=MK*KN,
    // in row major order, we just read them from CB and write out to DRAM
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n = 0; n < Nt; ++n) {
            // Wait for the matrix multiplication kernel to produce an output
            cb_wait_front(cb_id_out0, 1);
            // Write the output tile to DRAM.
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            noc_async_write_tile(m * Nt + n, s, l1_read_addr);
            noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                        // noc_async_write_flushed() can be faster because it waits
                                        // until the write request is sent. In that case, you have to
                                        // use noc_async_write_barrier() at least once at the end of
                                        // data movement kernel to make sure all writes are done.
            cb_pop_front(cb_id_out0, 1);
        }
    }
}
