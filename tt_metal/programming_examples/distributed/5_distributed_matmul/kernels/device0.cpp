// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    auto args = TensorAccessorArgs<0>{};
    uint32_t in1_addr = get_arg_val<uint32_t>(0);
    uint32_t in2_addr = get_arg_val<uint32_t>(1);
    uint32_t out_addr = get_arg_val<uint32_t>(2);
    uint32_t page_size = get_arg_val<uint32_t>(3);
    auto input1_accessor = TensorAccessor(args, in1_addr, page_size);
    auto input2_accessor = TensorAccessor(args, in2_addr, page_size);
    auto output_accessor = TensorAccessor(args, out_addr, page_size);

    uint32_t in1_cb_addr = get_write_ptr(0);
    uint32_t in2_cb_addr = get_write_ptr(1);
    uint32_t out_cb_addr = get_write_ptr(2);

    float* in1_ptr = reinterpret_cast<float*>(in1_cb_addr);
    float* in2_ptr = reinterpret_cast<float*>(in2_cb_addr);
    float* out_ptr = reinterpret_cast<float*>(out_cb_addr);

    for (int count = 0; count < 32; count++) {
        noc_async_read(input1_accessor.get_noc_addr(count), in1_cb_addr, page_size);
        noc_async_read(input2_accessor.get_noc_addr(count), in2_cb_addr, page_size);
        noc_async_read_barrier();
        noc_async_write_barrier();
        for (int i = 0; i < 64; i++) {
            out_ptr[i] = in1_ptr[i] + in2_ptr[i];
        }
        noc_async_write(out_cb_addr, output_accessor.get_noc_addr(count), page_size);
    }
}
