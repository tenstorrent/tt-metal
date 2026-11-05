// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// writes sticks from out0 with padded alignment stepping
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    uint32_t num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_out0(dfb::out0);
    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr uint32_t stick_size_padded_aligned = get_arg(args::stick_size_padded_aligned);

    const auto s = TensorAccessor(tensor::dst);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        dfb_out0.wait_front(num_sticks_per_barrier);
        uint32_t l1_read_addr = dfb_out0.get_read_ptr();
        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            uint64_t write_noc_addr = s.get_noc_addr(i_stick);
            noc_async_write(l1_read_addr, write_noc_addr, stick_size_bytes);
            l1_read_addr += stick_size_padded_aligned;
            i_stick += 1;
        }
        noc_async_write_barrier();
        dfb_out0.pop_front(num_sticks_per_barrier);
    }
}
