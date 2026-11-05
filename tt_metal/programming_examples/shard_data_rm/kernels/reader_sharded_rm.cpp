// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "api/debug/dprint.h"
#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// export TT_METAL_DPRINT_CORES='(0,0)-(0,3)' in order to see DPRINT messages

void kernel_main() {
    const uint32_t stick_size = get_arg(args::stick_size);
    const uint32_t padded_offset_bytes = get_arg(args::padded_offset_bytes);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_in0(dfb::in0);
    // The source tensor is bound to this kernel by name in the host code.
    const auto s0 = TensorAccessor(tensor::src);
    uint32_t stick_id = start_id;
    dfb_in0.reserve_back(stick_size);
    uint32_t l1_write_addr = dfb_in0.get_write_ptr();

    DPRINT("Core ({},{}): ", get_absolute_logical_x(), get_absolute_logical_y());
    constexpr uint32_t element_per_stick = 2;  // Each stick contains two bfloat16 values
    const uint32_t n_sticks = stick_size / element_per_stick;
    for (uint32_t i = 0; i < n_sticks; i++) {
        uint64_t src_noc_addr = s0.get_noc_addr(stick_id);
        // Read a tick at a time from the source address and write it to the L1 write address.
        noc_async_read(src_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();  // wait for the read to finish
        // We are reading 32 bits at a time, so we can read two bfloat16 values and print
        uint16_t* read_ptr_bf16 = (uint16_t*)l1_write_addr;
        DPRINT("{} {} ", bf16_t(read_ptr_bf16[0]), bf16_t(read_ptr_bf16[1]));
        stick_id++;
        l1_write_addr += padded_offset_bytes;
    }
    DPRINT("\n");
    dfb_in0.push_back(stick_size);

    // At this point we have read all the sticks into the dataflow buffer. Computation and proceed knowing
    // that it has data in dataflow buffer and in a specific format.
}
