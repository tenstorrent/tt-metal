// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "debug/dprint.h"
#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

// export TT_METAL_DPRINT_CORES='(0,0)-(0,3)' in order to see DPRINT messages

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t stick_size = get_arg_val<uint32_t>(1);
    const uint32_t padded_offset_bytes = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr auto s0_args = TensorAccessorArgs<1>();
    const auto s0 = TensorAccessor(s0_args, src_addr, stick_size);
    uint32_t stick_id = start_id;
    cb_reserve_back(cb_id_in0, stick_size);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    DPRINT << "Core (" << (uint32_t)get_absolute_logical_x() << "," << (uint32_t)get_absolute_logical_y() << "): ";
    constexpr uint32_t element_per_stick = 2;  // Each stick contains two bfloat16 values
    const uint32_t n_sticks = stick_size / element_per_stick;
    for (uint32_t i = 0; i < n_sticks; i++) {
        uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
        // Read a tick at a time from the source address and write it to the L1 write address.
        noc_async_read(src_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();  // wait for the read to finish
        // We are reading 32 bits at a time, so we can read two bfloat16 values and print
        uint16_t* read_ptr_bf16 = (uint16_t*)l1_write_addr;
        DPRINT << BF16(read_ptr_bf16[0]) << " ";
        DPRINT << BF16(read_ptr_bf16[1]) << " ";
        stick_id++;
        l1_write_addr += padded_offset_bytes;
    }
    DPRINT << ENDL();
    cb_push_back(cb_id_in0, stick_size);

    // At this point we have read all the sticks into the circular buffer. Computation and proceed knowing
    // that it has data in circular buffer and in a specific format.
}
