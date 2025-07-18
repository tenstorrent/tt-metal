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
    const uint32_t shard_size = get_arg_val<uint32_t>(1);
    const uint32_t padded_offset_bytes = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t element_per_stick = get_compile_time_arg_val(3);
    static_assert(stick_size == sizeof(uint16_t) * 2, "stick_size must be 8 bytes for two bfloat16 values");

    // Note the use of InterleavedAddrGen as input is not tiled (32x32 grild of values). But in row major. Making
    // InterleavedAddrGenFast unusable (which only supports addressing of tiled data).
    const InterleavedAddrGen<src_is_dram> s0 = {.bank_base_address = src_addr, .page_size = stick_size};
    uint32_t stick_id = start_id;
    cb_reserve_back(cb_id_in0, shard_size);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    DPRINT << "Core (" << (uint32_t)get_absolute_logical_x() << "," << (uint32_t)get_absolute_logical_y() << "): ";
    const uint32_t n_sticks = shard_size / element_per_stick;
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
    cb_push_back(cb_id_in0, shard_size);

    // At this point we have read all the sticks into the circular buffer. Computation and proceed knowing
    // that it has data in circular buffer and in a specific format.
}
