// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "dev_mem_map.h"

// DeviceTimestampedData/DeviceZoneScopedN are not available on Quasar emulator;
// use inline rdcycle CSR read instead.
static inline uint32_t rdcycles() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

void kernel_main() {
    constexpr uint32_t src_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_l1_data_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dst_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t dst_noc_y = get_compile_time_arg_val(3);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(4);
    constexpr uint32_t transaction_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t src_noc_x = get_compile_time_arg_val(6);
    constexpr uint32_t src_noc_y = get_compile_time_arg_val(7);

    DEVICE_PRINT(
        "[sender] running on core ({},{}), writing {} bytes to core ({},{}) for {} iterations\n",
        src_noc_x,
        src_noc_y,
        transaction_size_bytes,
        dst_noc_x,
        dst_noc_y,
        num_iterations);

    uint64_t dst_data_noc = get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_data_addr);

    for (uint32_t i = 0; i < num_iterations; i++) {
        // The first word of the payload carries the iteration count, which doubles as the
        // signal the receiver polls for. Written via the uncached alias so the NOC read picks
        // up the new value.
        *(volatile tt_l1_ptr uint32_t*)(src_l1_addr + MEM_L1_UNCACHED_BASE) = i + 1;

        noc_async_write(src_l1_addr, dst_data_noc, transaction_size_bytes);

        // Measures time for the write to retire on the sender side (write ACK received).
        // t0 is placed after noc_async_write() — the write may already be in-flight.
        uint32_t t0 = rdcycles();
        noc_async_write_barrier();
        uint32_t t1 = rdcycles();

        DEVICE_PRINT(
            "[sender] ({},{}) -> ({},{}) iter {}: {} cycles\n", src_noc_x, src_noc_y, dst_noc_x, dst_noc_y, i, t1 - t0);
    }
}
