// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole variant of the latency sender. Same protocol as the Quasar sender.cpp, but:
//   - timing uses the memory-mapped wall-clock register instead of the rdcycle CSR
//   - L1 is accessed directly (no MEM_L1_UNCACHED_BASE alias)
// Compile-time argument order is identical to sender.cpp so the host can share the layout.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "risc_common.h"

// Blackhole free-running wall-clock low word (core-cycle counter). A single volatile
// register read, matching what the on-device profiler uses.
static inline uint32_t read_timer() {
    return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
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
        // First word of the payload carries the iteration count, which doubles as the signal
        // the receiver polls for.
        *(volatile tt_l1_ptr uint32_t*)(src_l1_addr) = i + 1;

        noc_async_write(src_l1_addr, dst_data_noc, transaction_size_bytes);

        // Measures time for the write to retire on the sender side (write ACK received).
        // t0 is placed after noc_async_write() — the write may already be in-flight.
        uint32_t t0 = read_timer();
        noc_async_write_barrier();
        uint32_t t1 = read_timer();

        DEVICE_PRINT(
            "[sender] ({},{}) -> ({},{}) iter {}: {} cycles\n", src_noc_x, src_noc_y, dst_noc_x, dst_noc_y, i, t1 - t0);
    }
}
