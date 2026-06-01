// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole variant of the latency receiver. Same protocol as the Quasar receiver.cpp, but
// L1 is accessed directly (no MEM_L1_UNCACHED_BASE alias) and the poll loop invalidates the
// L1 cache each spin so incoming NOC writes become visible. Compile-time argument order is
// identical to receiver.cpp.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"

void kernel_main() {
    constexpr uint32_t data_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(1);
    constexpr uint32_t my_noc_x = get_compile_time_arg_val(2);  // used only for diagnostic print
    constexpr uint32_t my_noc_y = get_compile_time_arg_val(3);  // used only for diagnostic print

    DEVICE_PRINT("[receiver] running on core ({},{}), waiting for {} writes\n", my_noc_x, my_noc_y, num_iterations);

    volatile tt_l1_ptr uint32_t* data_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_l1_addr);

    for (uint32_t i = 0; i < num_iterations; i++) {
        while (*data_ptr != i + 1) {
            invalidate_l1_cache();
        }
        DEVICE_PRINT("[receiver] ({},{}) got write {}\n", my_noc_x, my_noc_y, i + 1);
    }

    DEVICE_PRINT("[receiver] core ({},{}) received all {} writes\n", my_noc_x, my_noc_y, num_iterations);
}
