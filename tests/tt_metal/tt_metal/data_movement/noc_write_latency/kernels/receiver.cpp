// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "dev_mem_map.h"

void kernel_main() {
    constexpr uint32_t data_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(1);
    constexpr uint32_t my_noc_x = get_compile_time_arg_val(2);  // used only for diagnostic print
    constexpr uint32_t my_noc_y = get_compile_time_arg_val(3);  // used only for diagnostic print

    DEVICE_PRINT("[receiver] running on core ({},{}), waiting for {} writes\n", my_noc_x, my_noc_y, num_iterations);

    // The sender writes the iteration count (i+1) as the first word of the payload, so we poll
    // the data location itself — no separate flag needed.
    volatile tt_l1_ptr uint32_t* data_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_l1_addr + MEM_L1_UNCACHED_BASE);

    for (uint32_t i = 0; i < num_iterations; i++) {
        while (*data_ptr != i + 1) {
        }
        DEVICE_PRINT("[receiver] ({},{}) got write {}\n", my_noc_x, my_noc_y, i + 1);
    }

    DEVICE_PRINT("[receiver] core ({},{}) received all {} writes\n", my_noc_x, my_noc_y, num_iterations);
}
