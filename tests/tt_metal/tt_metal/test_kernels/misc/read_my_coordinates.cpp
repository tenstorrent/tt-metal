// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Test kernel reads my_x, my_y. my_logical_x, my_logical_y, my_sub_device_x, and my_sub_device_y
//
// compile time arg 0: where to write the values that were read in the order above
// required space = 24B
//
// Applicable for Data Movement cores only
//

#include "hw/inc/dataflow_api.h"

void kernel_main() {
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
    results[0] = my_x[noc_index];
    results[1] = my_y[noc_index];
    results[2] = get_absolute_logical_x();
    results[3] = get_absolute_logical_y();
    results[4] = get_relative_logical_x();
    results[5] = get_relative_logical_y();
}
