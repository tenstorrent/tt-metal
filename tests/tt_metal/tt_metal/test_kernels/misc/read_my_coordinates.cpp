// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Test kernel reads my_x, my_y. my_logical_x, my_logical_y, my_sub_device_x, and my_sub_device_y
//
// compile time arg 0: where to write the values that were read in the order above
// required space = 24B
//

#include "compile_time_args.h"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC)
#define ANY_KERNEL_BEGIN void kernel_main() {
#define ANY_KERNEL_END }
#else
#include "compute_kernel_api/common.h"
#define ANY_KERNEL_BEGIN  \
    namespace NAMESPACE { \
    void MAIN {
#define ANY_KERNEL_END \
    }                  \
    }
#endif

ANY_KERNEL_BEGIN

volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(get_compile_time_arg_val(0));

#ifndef COMPILE_FOR_TRISC
results[0] = my_x[noc_index];
results[1] = my_y[noc_index];
#endif

results[2] = my_logical_x;
results[3] = my_logical_y;
results[4] = my_sub_device_x;
results[5] = my_sub_device_y;

ANY_KERNEL_END
