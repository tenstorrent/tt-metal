// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"  // get_arg_val
#endif

// A function that prints, called from two places: DEVICE_PRINT_FORCE_NOINLINE keeps one copy of it while printing is
// on.
DEVICE_PRINT_FORCE_NOINLINE void print_uint_arg(uint32_t x) { DEVICE_PRINT("Printing uint32_t from arg: {}\n", x); }

void kernel_main() {
    uint32_t x = get_arg_val<uint32_t>(0);
    print_uint_arg(x);
    print_uint_arg(x + 1);
}
