// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug_print.h"

/**
 * add two ints
 * args are in L1
 * result is in L1
*/

void kernel_main() {

    volatile tt_l1_ptr std::uint32_t* arg_a = (volatile tt_l1_ptr uint32_t*)(L1_ARG_BASE);
    volatile tt_l1_ptr std::uint32_t* arg_b = (volatile tt_l1_ptr uint32_t*)(L1_ARG_BASE + 4);
    volatile tt_l1_ptr std::uint32_t* result = (volatile tt_l1_ptr uint32_t*)(L1_RESULT_BASE);

    //Sample print statement
    // DPRINT << 123;
    result[0] = arg_a[0] + arg_b[0];

}
