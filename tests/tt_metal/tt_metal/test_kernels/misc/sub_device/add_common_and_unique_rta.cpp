// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

/**
 * add two ints
 * args are in L1
 * result is in L1
 */

void kernel_main() {
    tt_l1_ptr std::uint32_t* arg_a = (tt_l1_ptr uint32_t*)get_arg_addr(0);
    tt_l1_ptr std::uint32_t* arg_b = (tt_l1_ptr uint32_t*)get_common_arg_addr(0);
    constexpr uint32_t l1_address = get_compile_time_arg_val(0);

    volatile tt_l1_ptr std::uint32_t* result = (tt_l1_ptr uint32_t*)(l1_address);

    // Sample print statement
    //  DPRINT << 123;
    result[0] = arg_a[0] + arg_b[0];
}
