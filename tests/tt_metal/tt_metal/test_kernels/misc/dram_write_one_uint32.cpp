// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"

void kernel_main() {
    constexpr uint32_t l1_address = get_compile_time_arg_val(0);
    constexpr uint32_t value = get_compile_time_arg_val(1);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<tt_l1_ptr uint32_t*>(l1_address);
    result[0] = value;
}
