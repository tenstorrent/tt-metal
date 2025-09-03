// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t invalid_arg = get_named_compile_time_arg_val("invalid_arg");

    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    l1_ptr[0] = invalid_arg;
}
