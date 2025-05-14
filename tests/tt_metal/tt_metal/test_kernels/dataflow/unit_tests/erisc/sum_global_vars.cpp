// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

uint16_t global_array[2] __attribute__((used)) = {39, 10};

void kernel_main() {
    constexpr uint32_t result_addr = get_compile_time_arg_val(0);

    uint32_t result = global_array[0] + global_array[1];

    volatile tt_l1_ptr uint32_t* result_addr_ptr = (volatile tt_l1_ptr uint32_t*)(result_addr);
    result_addr_ptr[0] = result;
}
