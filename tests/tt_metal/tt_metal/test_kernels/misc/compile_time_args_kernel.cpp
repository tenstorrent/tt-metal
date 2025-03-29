// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

template <size_t N>
constexpr uint32_t sum(const std::array<uint32_t, N>& array) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        sum += array[i];
    }
    return sum;
}

void kernel_main() {
    uint32_t compile_time_args_sum = sum(kernel_compile_time_args);
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;
    *l1_ptr = compile_time_args_sum;
}
