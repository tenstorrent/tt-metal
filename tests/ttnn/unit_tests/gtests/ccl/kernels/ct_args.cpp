// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include "dataflow_api.h"

template <size_t index, size_t n_vals>
void ct_unrolled_loop(volatile uint32_t* buffer_ptr) {
    *buffer_ptr[index] = kernel_compile_time_arg(index);
    if constexpr (index < n_vals - 1) {
        ct_unrolled_loop<index + 1, n_vals>(buffer_ptr);
    }
}

void kernel_main() {
    auto cb_id = kernel_compile_time_args(0);
    auto n_vals = kernel_compile_time_arg(1);
    cb_reserve_back(cb_id, 1);  // Assume page size == n_vals * sizeof(val);
    auto buffer_ptr = get_write_ptr(cb_id);
    ct_unrolled_loop<0, n_vals>(buffer_ptr);

    noc_async_write(
        buffer_ptr,
        get_noc_addr(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)),
        n_vals * sizeof(uint32_t));  // assume val == uint32
}
