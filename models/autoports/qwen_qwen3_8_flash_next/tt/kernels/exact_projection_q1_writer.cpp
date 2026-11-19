// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto sa = TensorAccessorArgs<1>();
    const auto scratch = TensorAccessor(sa, get_arg_val<uint32_t>(0), 128);
    const uint32_t first = get_arg_val<uint32_t>(1), stride = get_arg_val<uint32_t>(2);
    const uint32_t groups = get_arg_val<uint32_t>(3);
    cb_reserve_back(5, 1);
    const uint32_t address = get_write_ptr(5);
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t g = first; g < groups; g += stride) {
        cb_wait_front(4, 1);
        const volatile tt_l1_ptr uint32_t* result =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_read_ptr(4));
        for (uint32_t i = 0; i < 32; ++i) {
            stage[i] = 0;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            stage[r] = result[r * 16];
        }
        noc_async_write(address, scratch.get_noc_addr(g), 128);
        noc_async_write_barrier();
        cb_pop_front(4, 1);
    }
}
