// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto pa = TensorAccessorArgs<0>();
    const auto partials = TensorAccessor(pa, get_arg_val<uint32_t>(0), 4096);
    const uint32_t column = get_arg_val<uint32_t>(1);
    const uint32_t blocks = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in = 0;
    for (uint32_t block = 0; block < blocks; ++block) {
        cb_reserve_back(cb_in, 1);
        noc_async_read(partials.get_noc_addr(block * 20 + column), get_write_ptr(cb_in), 4096);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
