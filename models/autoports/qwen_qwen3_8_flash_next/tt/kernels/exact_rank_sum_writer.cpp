// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0), 4096);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    for (uint32_t tile = start; tile < start + count; ++tile) {
        cb_wait_front(1, 1);
        noc_async_write(get_read_ptr(1), output.get_noc_addr(tile), 4096);
        noc_async_write_barrier();
        cb_pop_front(1, 1);
    }
}
