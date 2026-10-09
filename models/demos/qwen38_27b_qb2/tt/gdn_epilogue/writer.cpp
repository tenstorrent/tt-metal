// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = first + item * stride;
        cb_wait_front(7, 4);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            noc_async_write_tile(head * 4 + tile, output, get_read_ptr(7) + tile * 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(7, 4);
    }
}
