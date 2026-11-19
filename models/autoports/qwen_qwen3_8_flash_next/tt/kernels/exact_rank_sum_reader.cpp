// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    for (uint32_t tile = start; tile < start + count; ++tile) {
        cb_reserve_back(0, 4);
        for (uint32_t rank = 0; rank < 4; ++rank) {
            noc_async_read(input.get_noc_addr(rank * 80 + tile), get_write_ptr(0) + rank * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(0, 4);
    }
}
