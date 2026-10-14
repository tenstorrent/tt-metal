// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dot-product finish, reader (one core): the pages holding the per-core values (64-byte slots), VB per read batch.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t part_addr = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    constexpr auto ap = TensorAccessorArgs<0>();
    const auto PART = TensorAccessor(ap, part_addr, 4096);
    for (uint32_t t = 0; t < n; t += VB) {
        const uint32_t k = (n - t) < VB ? (n - t) : VB;
        cb_reserve_back(CB_A, k);
        const uint32_t base = get_write_ptr(CB_A);
        for (uint32_t i = 0; i < k; ++i) {
            noc_async_read(PART.get_noc_addr(t + i), base + i * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(CB_A, k);
    }
}
