// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0), 512);
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1), 512);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    // Private scratch, never published through CB tokens.
    const uint32_t scratch = get_write_ptr(9);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = first + item * stride;
        cb_reserve_back(0, 4);
        cb_reserve_back(1, 4);
        noc_async_read_page(head, q, scratch);
        noc_async_read_page(head, k, scratch + 512);
        noc_async_read_barrier();
        const auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        auto* qc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
        auto* kc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t tile = i / 32;
            const uint32_t lane = i % 32;
            const uint32_t column = tile * 1024 + (lane / 16) * 512 + (lane % 16) * 16;
            qc[column] = values[i];
            kc[column] = values[128 + i];
        }
        cb_push_back(0, 4);
        cb_push_back(1, 4);
    }
}
