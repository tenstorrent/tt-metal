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
    // Private compact output staging. Its contents live through the write barrier.
    const uint32_t scratch = get_write_ptr(10);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = first + item * stride;
        cb_wait_front(11, 4);
        cb_wait_front(12, 4);
        const auto* qc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(11));
        const auto* kc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(12));
        auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t tile = i / 32;
            const uint32_t lane = i % 32;
            const uint32_t column = tile * 1024 + (lane / 16) * 512 + (lane % 16) * 16;
            values[i] = qc[column];
            values[128 + i] = kc[column];
        }
        noc_async_write_page(head, q, scratch);
        noc_async_write_page(head, k, scratch + 512);
        noc_async_write_barrier();
        cb_pop_front(11, 4);
        cb_pop_front(12, 4);
    }
}
