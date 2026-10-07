// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(0), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1), 512);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    // CB10 is writer-private scratch, not shared with the reader/compute.
    const uint32_t scratch = get_write_ptr(10);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = first + item * stride;
        cb_wait_front(8, 4);  // proves compute has finished reading S_new
        cb_wait_front(7, 16);
        const uint32_t state_l1 = get_read_ptr(7);
        for (uint32_t tile = 0; tile < 16; ++tile) {
            noc_async_write_tile(head * 16 + tile, state, state_l1 + tile * 4096);
        }
        const auto* tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(8));
        auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t i = 0; i < 128; ++i) {
            row[i] = tiles[(i / 32) * 1024 + ((i % 32) / 16) * 256 + i % 16];
        }
        noc_async_write_page(head, output, scratch);
        noc_async_write_barrier();
        cb_pop_front(7, 16);
        cb_pop_front(8, 4);
    }
}
