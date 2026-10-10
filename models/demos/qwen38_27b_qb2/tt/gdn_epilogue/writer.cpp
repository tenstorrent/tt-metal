// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr bool compact = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t heads = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr auto oa = TensorAccessorArgs<3>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    uint32_t zero_row = 0;
    if constexpr (compact) {
        zero_row = get_write_ptr(12);
        auto* zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_row);
        for (uint32_t word = 0; word < 8; ++word) {
            zero[word] = 0;
        }
    }
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = first + item * stride;
        cb_wait_front(7, 4);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            if constexpr (compact) {
                const uint32_t user = head / heads;
                const uint32_t page = (head % heads) * 4 + tile;
                const uint32_t row = (user / 16) * 1024 + (user % 16) * 32;
                for (uint32_t face = 0; face < 2; ++face) {
                    // Blackhole writes require matching low four bits. Each
                    // worker owns an independent 32-byte face row; no tile
                    // read-modify-write can overwrite another user's result.
                    noc_async_write(
                        get_read_ptr(7) + tile * 2048 + face * 512, output.get_noc_addr(page, row + face * 512), 32);
                    if (user == batch - 1) {
                        // The final live user of each head owns only the
                        // padding rows, disjoint from every live writer.
                        for (uint32_t pad = batch; pad < 32; ++pad) {
                            const uint32_t pad_row = (pad / 16) * 1024 + (pad % 16) * 32;
                            noc_async_write(zero_row, output.get_noc_addr(page, pad_row + face * 512), 32);
                        }
                    }
                }
            } else {
                noc_async_write_tile(head * 4 + tile, output, get_read_ptr(7) + tile * 2048);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(7, 4);
    }
}
