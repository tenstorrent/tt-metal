// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t bank_base_address = get_arg_val<uint32_t>(0);
    uint32_t page_size = get_arg_val<uint32_t>(1);
    uint32_t dst_l1_addr = get_arg_val<uint32_t>(2);
    uint32_t num_pages_to_read = get_arg_val<uint32_t>(3);
    uint32_t num_iterations = get_arg_val<uint32_t>(4);

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, bank_base_address, page_size);

    for (uint32_t iter_idx = 0; iter_idx < num_iterations; iter_idx++) {
        for (uint32_t page_id = 0; page_id < num_pages_to_read; page_id++) {
            noc_async_read_tile(page_id, s, dst_l1_addr);
        }
    }

    noc_async_read_barrier();
}
