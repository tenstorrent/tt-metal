// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t core_index = get_arg_val<uint32_t>(0);
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t iter_count = get_arg_val<uint32_t>(2);
    uint32_t num_cores = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id);
    constexpr uint32_t tile_size_pow2_exponent = 11;
    const InterleavedPow2AddrGen<false> s = {
        .bank_base_address = l1_buffer_addr, .log_base_2_of_page_size = tile_size_pow2_exponent};

    uint32_t cb_addr;
    cb_reserve_back(cb_id, 1);
    cb_addr = get_write_ptr(cb_id);

    for (uint32_t i = 0; i < iter_count; i++) {
        uint32_t i_256 = i & 0xFF;
        uint64_t l1_buffer_noc_addr = get_noc_addr(i_256 * num_cores + core_index, s);
        noc_async_write(cb_addr, l1_buffer_noc_addr, single_tile_size_bytes);
        noc_async_write_barrier();
    }
}
