// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t cb_output_idx = tt::CBIndex::c_4;

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    const InterleavedAddrGen<true> output_addr_gen = {.bank_base_address = output_addr, .page_size = tile_bytes};

    // Wait for output tile
    cb_wait_front(cb_output_idx, 1);

    // Write output tile to DRAM
    uint32_t l1_read_addr = get_read_ptr(cb_output_idx);
    noc_async_write_tile(0, output_addr_gen, l1_read_addr);
    noc_async_write_barrier();

    // Pop output
    cb_pop_front(cb_output_idx, 1);
}
