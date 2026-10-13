// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t copy_addr = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t input_pages = get_arg_val<uint32_t>(4);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer weights(tt::CBIndex::c_1);
    CircularBuffer copy_input(tt::CBIndex::c_2);
    const uint32_t input_bytes = input.get_tile_size() * input_pages;
    const uint32_t weights_bytes = weights.get_tile_size();
    const uint32_t copy_bytes = copy_input.get_tile_size();

    for (uint32_t i = 0; i < count; ++i) {
        input.reserve_back(input_pages);
        weights.reserve_back(1);
        noc.async_read(dram_src, input, input_bytes, {.bank_id = 0, .addr = input_addr + i * input_bytes}, {});
        noc.async_read(dram_src, weights, weights_bytes, {.bank_id = 0, .addr = weights_addr + i * weights_bytes}, {});
        noc.async_read_barrier();
        input.push_back(input_pages);
        weights.push_back(1);
    }
    for (uint32_t i = 0; i < count; ++i) {
        copy_input.reserve_back(1);
        noc.async_read(dram_src, copy_input, copy_bytes, {.bank_id = 0, .addr = copy_addr + i * copy_bytes}, {});
        noc.async_read_barrier();
        copy_input.push_back(1);
    }
}
