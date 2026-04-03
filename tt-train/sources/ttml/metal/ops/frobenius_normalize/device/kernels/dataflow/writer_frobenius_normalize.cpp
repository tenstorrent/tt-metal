// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t rt = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt++);
    uint32_t num_tiles = get_arg_val<uint32_t>(rt++);
    uint32_t start_tile_id = get_arg_val<uint32_t>(rt++);

    constexpr uint32_t cb_output = tt::CBIndex::c_5;

    const uint32_t tile_bytes = get_tile_size(cb_output);
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output_addr_gen = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_output, 1);
        uint32_t l1_addr = get_read_ptr(cb_output);
        noc_async_write_page(start_tile_id + i, output_addr_gen, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output, 1);
    }
}
