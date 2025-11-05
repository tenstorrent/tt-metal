// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    // Runtime arguments
    uint32_t runtime_args_counter = 0;
    uint32_t output_address = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices
    constexpr uint32_t cb_output = tt::CBIndex::c_4;

    // Get tile size
    const uint32_t tile_bytes = get_tile_size(cb_output);

    // Setup tensor accessor for reading input
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output_address_generator = TensorAccessor(output_args, output_address, tile_bytes);

    constexpr uint32_t onetile = 1U;
    cb_wait_front(cb_output, onetile);
    uint32_t l1_read_addr = get_read_ptr(cb_output);
    noc_async_write_tile(0, output_address_generator, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_output, onetile);
}
