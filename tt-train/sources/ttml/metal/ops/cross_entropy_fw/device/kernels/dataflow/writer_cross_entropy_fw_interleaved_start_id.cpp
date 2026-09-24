// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_output_idx = tt::CBIndex::c_11;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // number of tiles in inner dimension

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    constexpr auto output_args = TensorAccessorArgs<2>();
    const auto output_addr_generator = TensorAccessor(output_args, output_addr);

    uint32_t end_row = start_row + num_rows_to_process;

    for (uint32_t r = start_row; r < end_row; r++) {
        cb_wait_front(cb_output_idx, onetile);
        auto output_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(cb_output_idx));
        for (uint32_t h = 0; h < TILE_HEIGHT; ++h) {
            for (uint32_t w = 1; w < TILE_WIDTH; ++w) {
                output_l1_ptr[get_tilized_idx(h, w)] = 0U;
            }
        }
        write_tiles_by_row(cb_output_idx, output_addr_generator, r, onetile, tile_bytes, onetile);
    }
}
