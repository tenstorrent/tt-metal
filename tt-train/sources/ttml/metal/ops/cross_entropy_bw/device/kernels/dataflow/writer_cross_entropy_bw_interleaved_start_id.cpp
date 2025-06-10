// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_target_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_output_idx = tt::CBIndex::c_10;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // number of tiles in inner dimension
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(2);

    constexpr uint32_t onetile = 1U;

    const float scaler = uint32_to_float(scaler_bits);
    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    const DataFormat data_format = get_dataformat(cb_output_idx);

    const InterleavedAddrGenFast</* is dram */ true> output_addr_generator = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t end_row = start_row + num_rows_to_process;

    uint32_t indices[TILE_HEIGHT];

    for (uint32_t r = start_row; r < end_row; r++) {
        cb_wait_front(cb_target_idx, onetile);
        auto target_indexes_l1_ptr = reinterpret_cast<uint32_t *>(get_read_ptr(cb_target_idx));

        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            indices[i] = i;
        }
        std::sort(
            indices, indices + TILE_HEIGHT, [&target_indexes_l1_ptr](const uint32_t &idx_1, const uint32_t &idx_2) {
                return target_indexes_l1_ptr[idx_1] < target_indexes_l1_ptr[idx_2];
            });

        uint32_t target_indices_idx = 0;

        for (uint32_t c = 0, idx = r * Wt; c < Wt; c += block_size) {
            cb_wait_front(cb_output_idx, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_output_idx);

            auto write_output_l1_ptr = reinterpret_cast<uint16_t *>(l1_read_addr);

            while (target_indices_idx < TILE_HEIGHT) {
                uint32_t h = indices[target_indices_idx];
                uint32_t target_value = target_indexes_l1_ptr[h];
                uint32_t tile_idx = target_value / TILE_WIDTH;
                if (tile_idx >= c + block_size) {
                    break;
                }

                uint32_t local_tile_idx = tile_idx - c;
                uint32_t index_inside_tile =
                    (TILE_WIDTH * TILE_HEIGHT * local_tile_idx) + get_tilized_idx(h, target_value);

                float value = bfloat16_to_float(write_output_l1_ptr[index_inside_tile]);
                value -= scaler;
                write_output_l1_ptr[index_inside_tile] = float_to_bfloat16(value);
                ++target_indices_idx;
            }

            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++idx) {
                noc_async_write_tile(idx, output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_idx, block_size);
        }

        cb_pop_front(cb_target_idx, onetile);
    }
}
