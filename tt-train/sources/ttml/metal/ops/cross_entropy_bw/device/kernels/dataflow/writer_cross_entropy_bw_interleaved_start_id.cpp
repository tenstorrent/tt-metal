// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "tt-train/sources/ttml/metal/ops/common/common_utils.hpp"

// constexpr uint32_t FACE_HEIGHT = 16;
// constexpr uint32_t FACE_WIDTH = 16;
// constexpr uint32_t TILE_HEIGHT = 32;
// constexpr uint32_t TILE_WIDTH = 32;

// uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
//     // Get local coordinates within the tile
//     uint32_t local_row = h % TILE_HEIGHT;
//     uint32_t local_col = w % TILE_WIDTH;

//     // Determine the index offset based on which quadrant we're in
//     uint32_t offset = 0;

//     // If we're in the right half (columns beyond FACE_WIDTH)
//     if (local_col >= FACE_WIDTH) {
//         local_col -= FACE_WIDTH;
//         offset += FACE_HEIGHT * FACE_WIDTH;  // Right face offset
//     }

//     // If we're in the bottom half (rows beyond FACE_WIDTH)
//     if (local_row >= FACE_WIDTH) {
//         local_row -= FACE_WIDTH;
//         offset += FACE_HEIGHT * TILE_WIDTH;  // Bottom face offset
//     }

//     // Final index within the tile
//     uint32_t index = offset + local_row * FACE_WIDTH + local_col;
//     return index;
// }

inline float bfloat16_to_float(uint16_t bf16) {
    uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &tmp, sizeof(result));
    return result;
}

inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_target_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_output_idx = tt::CBIndex::c_9;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // number of tiles in inner dimension

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    const DataFormat data_format = get_dataformat(cb_output_idx);

    const InterleavedAddrGenFast</* is dram */ true> output_addr_generator = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t end_row = start_row + num_rows_to_process;

    for (uint32_t r = start_row; r < end_row; r++) {
        cb_wait_front(cb_target_idx, onetile);
        auto target_indexes_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(cb_target_idx));

        for (uint32_t c = 0, idx = r * Wt; c < Wt; c += block_size) {
            cb_wait_front(cb_output_idx, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_output_idx);

            auto write_ouput_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_read_addr);

            for (uint32_t h = 0; h < TILE_HEIGHT; ++h) {
                auto target_value = target_indexes_l1_ptr[h];

                uint32_t tile_idx = target_value / TILE_WIDTH;

                if (tile_idx >= c && tile_idx < c + block_size) {
                    uint32_t local_tile_idx = tile_idx - c;
                    uint32_t index_inside_tile =
                        (TILE_WIDTH * TILE_HEIGHT * local_tile_idx) + get_tilized_idx(h, target_value);

                    float value = bfloat16_to_float(write_ouput_l1_ptr[index_inside_tile]);
                    value -= 1.0F;
                    write_ouput_l1_ptr[index_inside_tile] = float_to_bfloat16(value);
                }
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
