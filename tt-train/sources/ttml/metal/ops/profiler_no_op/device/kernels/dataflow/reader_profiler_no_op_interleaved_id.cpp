// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);        // input buffer address
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

    constexpr uint32_t cb_dataflow_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_scratch_idx = tt::CBIndex::c_1;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t padded_H = get_compile_time_arg_val(5);

    const uint32_t tile_bytes = get_tile_size(cb_dataflow_idx);
    constexpr auto input_args = TensorAccessorArgs<6>();
    const auto input_addr_generator = TensorAccessor(input_args, input_address);
    Noc noc;
    CircularBuffer dataflow_cb(cb_dataflow_idx);

    cb_reserve_back(cb_scratch_idx, 1U);
    const uint32_t scratch_l1_addr = get_write_ptr(cb_scratch_idx);

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t tile_row = start_row + i;
        const uint32_t nc = tile_row / Ht;
        const uint32_t ht = tile_row % Ht;
        const uint32_t first_input_row = nc * padded_H + ht * tt::constants::TILE_HEIGHT;
        const uint32_t valid_rows = std::min<uint32_t>(tt::constants::TILE_HEIGHT, H - ht * tt::constants::TILE_HEIGHT);

        for (uint32_t j = 0; j < Wt; j += block_size) {
            const uint32_t current_block_size = std::min<uint32_t>(block_size, Wt - j);
            cb_reserve_back(cb_dataflow_idx, block_size);
            const uint32_t block_l1_addr = get_write_ptr(cb_dataflow_idx);

            // The input is row-major while the output is tiled. Zero the whole block first so
            // partial height/width padding is deterministic, then gather each logical row into
            // the two 16-element tile faces.
            noc.async_write_zeros(dataflow_cb, block_size * tile_bytes);
            noc.write_zeros_l1_barrier();

            for (uint32_t b = 0; b < current_block_size; ++b) {
                const uint32_t first_col = (j + b) * tt::constants::TILE_WIDTH;
                const uint32_t valid_cols = std::min<uint32_t>(tt::constants::TILE_WIDTH, W - first_col);
                const uint32_t tile_l1_addr = block_l1_addr + b * tile_bytes;

                for (uint32_t h = 0; h < valid_rows; ++h) {
                    const uint32_t input_row = first_input_row + h;
                    const uint32_t row_bytes = valid_cols * sizeof(uint16_t);

                    // Tile columns start at a 64-byte boundary in each row-major page. Stage the
                    // contiguous logical row segment through an equally aligned L1 buffer before
                    // scattering it into the two 16-column faces. Direct DRAM-to-face reads have
                    // mismatched low address bits on Blackhole for every other face/row.
                    noc_async_read(
                        input_addr_generator.get_noc_addr(input_row, first_col * sizeof(uint16_t)),
                        scratch_l1_addr,
                        row_bytes);
                    noc_async_read_barrier();

                    const uint32_t first_face_cols = std::min<uint32_t>(tt::constants::FACE_WIDTH, valid_cols);
                    if (first_face_cols > 0U) {
                        tt_memmove<false, false, true, 0>(
                            noc,
                            tile_l1_addr + get_tilized_idx(h, 0U) * sizeof(uint16_t),
                            scratch_l1_addr,
                            first_face_cols * sizeof(uint16_t));
                    }

                    const uint32_t second_face_cols = valid_cols - first_face_cols;
                    if (second_face_cols > 0U) {
                        tt_memmove<false, false, true, 0>(
                            noc,
                            tile_l1_addr + get_tilized_idx(h, tt::constants::FACE_WIDTH) * sizeof(uint16_t),
                            scratch_l1_addr + first_face_cols * sizeof(uint16_t),
                            second_face_cols * sizeof(uint16_t));
                    }
                }
            }
            cb_push_back(cb_dataflow_idx, block_size);
        }
    }
}
