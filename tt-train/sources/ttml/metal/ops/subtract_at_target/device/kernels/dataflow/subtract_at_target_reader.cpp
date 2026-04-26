// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    using namespace tt::constants;

    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t target_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t first_v = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t last_v = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t subtract_bits = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_target_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_input_scratch = tt::CBIndex::c_1;
    constexpr uint32_t cb_output_idx = tt::CBIndex::c_2;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t tiled_H = get_compile_time_arg_val(1);
    constexpr uint32_t target_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t target_read_page_size = get_compile_time_arg_val(3);

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_input_scratch);

    constexpr auto input_accessor_args = TensorAccessorArgs<4>();
    constexpr auto target_accessor_args = TensorAccessorArgs<input_accessor_args.next_compile_time_args_offset()>();
    const auto input_addr_gen = TensorAccessor(input_accessor_args, input_address, tile_bytes);
    const auto target_addr_gen = TensorAccessor(target_accessor_args, target_address, target_page_size);

    union {
        uint32_t u;
        float f;
    } subtract_val;
    subtract_val.u = subtract_bits;

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t row = start_row + i;
        const uint32_t input_row_start = row * Wt;

        // Read 32 target indices for this tile row
        cb_reserve_back(cb_target_idx, onetile);
        uint32_t l1_target_write_addr = get_write_ptr(cb_target_idx);

        auto [page, offset] = get_page_and_offset(row, tiled_H);
        noc_async_read(get_noc_addr(page, target_addr_gen, offset), l1_target_write_addr, target_read_page_size);
        noc_async_read_barrier();

        auto target_indexes_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(cb_target_idx));

        // Process each tile column: read input tile, subtract at target positions, push to output
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // Read one input tile into scratch
            read_tiles_by_row(cb_input_scratch, input_addr_gen, input_row_start + wt, onetile, tile_bytes, onetile);

            // Copy scratch tile to output CB
            cb_reserve_back(cb_output_idx, onetile);
            uint32_t l1_output_addr = get_write_ptr(cb_output_idx);
            uint32_t l1_scratch_addr = get_read_ptr(cb_input_scratch);

            auto src_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_scratch_addr);
            auto dst_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_output_addr);
            const uint32_t tile_u32s = tile_bytes / sizeof(uint32_t);
            for (uint32_t j = 0U; j < tile_u32s; ++j) {
                dst_u32[j] = src_u32[j];
            }

            // For each row in the tile, check if the target falls in this tile column
            const uint32_t col_start = wt * TILE_WIDTH;
            const uint32_t col_end = col_start + TILE_WIDTH;

            auto output_bf16_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_output_addr);

            for (uint32_t h = 0U; h < TILE_HEIGHT; ++h) {
                const uint32_t c = target_indexes_l1_ptr[h];

                if (c >= first_v && c < last_v) {
                    const uint32_t local_c = c - first_v;

                    if (local_c >= col_start && local_c < col_end) {
                        uint16_t val_bf16 = output_bf16_ptr[get_tilized_idx(h, local_c)];
                        float val = bfloat16_to_float(val_bf16);
                        val -= subtract_val.f;
                        output_bf16_ptr[get_tilized_idx(h, local_c)] = float_to_bfloat16(val);
                    }
                }
            }

            cb_pop_front(cb_input_scratch, onetile);
            cb_push_back(cb_output_idx, onetile);
        }

        cb_pop_front(cb_target_idx, onetile);
    }
}
