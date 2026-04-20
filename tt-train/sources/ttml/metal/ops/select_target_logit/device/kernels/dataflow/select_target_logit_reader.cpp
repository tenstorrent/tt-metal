// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    using namespace tt::constants;

    uint32_t runtime_args_counter = 0U;
    uint32_t logit_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t target_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t first_v = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t last_v = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_target_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_logit_scratch = tt::CBIndex::c_1;
    constexpr uint32_t cb_output_idx = tt::CBIndex::c_2;

    // Compile-time args:
    //   0: Wt                          - tiles in vocab dimension per row
    //   1: tiled_H                     - tile rows per batch element (Ht)
    //   2: target_page_size            - bytes per batch-element page in target (H * 4)
    //   3: target_read_page_size       - bytes to read per tile row (32 * 4 = 128)
    //   4: output_is_fp32              - 1 if output tile is Float32, 0 if BFloat16
    //   5..: TensorAccessorArgs(logit)
    //   N..: TensorAccessorArgs(target)
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t tiled_H = get_compile_time_arg_val(1);
    constexpr uint32_t target_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t target_read_page_size = get_compile_time_arg_val(3);
    constexpr bool output_is_fp32 = get_compile_time_arg_val(4) != 0;

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_logit_scratch);

    constexpr auto logit_accessor_args = TensorAccessorArgs<5>();
    constexpr auto target_accessor_args = TensorAccessorArgs<logit_accessor_args.next_compile_time_args_offset()>();
    const auto logit_addr_gen = TensorAccessor(logit_accessor_args, logit_address, tile_bytes);
    const auto target_addr_gen = TensorAccessor(target_accessor_args, target_address, target_page_size);

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t row = start_row + i;
        const uint32_t logit_row_start = row * Wt;

        // Read 32 target indices for this tile row
        cb_reserve_back(cb_target_idx, onetile);
        uint32_t l1_target_write_addr = get_write_ptr(cb_target_idx);

        auto [page, offset] = get_page_and_offset(row, tiled_H);
        noc_async_read(get_noc_addr(page, target_addr_gen, offset), l1_target_write_addr, target_read_page_size);
        noc_async_read_barrier();

        // Reserve output tile and zero-initialise it.
        // Positions where c is out of [first_v, last_v) will stay 0.0 (zero bits for both BF16 and FP32).
        cb_reserve_back(cb_output_idx, onetile);
        uint32_t l1_output_write_addr = get_write_ptr(cb_output_idx);

        const uint32_t output_tile_bytes = get_tile_size(cb_output_idx);
        auto out_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_output_write_addr);
        const uint32_t out_tile_u32s = output_tile_bytes / sizeof(uint32_t);
        for (uint32_t j = 0U; j < out_tile_u32s; ++j) {
            out_u32[j] = 0U;
        }

        auto target_indexes_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(cb_target_idx));

        for (uint32_t h = 0U; h < TILE_HEIGHT; ++h) {
            const uint32_t c = target_indexes_l1_ptr[h];

            if (c >= first_v && c < last_v) {
                const uint32_t local_c = c - first_v;

                read_tiles_by_row(
                    cb_logit_scratch,
                    logit_addr_gen,
                    logit_row_start + local_c / TILE_WIDTH,
                    onetile,
                    tile_bytes,
                    onetile);

                auto read_logit_tile_l1_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(cb_logit_scratch));

                if constexpr (output_is_fp32) {
                    // BF16 → FP32: left-shift 16 bits (BF16 is the upper 16 bits of IEEE 754 float32)
                    auto output_write_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_output_write_addr);
                    uint16_t bf16_val = read_logit_tile_l1_ptr[get_tilized_idx(h, local_c)];
                    output_write_l1_ptr[get_tilized_idx(h, 0U)] = static_cast<uint32_t>(bf16_val) << 16;
                } else {
                    auto output_write_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_output_write_addr);
                    output_write_l1_ptr[get_tilized_idx(h, 0U)] = read_logit_tile_l1_ptr[get_tilized_idx(h, local_c)];
                }

                cb_pop_front(cb_logit_scratch, onetile);
            }
        }

        cb_push_back(cb_output_idx, onetile);
        cb_pop_front(cb_target_idx, onetile);
    }
}
