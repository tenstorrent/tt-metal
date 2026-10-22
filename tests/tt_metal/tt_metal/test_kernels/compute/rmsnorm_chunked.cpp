// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/rmsnorm.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr std::uint32_t capacity = get_compile_time_arg_val(1);
    constexpr std::uint32_t rows = get_compile_time_arg_val(2);
    constexpr bool clear_only = get_compile_time_arg_val(3) != 0;
    CircularBuffer input_a(tt::CBIndex::c_0);
    CircularBuffer input_b(tt::CBIndex::c_1);
    CircularBuffer output(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    for (std::uint32_t row = 0; row < rows; ++row) {
        if constexpr (clear_only) {
            for (std::uint32_t target = 0; target < capacity - 1; ++target) {
                output.reserve_back(capacity);
                mul_reduce_scalar_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
                fill_tile_init();
                tile_regs_acquire();
                for (std::uint32_t tile = 0; tile < capacity; ++tile) {
                    fill_tile(tile, static_cast<float>(row * 16 + tile + 1));
                }
                // Verify every face of every slot, including the accumulator
                // and the neighboring bank, survives unless explicitly cleared.
                MATH((llk_math_rmsnorm_clear_product_tile<capacity, DST_ACCUM_MODE>(target)));
                tile_regs_commit();
                tile_regs_wait();
                for (std::uint32_t tile = 0; tile < capacity; ++tile) {
                    pack_tile(tile, tt::CBIndex::c_16);
                }
                tile_regs_release();
                output.push_back(capacity);
            }
        } else {
            input_a.wait_front(num_tiles);
            input_b.wait_front(num_tiles);
            output.reserve_back(1);
            mul_reduce_scalar_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
            fill_tile_init();
            add_binary_tile_init();
            tile_regs_acquire();
            // A fresh acquire may already have zeroed DEST. Poison its slots so
            // the first chunk also requires the API's targeted product clear.
            for (std::uint32_t tile = 0; tile < capacity; ++tile) {
                fill_tile(tile, static_cast<float>(row * 16 + tile + 1));
            }
            // GAPOOL applies the scaler in both the column and row reductions.
            // Use sqrt(1/1024), matching RMSNorm's 1/sqrt(width) convention.
            mul_reduce_scalar_chunked_tile<num_tiles, capacity>(
                tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, 1.0f / 32.0f);
            mul_reduce_scalar_uninit();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(capacity - 1, tt::CBIndex::c_16);
            tile_regs_release();
            input_a.pop_front(num_tiles);
            input_b.pop_front(num_tiles);
            output.push_back(1);
        }
    }
}
