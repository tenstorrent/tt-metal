// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/row_norm.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t mode = get_compile_time_arg_val(0);
    constexpr std::uint32_t num_rows = get_compile_time_arg_val(1);
    constexpr std::uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr bool stats_is_fp32 = get_compile_time_arg_val(4) != 0;
    constexpr std::uint32_t inv_width_bits = get_compile_time_arg_val(5);
    constexpr std::uint32_t epsilon_bits = get_compile_time_arg_val(6);

    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer stats(tt::CBIndex::c_1);
    CircularBuffer output(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    input.wait_front(num_tiles);

    if constexpr (mode == 0 || mode == 1 || mode == 2) {
        constexpr std::uint32_t out_tiles = (mode == 1) ? 1u : num_tiles;
        output.reserve_back(out_tiles);
        copy_init(tt::CBIndex::c_0);
        row_norm_init();
        tile_regs_acquire();
        for (std::uint32_t t = 0; t < num_tiles; ++t) {
            copy_tile(tt::CBIndex::c_0, t, t);
        }
        if constexpr (mode == 0) {
            row_rmsnorm<num_rows, tile_h, num_tiles>(0, inv_width_bits, epsilon_bits);
        } else if constexpr (mode == 1) {
            row_reduce_scale_add_rsqrt<num_rows, tile_h, num_tiles>(0, inv_width_bits, epsilon_bits);
        } else {
            row_layernorm<num_rows, tile_h, num_tiles>(0, inv_width_bits, epsilon_bits);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t t = 0; t < out_tiles; ++t) {
            pack_tile(t, tt::CBIndex::c_16);
        }
        tile_regs_release();
        output.push_back(out_tiles);
    } else {
        // Wide-row 2-chunk LayerNorm with L1 stats CB pack/reload between every pass.
        static_assert(mode != 3 || num_tiles == 4, "wide-row test expects 4 tiles (2 chunks of 2)");
        constexpr std::uint32_t chunk_tiles = 2;
        constexpr std::uint32_t stats_tile = chunk_tiles;

        // Pass 1a: sum chunk 0 + chunk 1 in DEST, reduce mean into stats_tile, pack to stats CB.
        tile_regs_acquire();
        reconfig_data_format_srca(tt::CBIndex::c_0);
        copy_init(tt::CBIndex::c_0);
        add_binary_tile_init();
        for (std::uint32_t t = 0; t < chunk_tiles; ++t) {
            copy_tile(tt::CBIndex::c_0, t, t);
            copy_tile(tt::CBIndex::c_0, chunk_tiles + t, stats_tile);
            add_binary_tile(t, stats_tile, t);
        }
        row_norm_init();
        row_reduce_mean<num_rows, tile_h, chunk_tiles, stats_tile, stats_is_fp32>(0, inv_width_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(tt::CBIndex::c_1);
        stats.reserve_back(1);
        pack_tile(stats_tile, tt::CBIndex::c_1);
        stats.push_back(1);
        tile_regs_release();

        // Pass 1b chunk 0: reload stats from L1 CB, compute chunk 0 centered squares (accumulate=false).
        stats.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format_srca(tt::CBIndex::c_0);
        copy_init(tt::CBIndex::c_0);
        for (std::uint32_t t = 0; t < chunk_tiles; ++t) {
            copy_tile(tt::CBIndex::c_0, t, t);
        }
        reconfig_data_format_srca(tt::CBIndex::c_1);
        copy_init(tt::CBIndex::c_1);
        copy_tile(tt::CBIndex::c_1, 0, stats_tile);
        stats.pop_front(1);
        row_norm_init();
        row_sum_centered_squares<
            num_rows,
            tile_h,
            chunk_tiles,
            stats_tile,
            false /* accumulate */,
            false /* zero_mean */,
            stats_is_fp32>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(tt::CBIndex::c_1);
        stats.reserve_back(1);
        pack_tile(stats_tile, tt::CBIndex::c_1);
        stats.push_back(1);
        tile_regs_release();

        // Pass 1b chunk 1: reload stats from L1 CB, accumulate chunk 1 (accumulate=true), finalize inv_std.
        stats.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format_srca(tt::CBIndex::c_0);
        copy_init(tt::CBIndex::c_0);
        for (std::uint32_t t = 0; t < chunk_tiles; ++t) {
            copy_tile(tt::CBIndex::c_0, chunk_tiles + t, t);
        }
        reconfig_data_format_srca(tt::CBIndex::c_1);
        copy_init(tt::CBIndex::c_1);
        copy_tile(tt::CBIndex::c_1, 0, stats_tile);
        stats.pop_front(1);
        row_norm_init();
        row_sum_centered_squares<
            num_rows,
            tile_h,
            chunk_tiles,
            stats_tile,
            true /* accumulate */,
            false /* zero_mean */,
            stats_is_fp32>(0);
        row_finalize_inv_std<num_rows, stats_tile, false /* fast_approx */, stats_is_fp32>(
            0, inv_width_bits, epsilon_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(tt::CBIndex::c_1);
        stats.reserve_back(1);
        pack_tile(stats_tile, tt::CBIndex::c_1);
        stats.push_back(1);
        tile_regs_release();

        // Pass 2: reload finalized stats from L1 CB and normalize chunk 0 and chunk 1.
        stats.wait_front(1);
        pack_reconfig_data_format(tt::CBIndex::c_16);
        for (std::uint32_t chunk = 0; chunk < 2; ++chunk) {
            output.reserve_back(chunk_tiles);
            tile_regs_acquire();
            reconfig_data_format_srca(tt::CBIndex::c_0);
            copy_init(tt::CBIndex::c_0);
            for (std::uint32_t t = 0; t < chunk_tiles; ++t) {
                copy_tile(tt::CBIndex::c_0, chunk * chunk_tiles + t, t);
            }
            reconfig_data_format_srca(tt::CBIndex::c_1);
            copy_init(tt::CBIndex::c_1);
            copy_tile(tt::CBIndex::c_1, 0, stats_tile);
            row_norm_init();
            row_center_scale<num_rows, tile_h, chunk_tiles, stats_tile, false /* zero_mean */, stats_is_fp32>(0);
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t t = 0; t < chunk_tiles; ++t) {
                pack_tile(t, tt::CBIndex::c_16);
            }
            tile_regs_release();
            output.push_back(chunk_tiles);
        }
        stats.pop_front(1);
    }

    input.pop_front(num_tiles);
}
