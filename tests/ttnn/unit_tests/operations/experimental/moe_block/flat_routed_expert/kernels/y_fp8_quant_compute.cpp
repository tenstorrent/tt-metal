// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FP8_Y_PLAN.md M2: the flat expert's fp8 y quantize (y_fp8_quant.hpp) on one core, as the down compute runs it: fp32
// half sync, a row tile in passes of `pass_columns` tiles, each pass's tiles copied into DEST (the matmul's place),
// quantized on the PACK thread, pack-untilized into the e4m3 out CB; the scales into the scale CB (a page a row tile:
// 32 rows x cell_words fp32 words, word b of a row its block b's scale, the rest 0).
//
// Compile-time args: row_tiles, columns (tiles a row tile), pass_columns, quantize, (unused), row_stride_tiles,
// block_tiles, cell_words
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/device/kernels/y_fp8_quant.hpp"

void kernel_main() {
    constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr uint32_t pass_columns = get_compile_time_arg_val(2);
    constexpr bool quantize = get_compile_time_arg_val(3) != 0;  // (debug: 0 packs the copied tiles as they are)
    // the out CB's row stride in tiles (an odd pass width into e4m3 writes a tile past the pass: y_fp8_quant.hpp)
    constexpr uint32_t row_stride_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t scale_cb = tt::CBIndex::c_17;
    constexpr uint32_t block_tiles = get_compile_time_arg_val(6);  // tiles a scale block
    constexpr uint32_t cell_words = get_compile_time_arg_val(7);   // a row's scale cell (fp32 words)
    constexpr uint32_t pass_blocks = pass_columns / block_tiles;
    static_assert(columns % pass_columns == 0 && columns / block_tiles <= cell_words);

    compute_kernel_hw_startup(in_cb, out_cb);
    copy_init(in_cb);
    pack_untilize_dest_init<pass_columns, row_stride_tiles>(out_cb);
    y_fp8::init();
    for (uint32_t row = 0; row < row_tiles; ++row) {
        cb_wait_front(in_cb, columns);
        cb_reserve_back(out_cb, row_stride_tiles);
        cb_reserve_back(scale_cb, 1);
        uint32_t scales = 0;
        PACK((scales = get_local_cb_interface(scale_cb).fifo_wr_ptr << 4));
        PACK(({
            volatile uint32_t* cell = reinterpret_cast<volatile uint32_t*>(scales);
            for (uint32_t i = 0; i < 32 * cell_words; ++i) {
                cell[i] = 0;
            }
        }));
        for (uint32_t pass = 0; pass < columns / pass_columns; ++pass) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < pass_columns; ++c) {
                copy_tile(in_cb, pass * pass_columns + c, c);
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (quantize) {
                y_fp8::quantize_block<pass_columns, block_tiles>(scales + pass * pass_blocks * 4, cell_words);
            }
            pack_untilize_dest<pass_columns, row_stride_tiles>(out_cb, 1, pass);
            tile_regs_release();
        }
        cb_push_back(out_cb, row_stride_tiles);
        cb_push_back(scale_cb, 1);
        cb_pop_front(in_cb, columns);
    }
}
