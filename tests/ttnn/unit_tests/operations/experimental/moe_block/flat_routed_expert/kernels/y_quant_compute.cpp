// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The flat expert's y quantize (flat_routed_expert kernels/y_quant.hpp) on one core, as the down compute runs it: fp32
// half sync, a row tile in passes of `pass_columns` tiles, each pass's tiles copied into DEST (the matmul's place),
// quantized on the PACK thread, pack-untilized into the UInt8 out CB (a row stride of row_stride_tiles); the exponent
// bytes into the cells CB (a page a row tile: 32 rows' 16-byte cells).
//
// Compile-time args: format (quantized_row::Format), row_tiles, columns (tiles a row tile), pass_columns, block_tiles,
// row_stride_tiles
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/device/kernels/y_quant.hpp"

void kernel_main() {
    constexpr auto format = static_cast<quantized_row::Format>(get_compile_time_arg_val(0));
    constexpr uint32_t row_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t columns = get_compile_time_arg_val(2);
    constexpr uint32_t pass_columns = get_compile_time_arg_val(3);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t row_stride_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t cells_cb = tt::CBIndex::c_17;
    constexpr uint32_t cell_bytes = quantized_row::kCellBytes;
    static_assert(columns % pass_columns == 0 && 2 * columns <= cell_bytes);

    compute_kernel_hw_startup(in_cb, out_cb);
    copy_init(in_cb);
    pack_untilize_dest_init<pass_columns, row_stride_tiles>(out_cb);
    y_quant::init();
    for (uint32_t row = 0; row < row_tiles; ++row) {
        cb_wait_front(in_cb, columns);
        cb_reserve_back(out_cb, row_stride_tiles);
        cb_reserve_back(cells_cb, 1);
        uint32_t cells = 0;
        PACK((cells = get_local_cb_interface(cells_cb).fifo_wr_ptr << 4));
        PACK(({  // (the cells' spare bytes: 0)
            volatile uint32_t* c = reinterpret_cast<volatile uint32_t*>(cells);
            for (uint32_t i = 0; i < 32 * cell_bytes / 4; ++i) {
                c[i] = 0;
            }
        }));
        for (uint32_t pass = 0; pass < columns / pass_columns; ++pass) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < pass_columns; ++c) {
                copy_tile(in_cb, pass * pass_columns + c, c);
            }
            tile_regs_commit();
            tile_regs_wait();
            y_quant::quantize_pass<format, pass_columns, block_tiles>(cells, cell_bytes, pass);
            pack_untilize_dest<pass_columns, row_stride_tiles>(out_cb, 1, pass);
            tile_regs_release();
        }
        cb_push_back(out_cb, row_stride_tiles);
        cb_push_back(cells_cb, 1);
        cb_pop_front(in_cb, columns);
    }
}
