// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

// Blackhole: pack_reconfig_data_format between a narrow-row pack_untilize_dest_init and pack_untilize_dest. The
// narrow-row init programs the packer X (datum) counter end to row_num_datums - 1 and the execute never
// re-programs it; a reconfig that resets the counter end to a full face row between the two makes every packed
// row carry 16 datums and overrun its L1 row slot. The output CB is a one-face (16x16) tile; a block row of width
// input tiles packs 16 rows of width * row_num_datums datums, the first row_num_datums datums of face 0 of each
// tile side by side, which is width * row_num_datums / 16 output pages.
void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t width = get_compile_time_arg_val(1);
    constexpr uint32_t row_num_datums = get_compile_time_arg_val(2);
    static_assert((width * row_num_datums) % FACE_C_DIM == 0, "a block row must fill whole one-face output pages");
    constexpr uint32_t out_pages = width * row_num_datums / FACE_C_DIM;

    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);

    compute_kernel_hw_startup(in_cb, out_cb);
    copy_init(in_cb);
    pack_untilize_dest_init<width, width, true /*narrow_row*/, row_num_datums>(out_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t row = 0; row < rows; ++row) {
        cb_in.wait_front(width);
        cb_out.reserve_back(out_pages);
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < width; ++tile) {
            copy_tile(in_cb, tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest<width, width, false /*diagonal*/, true /*narrow_row*/, row_num_datums>(
            out_cb, 1 /*block_rt_dim*/, 0 /*block_c_index*/);
        tile_regs_release();
        cb_in.pop_front(width);
        cb_out.push_back(out_pages);
    }
    pack_untilize_uninit(out_cb);
}
