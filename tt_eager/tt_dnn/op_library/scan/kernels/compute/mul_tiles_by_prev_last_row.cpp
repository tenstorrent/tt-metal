

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "debug/dprint.h"

#define PRINTER PACK

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);
constexpr auto cb_last_row = get_compile_time_arg_val(2);

constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;

constexpr uint32_t ndst = 8;


namespace NAMESPACE {
void MAIN {
    uint32_t tiles_per_col = get_arg_val<uint32_t>(0);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(1);
    uint32_t total_tiles = get_arg_val<uint32_t>(2);
    uint32_t ncores = get_arg_val<uint32_t>(3);
    uint32_t ntiles_last_row_cb = get_arg_val<uint32_t>(4);

    cb_wait_front(cb_last_row, ntiles_last_row_cb);

    mul_tiles_init(cb_src, cb_last_row);

    PRINTER(DPRINT << "BEFORE" << ENDL());
    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        PRINTER(DPRINT << "ROW: " << row << ENDL());
        uint32_t factor_idx = 0;
        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            PRINTER(DPRINT << "ONE" << ENDL());
            cb_wait_front(cb_src, tiles_per_reshape);
            PRINTER(DPRINT << "TWO" << ENDL());

            for (uint32_t tile = 0; tile < tiles_per_reshape; tile += ndst) {
                PRINTER(DPRINT << "THREE" << ENDL());
                tile_regs_acquire();
                for (uint32_t t = 0; t < ndst; ++t) {
                    mul_tiles(cb_src, cb_last_row, t, factor_idx, t);
                }
                PRINTER(DPRINT << "FOUR" << ENDL());
                cb_pop_front(cb_src, ndst);
                PRINTER(DPRINT << "FIVE" << ENDL());
                tile_regs_commit();
                cb_reserve_back(cb_dst, ndst);
                tile_regs_wait();
                PRINTER(DPRINT << "SIX" << ENDL());
                // PACKER STALLS HERE ON THE FIRST ITERATION
                for (uint32_t t = 0; t < ndst; ++t) {
                    pack_tile(t, cb_dst);
                }
                PRINTER(DPRINT << "SEVEN" << ENDL());
                tile_regs_release();
                PRINTER(DPRINT << "EIGHT" << ENDL());
                cb_push_back(cb_dst, ndst);
            }
            PRINTER(DPRINT << "NINE" << ENDL());

            factor_idx += ncores;
        }
    }
    PRINTER(DPRINT << "AFTER" << ENDL()); // unpack goes here but cannot finish

}
}  // namespace NAMESPACE
