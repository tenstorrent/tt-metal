// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Round 3 eltwise binary (#58723 third review) dump kernel: c_0 op c_1 into c_16, tile by tile (forms 0 and 1) or in blocks of
// 8 (form 2). EB_DUMP_PER_TILE (host define, 0 or 1) is the kernel's hand-off: 0 is main's per-face program, 1 the whole-tile
// program. CT args: n tiles, form (0 standard, 1 column broadcast multiply, 2 post_sdpa's srcB-reuse multiply with P1 from
// column 0 of c_1's first tile of the block and c_2 zero), op (0 add, 1 sub, 2 mul).
#define ELTWISE_BINARY_PER_TILE_HANDOFF EB_DUMP_PER_TILE
#define ELTWISE_BINARY_PER_TILE_HANDOFF_BCAST EB_DUMP_PER_TILE
#define SDPA_BCAST_COL_REUSE_PER_TILE_HANDOFF EB_DUMP_PER_TILE
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/sdpa.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t form = get_compile_time_arg_val(1);
    constexpr uint32_t op = get_compile_time_arg_val(2);
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_z = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    if constexpr (form == 2) {
        constexpr uint32_t nt = 8;
        for (uint32_t blk = 0; blk < n / nt; ++blk) {
            cb_wait_front(cb_a, nt);
            cb_wait_front(cb_b, nt);
            cb_wait_front(cb_z, nt);
            cb_reserve_back(cb_out, nt);
            tile_regs_acquire();
            copy_init(cb_b);
            copy_tile(cb_b, 0, 0);
            copy_init(cb_z);
            copy_tile(cb_z, 0, 1);
            sdpa_mul_bcast_col_reuse_tiles_init<nt>(cb_a);
            sdpa_bcast_col_reuse_preamble<true>();
            sdpa_mul_bcast_col_reuse_tiles<nt>(cb_a, cb_z, 0, 0);
            sdpa_bcast_col_reuse_postamble();
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nt; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, nt);
            cb_pop_front(cb_a, nt);
            cb_pop_front(cb_b, nt);
            cb_pop_front(cb_z, nt);
        }
    } else {
        if constexpr (form == 1) {
            mul_bcast_cols_init(cb_a, cb_b);
        } else if constexpr (op == 2) {
            mul_init(cb_a, cb_b);
        } else if constexpr (op == 1) {
            sub_init(cb_a, cb_b);
        } else {
            add_init(cb_a, cb_b);
        }
        for (uint32_t i = 0; i < n; ++i) {
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, 1);
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            if constexpr (form == 1) {
                mul_tiles_bcast_cols(cb_a, cb_b, 0, 0, 0);
            } else if constexpr (op == 2) {
                mul_tiles(cb_a, cb_b, 0, 0, 0);
            } else if constexpr (op == 1) {
                sub_tiles(cb_a, cb_b, 0, 0, 0);
            } else {
                add_tiles(cb_a, cb_b, 0, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_b, 1);
        }
    }
}
