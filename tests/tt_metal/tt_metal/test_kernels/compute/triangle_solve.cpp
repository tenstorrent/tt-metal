// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/triangle_solve.h"
#include "api/dataflow/circular_buffer.h"

// One triangle_solve_tile per RHS tile: L X = RHS with L unit lower-triangular, solved on the SFPU.
//
//   c_0  : L tiles, Float32 or Float16_b, read by the solve in place from L1 (never unpacked)
//   c_1  : RHS tiles, Float32 on the unpack-to-dest path so RHS reaches DST bit-exact
//   c_16 : X tiles, Float32
//
// Compile-time args:
//   0 NUM_TILES         RHS tiles solved (= X tiles produced)
//   1 L_TILES_PER_BLOCK L tiles front-waited together; within a block they are used last-first, so RHS tile i
//                       pairs with L tile (i / L_TILES_PER_BLOCK) * L_TILES_PER_BLOCK + (L_TILES_PER_BLOCK - 1 -
//                       i % L_TILES_PER_BLOCK) and a nonzero l_tile_idx reaches get_tile_address
//   2 L_IS_BF16         1: the L tiles are Float16_b, 0: Float32
//   3 L_NEGATED         1: the L tiles hold -L below the diagonal
//
// Per RHS tile: copy RHS to DST 0, solve into DST 1, pack DST 1. Requires fp32_dest_acc_en.

namespace {
constexpr uint32_t NUM_TILES = get_compile_time_arg_val(0);
constexpr uint32_t L_TILES_PER_BLOCK = get_compile_time_arg_val(1);
constexpr bool L_IS_BF16 = get_compile_time_arg_val(2) != 0;
constexpr bool L_NEGATED = get_compile_time_arg_val(3) != 0;
constexpr DataFormat L_FORMAT = L_IS_BF16 ? DataFormat::Float16_b : DataFormat::Float32;
static_assert(NUM_TILES % L_TILES_PER_BLOCK == 0, "every L block pairs with L_TILES_PER_BLOCK RHS tiles");

constexpr uint32_t DST_RHS = 0;  // the copied right-hand side
constexpr uint32_t DST_X = 1;    // the solution; must differ from DST_RHS
}  // namespace

void kernel_main() {
    constexpr auto cb_l = tt::CBIndex::c_0;
    constexpr auto cb_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    CircularBuffer l(cb_l);
    CircularBuffer rhs(cb_rhs);
    CircularBuffer out(cb_out);

    // The solve loads and stores DST rows in the SrcB-implied format, so both source formats are configured from
    // the fp32 RHS CB and DST is read back as fp32.
    compute_kernel_hw_startup(cb_rhs, cb_out);
    copy_init(cb_rhs);
    triangle_solve_tile_init();

    for (uint32_t block = 0; block < NUM_TILES / L_TILES_PER_BLOCK; ++block) {
        l.wait_front(L_TILES_PER_BLOCK);
        for (uint32_t j = 0; j < L_TILES_PER_BLOCK; ++j) {
            const uint32_t l_tile_idx = L_TILES_PER_BLOCK - 1 - j;

            rhs.wait_front(1);
            out.reserve_back(1);

            tile_regs_acquire();
            copy_tile(cb_rhs, 0, DST_RHS);
            triangle_solve_tile<L_FORMAT, L_NEGATED>(l, l_tile_idx, DST_RHS, DST_X);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(DST_X, cb_out);
            tile_regs_release();

            rhs.pop_front(1);
            out.push_back(1);
        }
        l.pop_front(L_TILES_PER_BLOCK);
    }
}
