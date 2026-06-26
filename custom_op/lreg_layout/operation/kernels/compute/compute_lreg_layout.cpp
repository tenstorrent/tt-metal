// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for lreg_layout custom op.
//
// Per-face SFPU LLK: writes [0..255] across the 256 elements of one 16x16 face.
// vConstTileId (CREG #15) holds [0,2,4,…,62] across the 32 lanes; we shift
// right by 1 once outside the loop so vdata starts at [0..31] (lane id), then
// step by 32 each iteration so the 8 iterations cover [0..31], [32..63], …,
// [224..255]. Read back as a uint32 tile, the (row, col) → ID mapping reveals
// the (face, iter, lane) layout.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH

namespace {

// Called once per face by _llk_math_eltwise_unary_sfpu_params_; the params
// helper increments the DST face base address between calls, so dst_reg[i]
// here addresses positions i ∈ [0,8) within the *current* face.
inline void write_lreg_ids_face() {
    sfpi::vUInt vdata = sfpi::reinterpret<sfpi::vUInt>(sfpi::vConstTileId) >> 1;
#pragma GCC unroll 8
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::dst_reg[i] = vdata;
        vdata += 32;
    }
}

}  // namespace

#endif  // TRISC_MATH

inline void write_lreg_ids_tile(uint32_t idx_dst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(write_lreg_ids_face, idx_dst)));
}

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto c_in = tt::CBIndex::c_0;
    constexpr auto c_out = tt::CBIndex::c_2;

    init_sfpu(c_in, c_out);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(c_in, 1);
        cb_reserve_back(c_out, 1);

        tile_regs_acquire();
        copy_tile(c_in, 0, 0);     // bring input into DST (data is overwritten below)
        write_lreg_ids_tile(0);    // SFPU writes lane/iter IDs into DST tile 0
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, c_out);
        tile_regs_release();

        cb_push_back(c_out, 1);
        cb_pop_front(c_in, 1);
    }
}
