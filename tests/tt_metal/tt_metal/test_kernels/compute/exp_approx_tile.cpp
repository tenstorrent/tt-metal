// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Repro kernel for test_exp_approx_disable_sfploadmacro.py: c_0 -> exp_tile<approx=true>(x) -> c_16, one tile at a
// time.
//   compile_time_args = [num_tiles, clamp_negative]
// clamp_negative=0 is the InputClamping::None branch SDPA uses (tt-metal#59499).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr bool clamp = get_compile_time_arg_val(1) == 1;
    constexpr InputClamping clamping = clamp ? InputClamping::ClampToNegative : InputClamping::None;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    CircularBuffer in_cb(cb_in);
    CircularBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    exp_tile_init<true /*approx*/, 0x3F800000 /*scale 1.0*/, clamping>();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        in_cb.wait_front(1);
        out_cb.reserve_back(1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        exp_tile<true /*approx*/, false /*scale_en*/, clamping>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        in_cb.pop_front(1);
        out_cb.push_back(1);
    }
}
