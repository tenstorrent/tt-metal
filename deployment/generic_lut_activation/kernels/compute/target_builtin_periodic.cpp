// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compact closed lowering for a typed target periodic builtin.  The compiler
// emits exactly one phase macro.  This is intentionally a normal generated
// kernel include: exhaustive accuracy and wall timing execute this same body.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/trigonometry.h"

#if (                                                                                       \
    defined(TT_TARGET_BUILTIN_PERIODIC_SINE) + defined(TT_TARGET_BUILTIN_PERIODIC_COSINE) + \
    defined(TT_TARGET_BUILTIN_PERIODIC_TANGENT)) != 1
#error "typed periodic builtin lowering requires exactly one phase"
#endif

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_in, cb_out);
#if defined(TT_TARGET_BUILTIN_PERIODIC_SINE)
    ckernel::sin_tile_init();
#elif defined(TT_TARGET_BUILTIN_PERIODIC_COSINE)
    ckernel::cos_tile_init();
#endif

    for (uint32_t tile = 0; tile < n_tiles; ++tile) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        // Reserve before issuing MATH, matching the stock unary pipeline.  A
        // late reserve serializes output-CB availability behind the entire
        // intrinsic and showed up as fixed wrapper latency on short timings.
        cb_reserve_back(cb_out, 1);
        copy_tile(cb_in, 0, 0);

#if defined(TT_TARGET_BUILTIN_PERIODIC_SINE)
        ckernel::sin_tile(0);
#elif defined(TT_TARGET_BUILTIN_PERIODIC_COSINE)
        ckernel::cos_tile(0);
#else
        // The public tangent body mutates its programmable constants.  TTNN
        // initializes it in the per-tile unary chain, so preserve that exact
        // target contract here.
        ckernel::tan_tile_init();
        ckernel::tan_tile(0);
#endif

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
