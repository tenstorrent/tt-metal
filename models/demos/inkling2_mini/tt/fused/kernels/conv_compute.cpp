// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused short conv, compute (fp32 SFPU math, exact): per dense page t
//   xf = fp32(x); y = x*tap0 + ring1*tap1 + ring2*tap2 + ring3*tap3   (tap0 has the residual folded in, all /sqrt5)
//   yr = y rounded to x's dtype (HF: conv(...).to(input dtype))
//   out = yr (+ resid, fp32 add) packed in the output dtype.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t has_resid = get_named_compile_time_arg_val("has_resid");
    constexpr uint32_t stage2 = get_named_compile_time_arg_val("stage2");  // 0: yr is the output
    constexpr uint32_t cb_x = 0, cb_ring = 1, cb_taps = 2, cb_resid = 3, cb_xf = 16, cb_y = 17, cb_yr = 18;

    compute_kernel_hw_startup(cb_x, cb_xf);
    cb_wait_front(cb_x, T);
    cb_wait_front(cb_taps, 4 * T);
    cb_wait_front(cb_ring, 3 * T);
    for (uint32_t t = 0; t < T; ++t) {
        tile_regs_acquire();
        load(cb_x, t, 0);
        load(cb_taps, t, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 1);
        for (uint32_t k = 1; k < 4; ++k) {
            load(cb_ring, (k - 1) * T + t, 2);
            load(cb_taps, k * T + t, 3);
            mul_binary_tile_init();
            mul_binary_tile(2, 3, 2);
            add_binary_tile_init();
            add_binary_tile(1, 2, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_xf);
        put(1, stage2 ? cb_yr : cb_y);
        tile_regs_release();
    }
    cb_pop_front(cb_x, T);
    cb_pop_front(cb_taps, 4 * T);
    cb_pop_front(cb_ring, 3 * T);

    if constexpr (stage2) {
        cb_wait_front(cb_yr, T);
        if constexpr (has_resid) {
            cb_wait_front(cb_resid, T);
        }
        for (uint32_t t = 0; t < T; ++t) {
            tile_regs_acquire();
            load(cb_yr, t, 0);
            if constexpr (has_resid) {
                load(cb_resid, t, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            put(0, cb_y);
            tile_regs_release();
        }
        cb_pop_front(cb_yr, T);
        if constexpr (has_resid) {
            cb_pop_front(cb_resid, T);
        }
    }
}
