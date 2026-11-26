// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix, PACKED decode mode (T <= 4) — compute: the core's whole post-mix on ONE tile (reader_p.cpp: row
// r = (stream s, column tile j, token t)): out = (y [+ y2]) * bcast_col(post) + sum_i x_i * bcast_col(C_i), exact fp32
// SFPU products / sums in the order of compute.cpp (bitwise the unpacked program).
// Compile-time args: ADD.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/kernel_profiler.hpp"

#include "mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#if defined(MPOST_P_FMA)
#include "api/compute/eltwise_unary/fill.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_post/mhc_post_v2_sfpu.hpp"
#endif

using namespace mpost;

namespace {

constexpr uint32_t ADD = get_compile_time_arg_val(PH_CT0 + 0);

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);  // bf16 y / fp32 residual + coefficients alternate
    copy_init(cb);
    copy_tile(cb, i, d);
}

#if defined(MPOST_P_FMA)
// DST tile d <- column 0 of DST tile c broadcast over its row (1.0 * c: exact)
FORCE_INLINE void bcast_row_coef(uint32_t d, uint32_t c) {
    fill_tile_init();
    fill_tile(d, 1.0f);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(d, c);
}
#endif

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_OUT);
    cb_wait_front(CB_Y, 1);
    if constexpr (ADD) {
        cb_wait_front(CB_Y2, 1);
    }
    cb_wait_front(CB_X, N_STREAMS);
    cb_wait_front(CB_COEF, 1 + N_STREAMS);
    DeviceZoneScopedN("MP-MIX");
    cb_reserve_back(CB_OUT, 1);
    tile_regs_acquire();
    load(CB_Y, 0, 0);
    if constexpr (ADD) {
        load(CB_Y2, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
#if defined(MPOST_P_FMA) && MPOST_P_FMA == 2
    // MPOST_P_FMA=2: the same arithmetic with the column broadcast fused into each SFPU pass (mhc_post_p_bcast_mad: no
    // fill / LLK broadcast per term); tile 1 = x_0 * C_0, then tile 0 = y * post + tile 1, then tile 0 += x_i * C_i
    load(CB_X, 0, 1);
    load(CB_COEF, 1, 2);
    MATH((SFPU_UNARY_INIT(unused)));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, mhc_post_p_bcast_mad, (1, 1, 2, 0, false), 0, VectorMode::RC_custom)));
    load(CB_COEF, 0, 2);
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, mhc_post_p_bcast_mad, (0, 0, 2, 1, true), 0, VectorMode::RC_custom)));
    for (uint32_t i = 1; i < N_STREAMS; ++i) {
        load(CB_X, i, 1);
        load(CB_COEF, 1 + i, 2);
        MATH((SFPU_UNARY_INIT(unused)));
        MATH((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, mhc_post_p_bcast_mad, (0, 1, 2, 0, true), 0, VectorMode::RC_custom)));
    }
#elif defined(MPOST_P_FMA)
    // Stage 14b P1-3 option 2: pv2's arithmetic as compiled (mhc_post_v2_sfpu.hpp mhc_post_p_mad): x_0 * C_0 rounded,
    // then y * post + that as ONE SFPMAD, then + x_i * C_i as one SFPMAD each; tile 2 = a coefficient broadcast over
    // its row (1.0 * bcast_col: exact)
    load(CB_X, 0, 1);
    load(CB_COEF, 1, 3);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(1, 3);
    load(CB_COEF, 0, 3);
    bcast_row_coef(2, 3);
    MATH((SFPU_UNARY_INIT(unused)));
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, mhc_post_p_mad, (0, 0, 2, 1), 0, VectorMode::RC_custom)));
    for (uint32_t i = 1; i < N_STREAMS; ++i) {
        load(CB_X, i, 1);
        load(CB_COEF, 1 + i, 3);
        bcast_row_coef(2, 3);
        MATH((SFPU_UNARY_INIT(unused)));
        MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, mhc_post_p_mad, (0, 1, 2, 0), 0, VectorMode::RC_custom)));
    }
#else
    load(CB_COEF, 0, 3);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 3);
    for (uint32_t i = 0; i < N_STREAMS; ++i) {
        load(CB_X, i, 1);
        load(CB_COEF, 1 + i, 3);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(1, 3);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_OUT);
    pack_tile(0, CB_OUT);
    tile_regs_release();
    cb_push_back(CB_OUT, 1);
    cb_pop_front(CB_Y, 1);
    if constexpr (ADD) {
        cb_pop_front(CB_Y2, 1);
    }
    cb_pop_front(CB_X, N_STREAMS);
    cb_pop_front(CB_COEF, 1 + N_STREAMS);
}
