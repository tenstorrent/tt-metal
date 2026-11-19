// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix — compute: per column tile j, ys = y (+ y2) (exact fp32 SFPU), then for each output stream s:
// out = ys * bcast_col(post_s) + sum_i x_i * bcast_col(C[i][s]) (exact fp32 SFPU products and sums, the mix_streams
// fp32-mode arithmetic). Compile-time args: ADD.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

#include "mhc_post_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#if defined(MPOST_V2)
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

FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}

}  // namespace

#if defined(MPOST_V2)
// Stage 14c v2 (mhc_post_v2_sfpu.hpp): per column tile j ONE DST window: y (+ y2) -> 0, x_0..3 -> 1..4, the compact
// coefficient tile -> 5; the fused mix in place -> pack 1..4 = out_0..3. Runtime arg: T (tokens).
void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_OUT);
    const uint32_t nb = (get_arg_val<uint32_t>(ph_rt0 + 0) + 3) / 4;
    CircularBuffer cb_y(CB_Y), cb_y2(CB_Y2), cb_x(CB_X), cb_coef(CB_COEF), cb_out(CB_OUT);
    cb_y.wait_front(COLS_PER_CORE);
    if constexpr (ADD) {
        cb_y2.wait_front(COLS_PER_CORE);
    }
    cb_x.wait_front(X_TILES);
    cb_coef.wait_front(1);
    for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
        cb_out.reserve_back(N_STREAMS);
        tile_regs_acquire();
        load(CB_Y, j, 0);
        if constexpr (ADD) {
            load(CB_Y2, j, 7);
            add_binary_tile_init();
            add_binary_tile(0, 7, 0);
        }
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            load(CB_X, s * COLS_PER_CORE + j, 1 + s);
        }
        load(CB_COEF, 0, 5);
        MATH((SFPU_UNARY_INIT(unused)));
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, mhc_post_v2_mix, 0, VectorMode::RC_custom, nb)));
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            pack_to(1 + s, CB_OUT);
        }
        tile_regs_release();
        cb_out.push_back(N_STREAMS);
    }
    cb_y.pop_front(COLS_PER_CORE);
    if constexpr (ADD) {
        cb_y2.pop_front(COLS_PER_CORE);
    }
    cb_x.pop_front(X_TILES);
    cb_coef.pop_front(1);
}
#else
void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_OUT);
    CircularBuffer cb_y(CB_Y), cb_y2(CB_Y2), cb_x(CB_X), cb_coef(CB_COEF), cb_ys(CB_YS), cb_out(CB_OUT);
    cb_y.wait_front(COLS_PER_CORE);
    if constexpr (ADD) {
        cb_y2.wait_front(COLS_PER_CORE);
    }
    cb_x.wait_front(X_TILES);
    cb_coef.wait_front(N_COEF);
    DeviceZoneScopedN("MP-MIX");  // device-profiler zone: the arithmetic once every input is in L1
    for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
        cb_ys.reserve_back(1);
        tile_regs_acquire();
        load(CB_Y, j, 0);
        if constexpr (ADD) {
            load(CB_Y2, j, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_YS);
        tile_regs_release();
        cb_ys.push_back(1);
        cb_ys.wait_front(1);
        cb_out.reserve_back(N_STREAMS);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            tile_regs_acquire();
            load(CB_YS, 0, 0);
            load(CB_COEF, s, 3);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 3);
            for (uint32_t i = 0; i < N_STREAMS; ++i) {
                load(CB_X, i * COLS_PER_CORE + j, 1);
                load(CB_COEF, N_STREAMS + i * N_STREAMS + s, 3);
                sfpu_mul_bcast_col_init();
                sfpu_mul_bcast_col(1, 3);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, CB_OUT);
            tile_regs_release();
        }
        cb_out.push_back(N_STREAMS);
        cb_ys.pop_front(1);
    }
    cb_y.pop_front(COLS_PER_CORE);
    if constexpr (ADD) {
        cb_y2.pop_front(COLS_PER_CORE);
    }
    cb_x.pop_front(X_TILES);
    cb_coef.pop_front(N_COEF);
}
#endif  // MPOST_V2
