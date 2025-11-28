// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"

#ifdef TRISC_MATH
template <int ITERATIONS = 8>
inline void stochastic_round_tile_face() {
#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; i++) {
        vFloat a = dst_reg[0];
        vUInt rounded = float_to_fp16b(a, 1);
        dst_reg[0] = reinterpret<vFloat>(rounded);
        dst_reg++;
    }
}
#endif

inline void stochastic_round_tile(uint32_t idx_dst0) {
    MATH(_llk_math_eltwise_unary_sfpu_params_<false>(stochastic_round_tile_face<8>, idx_dst0));
}

namespace NAMESPACE {
void MAIN {
    uint32_t seed = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = get_compile_time_arg_val(0);
    constexpr auto cb_out0 = get_compile_time_arg_val(1);

    constexpr uint32_t dst_tile_idx = 0;

    init_sfpu(cb_in0, cb_out0);
    init_prng_seed(seed);

    const uint32_t end_tile_id = start_tile_id + n_tiles;

    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        tile_regs_acquire();

        cb_wait_front(cb_in0, 1);

        copy_tile(cb_in0, 0, dst_tile_idx);
        stochastic_round_tile(dst_tile_idx);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 1);

        pack_reconfig_data_format(cb_out0);
        pack_tile(dst_tile_idx, cb_out0);

        cb_pop_front(cb_in0, 1);

        tile_regs_release();

        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
