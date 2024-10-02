// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/dropout.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "llk_math_eltwise_unary_sfpu_dropout.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"

using namespace sfpi;

namespace NAMESPACE {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _my_calculate_dropout_(const int iterations) {
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(0, 4, 3, 0);
        TTI_SFPIADD(32, p_sfpu::LREG0, p_sfpu::LREG0, 5);
        // vUInt a = 31;
        // vUInt b = dst_reg[0];
        // dst_reg[0] = b >> 29;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void my_calculate_dropout() {
    _my_calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS);
}

template <bool APPROXIMATE>
inline void my_llk_math_eltwise_unary_sfpu_dropout(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    // Init a nice per-lane counter
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(-1 & 0xfff, 0, p_sfpu::LREG0, 1);

    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(my_calculate_dropout<APPROXIMATE>, dst_index, vector_mode);
}

ALWI void my_dropout_tile(uint32_t idst) { MATH((my_llk_math_eltwise_unary_sfpu_dropout<APPROX>(idst))); }

void MAIN {
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    dropout_tile_init(0xDEADBEEF);

    // wait for a block of tiles in each of input CBs
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();  // acquire 8 tile registers

    my_dropout_tile(0);

    tile_regs_commit();  // signal the packer

    tile_regs_wait();  // packer waits here
    pack_tile(0, cb_out0);
    tile_regs_release();  // packer releases

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);

    cb_push_back(cb_out0, 1);
}
}  // namespace NAMESPACE
