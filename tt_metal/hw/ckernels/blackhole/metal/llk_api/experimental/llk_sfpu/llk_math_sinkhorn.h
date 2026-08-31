// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_sinkhorn.h"

namespace ckernel {

inline void _llk_math_sinkhorn_4x4_init_() {
    // DEST is 16-bit throughout the sinkhorn 4x4 path (see api/compute/experimental/sinkhorn.h),
    // so is_fp32_dest_acc_en=false is fixed here. This init is optional -- exp_tile_init programs
    // the same ADDR_MOD_7 slot immediately before sinkhorn_row_max_sub / sinkhorn_4x4 -- but the
    // entry point still has to compile whenever the header is included.
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*is_fp32_dest_acc_en=*/false>();
}

inline void _llk_math_sinkhorn_row_max_sub_(std::uint32_t input_index) {
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_sinkhorn_row_max_sub_4x4_, input_index, VectorMode::RC_custom);
}

template <
    std::uint32_t NUM_FACES_USED,
    std::uint32_t ITERS,
    std::uint32_t EPS_BITS,
    bool SINGLE_SUBMAT,
    std::uint32_t VALID_H,
    std::uint32_t VALID_W>
inline void _llk_math_sinkhorn_4x4_(std::uint32_t input_index) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_sinkhorn_4x4_<NUM_FACES_USED, ITERS, EPS_BITS, SINGLE_SUBMAT, VALID_H, VALID_W>,
        input_index,
        VectorMode::RC_custom);
}

}  // namespace ckernel
