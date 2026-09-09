// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_csa_index_remap.h"

#include "sanitizer/api.h"

namespace ckernel {

template <std::uint32_t ROW_OFFSET>
inline void llk_math_eltwise_unary_sfpu_csa_index_remap(std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    // RC_custom advances over the 32 destination rows that make up one tile.
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_csa_index_remap_</*ITERATIONS=*/32, ROW_OFFSET>, dst_index, VectorMode::RC_custom);
}

}  // namespace ckernel
