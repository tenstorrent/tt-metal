// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "ckernel_sfpu_gdn_inv_rms_diag.h"

// chunk_gated_delta_rule (prep) only: see ckernel_sfpu_gdn_inv_rms_diag.h.

namespace ckernel {

// The SFPU config, the DST-absolute address mode and the rsqrt body's programmable constants (rsqrt_tile_init's).
template <bool APPROXIMATE>
inline void llk_math_gdn_inv_rms_diag_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(sfpu::sqrt_init<APPROXIMATE>);
}

// DST[idst_x] <- the inverse-RMS diagonal of the Gram tile it holds; DST[idst_eye] holds the identity.
template <bool APPROXIMATE, bool DO_SCALE>
inline void llk_math_gdn_inv_rms_diag(
    const std::uint32_t idst_x,
    const std::uint32_t idst_eye,
    const std::uint32_t eps_bits,
    const std::uint32_t scale_bits) {
    _llk_math_eltwise_sfpu_start_(0);
    sfpu::calculate_gdn_inv_rms_diag<APPROXIMATE, DO_SCALE>(idst_x, idst_eye, eps_bits, scale_bits);
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel
