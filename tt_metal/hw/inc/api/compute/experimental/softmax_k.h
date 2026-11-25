// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"
// Blackhole-only: the softmax_k datacopy/SFPU LLKs live only in the Blackhole llk_lib.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_unary_datacopy_softmax_k_api.h"
#include "experimental/llk_sfpu/llk_math_softmax_k_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_A.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softmax_k_init(uint32_t icb) {
    UNPACK((llk_unpack_A_init<BroadcastType::SCALAR, false, EltwiseBinaryReuseDestType::NONE, false>(false, 1, icb)));
    MATH((llk_math_eltwise_unary_datacopy_softmax_k_init()));
    MATH((sfpu::llk_math_sfpu_softmax_k_init<is_fp32_dest_acc_en>()));
}

template <int k = 8, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softmax_k(uint32_t icb) {
    // NOTE: k is the number of valid columns of the 16-wide row.
    // Columns >= k are padding: their input is ignored and their output is 0.0.

    // Unpack row0 to srcB
    UNPACK((llk_unpack_A<BroadcastType::SCALAR, false, EltwiseBinaryReuseDestType::NONE, false>(icb, 0)));

    // B2D direct copy and scalar broadcast.
    MATH((llk_math_eltwise_unary_datacopy_softmax_k(0)));

    // Softmax
    MATH((sfpu::llk_math_sfpu_softmax_k<k, is_fp32_dest_acc_en>()));
}

#endif

}  // namespace ckernel
