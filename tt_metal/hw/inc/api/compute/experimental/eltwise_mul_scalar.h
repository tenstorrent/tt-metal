// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif

namespace ckernel {

// Blackhole-only: the HiFi dest-reuse init workaround below calls the Blackhole
// LLK primitive directly, and all current consumers are Blackhole kernels.
#if defined(ARCH_BLACKHOLE)

// ============================================================================
// Scalar broadcast multiply
// ============================================================================

/**
 * Short init for scalar broadcast multiply (assumes hw already configured)
 */
ALWI void deepseek_mul_bcast_scalar_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/**
 * Scalar broadcast multiply with configurable fp32 accumulation
 */
template <bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void deepseek_mul_tiles_bcast_scalar(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::SCALAR,
          fp32_dest_acc_en,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

// ============================================================================
// Binary dest reuse multiply
// ============================================================================

/**
 * Init for binary dest reuse multiply
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_tiles_init(uint32_t icb0, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, call_line);
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, binary_reuse_dest>(false, false, icb0)));
    // HiFi-only workaround (tt-blaze #1760). The shorthand
    // llk_math_eltwise_binary_init<...>(icb0, icb0) mis-specializes the tile
    // shape and corrupts silu(gate)*up on the HiFi path (fixed M2 MoE HiFi4
    // 0.70->0.9996). At HiFi, use the general init instead; LoFi keeps the
    // original shorthand so its codegen is byte-identical.
    //
    // The fidelity gate MUST stay INSIDE MATH(): MATH_FIDELITY is only defined for
    // the math thread (trisc1); referencing it on the unpack/pack threads
    // (trisc0/trisc2) fails to compile. The immediately-invoked lambda keeps every
    // MATH_FIDELITY use within the MATH()-elided math-thread build.
    MATH(([&]() {
        if constexpr (MATH_FIDELITY != MathFidelity::LoFi) {
            _llk_math_eltwise_binary_init_<
                EltwiseBinaryType::ELWMUL,
                BroadcastType::NONE,
                MATH_FIDELITY,
                binary_reuse_dest>(ckernel::DEFAULT_TENSOR_SHAPE, 0 /*acc_to_dest*/);
        } else {
            llk_math_eltwise_binary_init<
                EltwiseBinaryType::ELWMUL,
                BroadcastType::NONE,
                MATH_FIDELITY,
                binary_reuse_dest>(icb0, icb0, false /*acc_to_dest*/);
        }
    }()));
}

/**
 * Binary dest reuse multiply
 * dest[idst] = dest[idst] * cb[in_tile_index]
 */
template <
    bool fp32_dest_acc_en = DST_ACCUM_MODE,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_tiles(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, true, binary_reuse_dest>(icb, in_tile_index)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          fp32_dest_acc_en,
          MATH_FIDELITY,
          binary_reuse_dest>(icb, icb, idst, true /*clear_fp32_dst_acc*/)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
