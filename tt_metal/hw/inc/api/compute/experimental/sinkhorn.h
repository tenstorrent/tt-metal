// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute-API layer for the sinkhorn 4x4 SFPU fast path. The kernels and the
// llk_math entry points live in
// `tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sinkhorn.h` and
// `llk_api/experimental/llk_sfpu/llk_math_sinkhorn.h`.

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/llk_math_sinkhorn.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// Public API (visible to all TRISCs; only TRISC_MATH does work).
// Unary SFPU init (ADDR_MOD_7 dest-incr 0). This op does not call it:
// exp_tile_init already programs the same slot immediately before
// sinkhorn_row_max_sub / sinkhorn_4x4.
inline void sinkhorn_4x4_init() { MATH((_llk_math_sinkhorn_4x4_init_())); }

// Comb 4x4 per-row max-sub. Runs after exp_tile_init (ADDR_MOD_7) and before
// exp_tile; must not clobber LREG12..14.
ALWI void sinkhorn_row_max_sub(std::uint32_t input_index) { MATH((_llk_math_sinkhorn_row_max_sub_(input_index))); }

// EPS_BITS is the fp32 bit pattern of the eps constant guarding 0/0 in the
// row/col reciprocals. The default preserves the pre-EPS_BITS-plumbing
// behavior (bf16 0x3589 == 1.001358e-06, zero-extended to fp32).
//
// NUM_FACES_USED enables faces as a prefix: face 0 is always processed, and
// faces 1..3 are added as the count rises. VALID_H / VALID_W describe the
// logical rectangle inside that face set, and the padding outside it is zeroed.
// The tt-blaze caller drives all six parameters from one op-level config, so
// the face count and the rectangle always agree; nothing here diagnoses a
// combination where the rectangle reaches past the selected faces, and a count
// of 0 still normalizes face 0.
//
// DEST is assumed to be 16-bit: the strip-pair store converts the fp32 LREG
// values with FP32_TO_FP16B on the way out, so an FP32 DEST would be addressed
// and rounded incorrectly. The mode is not threaded into this API.
template <
    std::uint32_t NUM_FACES_USED = 4,
    std::uint32_t ITERS = 20,
    std::uint32_t EPS_BITS = 0x35890000,
    bool SINGLE_SUBMAT = false,
    std::uint32_t VALID_H = 32,
    std::uint32_t VALID_W = 32>
ALWI void sinkhorn_4x4(std::uint32_t input_index) {
    MATH((_llk_math_sinkhorn_4x4_<NUM_FACES_USED, ITERS, EPS_BITS, SINGLE_SUBMAT, VALID_H, VALID_W>(input_index)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
