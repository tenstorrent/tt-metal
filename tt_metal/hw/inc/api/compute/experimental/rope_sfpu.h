// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"

// Blackhole-only: the SFPU rope LLK lives only in the Blackhole llk_lib.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "sfpu/experimental/ckernel_sfpu_rope.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// Kept for callers that size DEST slots against the rope's tile geometry.
constexpr std::uint32_t ROPE_SFPU_TILE_ROWS = 64;

ALWI void rope_sfpu_init() { MATH((sfpu::sfpu_rope_configure_addrmod())); }

/**
 * Rotate in place, x tiles laid out at an arbitrary DEST stride.
 * Requires rope_sfpu_init(). See sfpu_rope_all_rows for the operand layout.
 *
 * With has_scale, scale_fp32 (an fp32 bit pattern, read from L1 at runtime) is
 * folded into cos/sin for a deferred normalization. scale_fp32 is required, not
 * defaulted: a missing value under has_scale would scale cos/sin by zero and
 * silently zero every output. Pass 0 explicitly when has_scale is false.
 */
template <
    std::uint32_t Ht,
    std::uint32_t Wt,
    std::uint32_t x_base,
    std::uint32_t x_stride,
    std::uint32_t cos_base,
    std::uint32_t sin_base,
    std::uint32_t cs_stride,
    bool has_scale = false>
ALWI void rope_sfpu_inplace_rows(const std::uint32_t scale_fp32) {
    static_assert(Wt == 1 || Wt == 2, "rope_sfpu: Wt must be 1 or 2 (decode rotary head_dim <= 64)");
    static_assert((x_base % 4) == 0 && (x_stride % 4) == 0, "x rows must be 4-row aligned");
    MATH((sfpu::sfpu_rope_dest_setup()));
    MATH((sfpu::sfpu_rope_all_rows<Ht, Wt, x_base, x_stride, cos_base, sin_base, cs_stride, has_scale>(scale_fp32)));
}

/**
 * Standalone form: x in Tile32x32 slots [0, Ht*Wt), cos in [Ht*Wt, +Wt), sin in
 * [Ht*Wt+Wt, +Wt) — the layout the RopeSfpu micro-op copy_tile's into DEST.
 */
template <std::uint32_t Ht, std::uint32_t Wt>
ALWI void rope_sfpu_inplace() {
    static_assert(Ht * Wt + 2 * Wt <= 8, "rope_sfpu: x + cos + sin must fit the 8 Tile32x32 slots of half DEST");
    constexpr std::uint32_t T = ROPE_SFPU_TILE_ROWS;
    rope_sfpu_inplace_rows<Ht, Wt, 0, T, (Ht * Wt) * T, (Ht * Wt + Wt) * T, T>(/*scale_fp32=*/0);
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
