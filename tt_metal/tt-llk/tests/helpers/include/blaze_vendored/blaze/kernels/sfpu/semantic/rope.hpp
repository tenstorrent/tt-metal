// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SEMANTIC LIFT of blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/
// common/inc/sfpu/experimental/ckernel_sfpu_rope.h (hand-scheduled raw-TTI original).
//
// Typed-SFPI body: same math, same DEST addressing, no raw TTI.  See
// SEMANTIC-LIFT.md in this directory for the correctness argument and the
// STATIC-ESTIMATE word-count comparison.  The original is byte-untouched.
//
// Addressing bridge (BlackholeA0 SFPLOAD, fp32-mode DEST view):
//   sfpi::dst_reg[i] emits SFPLOAD/SFPSTORE at address 2*i (SFP_DESTREG_STRIDE == 2).
//   The original addresses x_addr (even columns) and x_addr + 2 (odd columns);
//   x_addr is asserted 4-row aligned, so those are dst_reg[x_addr/2] and
//   dst_reg[x_addr/2 + 1].
//   The original's explicit FP16B load/store format (InstrModLoadStore::FP16B,
//   mod0 = 2) is reproduced with the DataLayout::F16b access view.
//
// The caller-side scaffolding (sfpu_rope_configure_addrmod, sfpu_rope_dest_setup)
// is NOT lifted: the typed body performs no DEST-counter increments, so it needs
// only the same dest-base setup the original needs (callers keep using the
// original's setup functions).

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel {
namespace sfpu {
namespace semantic {

namespace rope {
// DEST rows per Tile32x32 slot and per face (same constants as the original).
constexpr uint32_t TILE_ROWS = 64;
constexpr uint32_t FACE_ROWS = 16;
}  // namespace rope

/**
 * One face: 8 complex pairs of a [1, 32] tile, rotated by cos/sin.
 *
 *   x'_even = cos*x_even - sin*x_odd
 *   x'_odd  = sin*x_even + cos*x_odd
 *
 * xi = x_addr / 2 (dst_reg index of the face's even-parity vector).
 */
sfpi_inline void sfpu_rope_face(const std::uint32_t xi, sfpi::vFloat cs_cos, sfpi::vFloat cs_sin) {
    using namespace sfpi;
    vFloat x_even = dst_reg[xi].mode<sfpi::DataLayout::F16b>();
    vFloat x_odd = dst_reg[xi + 1].mode<sfpi::DataLayout::F16b>();

    // Same two-rounding dataflow as the original's MAD pairs:
    //   t = cos*x_even;  out_even = -(sin*x_odd) + t
    //   u = sin*x_even;  out_odd  =   cos*x_odd  + u
    vFloat out_even = cs_cos * x_even - cs_sin * x_odd;
    vFloat out_odd = cs_sin * x_even + cs_cos * x_odd;

    dst_reg[xi].mode<sfpi::DataLayout::F16b>() = out_even;
    dst_reg[xi + 1].mode<sfpi::DataLayout::F16b>() = out_odd;
}

/**
 * Ht*Wt x-tiles starting at DEST row ``x_base``, ``x_stride`` rows apart, against
 * Wt cos tiles at ``cos_base`` and Wt sin tiles at ``sin_base`` (``cs_stride``
 * apart).  Loop structure and parameters identical to the original
 * sfpu_rope_all_rows; cos/sin are loop-invariant across the Ht inner heads, so
 * their residency (the original's hand-hoisted LREG0/LREG1) is the compiler's
 * invariant-hoisting decision here.
 */
template <
    uint32_t Ht,
    uint32_t Wt,
    uint32_t x_base,
    uint32_t x_stride,
    uint32_t cos_base,
    uint32_t sin_base,
    uint32_t cs_stride,
    bool has_scale = false>
inline void sfpu_rope_all_rows(const std::uint32_t scale_fp32 = 0) {
    using namespace sfpi;
    constexpr uint32_t F = rope::FACE_ROWS;

    static_assert((F & 3) == 0, "face stride must keep rows 4-row aligned");
    static_assert((x_base & 3) == 0 && (x_stride & 3) == 0, "x rows must be 4-row aligned");
    static_assert(
        (cos_base & 3) == 0 && (sin_base & 3) == 0 && (cs_stride & 3) == 0, "cos/sin rows must be 4-row aligned");

    constexpr uint32_t head_stride = Wt * x_stride;

    for (std::uint32_t w = 0; w < Wt; w++) {
        for (std::uint32_t f = 0; f < 2; f++) {
            const std::uint32_t cs_off = w * cs_stride + f * F;
            // Even-parity FP16B load serves both x parities (interleaved Meta
            // layout, each angle duplicated across its pair — see the original).
            vFloat cs_cos = dst_reg[(cos_base + cs_off) / 2].mode<sfpi::DataLayout::F16b>();
            vFloat cs_sin = dst_reg[(sin_base + cs_off) / 2].mode<sfpi::DataLayout::F16b>();
            if constexpr (has_scale) {
                // Amortized over the Ht heads that reuse this cos/sin pair.
                const vFloat s = sfpi::as<vFloat>(vInt(static_cast<int>(scale_fp32)));
                cs_cos = cs_cos * s;
                cs_sin = cs_sin * s;
            }
            std::uint32_t x_addr = x_base + w * x_stride + f * F;
            for (std::uint32_t h = 0; h < Ht; h++) {
                sfpu_rope_face(x_addr / 2, cs_cos, cs_sin);
                x_addr += head_stride;
            }
        }
    }
}

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
