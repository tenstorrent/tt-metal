// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CROSS-LANE MIGRATION of the softmax_k semantic lift (lane FK, 2026-08-22).
// Bridge-lift parent (byte-untouched): semantic/softmax_k.hpp (lane EX).
// Original hand kernel (byte-untouched):
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_softmax_k.h
//
// This file re-spells the lift on the GRADUATED typed cross-lane surface
// (sfpi_crosslane.h, lane FA X1) instead of the sfpu_bridge.hpp kit, so the
// rvtt-crosslane-lower pass (lane FG X4, -mtt-tensix-optimize-crosslane) sees
// the canonical inline frames:
//   * ror1/ror1_ip chains        -> sfpi::subvec_rotr<K> (R1 rotate frames)
//   * vConstTileId bit tests     -> sfpi::lane_col() predicates (the
//                                   canonical CC-compare seed the pass pins)
// Arithmetic order, rotation distances, predicate lane sets, Dst addresses
// and the SFPARECIP recip ladder are IDENTICAL to the bridge lift: at flag
// OFF the two bodies must be CRAQ bit-exact (gate) and near-identical in
// .text (the only intended word delta is the lane_col() shift in the
// predicate seeds).
//
// Documented intentional differences vs the bridge lift (value-identical):
//  M1 rotate chains are spelled subvec_rotr<K> (plain-form chains); the
//     bridge used ror1 + ror1_ip.  Same SFPSHFT2 SUBVEC_SHFLROR1 words;
//     register shapes are the allocator's (pressure here is far below 8).
//  M2 both column predicates are computed from sfpi::lane_col() instead of
//     raw vConstTileId bit tests -- same lane sets (LTILEID == 2*lane_id):
//     (TileId & 15) == 0  <=>  lane_col() == 0,
//     ((TileId >> 1) & 7) == c  <=>  lane_col() == c.
//
// S1-S4 (vs the hand original) are inherited from the bridge lift verbatim.

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"  // production typed exp (calculate_exponential, exp_init)

namespace ckernel {
namespace sfpu {
namespace semantic {
namespace crosslane {

sfpi_inline void _init_semantic_softmax_k_()
{
    // Identical to the original's _init_softmax_k_.
    sfpu::exp_init<false, 0x3F800000, true, DST_ACCUM_MODE>();
}

// For odd k, the final valid even lane predicates its paired odd tail lane;
// clear that extra exponential before it contributes to the row sum.
// Migrated mechanism: the typed column predicate (sfpi::lane_col()).
template <int k>
sfpi_inline void _semantic_zero_paired_odd_tail_lane_()
{
    if constexpr ((k & 1) && k < 16) {
        constexpr int tail_column = (k - 1) / 2;
        v_if (sfpi::lane_col() == tail_column) {
            sfpi::dst_reg[1] = sfpi::vFloat(0.0f);  // odd-cols half, addr 2
        }
        v_endif;
    }
}

template <int k>
inline void _semantic_softmax_k_()
{
    // x - max(x), predicated on |x_even| > 0 (the original's SFPABS + SFPGT
    // SET_CC window; the window deliberately covers the exponential too).
    sfpi::vFloat a = sfpi::dst_reg[0];  // even columns, addr 0
    sfpi::vFloat b = sfpi::dst_reg[1];  // odd columns,  addr 2
    v_if (sfpi::abs(a) > 0.0f) {
        sfpi::vFloat m = sfpi::dst_reg[4];  // row max, addr 8
        a = a - m;
        b = b - m;
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg[1] = b;

        // exp(x - max(x)) — the same production typed kernel the original
        // calls, under the same lane predicate.
        sfpu::calculate_exponential<
            false,
            DST_ACCUM_MODE,
            false,  // scaling
            2,      // iterations
            true    // clamp negatives
            >();
    }
    v_endif;

    math::clear_dst_reg_addr();
    _semantic_zero_paired_odd_tail_lane_<k>();

    // sum(exp(x - max(x))) within each subvector row (S2: the original's
    // dead LREG4=0 companion fold is dropped).  Rotation-ladder tree with
    // the original's 4/2/1 stage order, spelled as canonical subvec_rotr
    // frames (M1).
    sfpi::vFloat s = sfpi::dst_reg[0];
    s = s + sfpi::vFloat(sfpi::dst_reg[1]);
    s = s + sfpi::subvec_rotr<4>(s);
    s = s + sfpi::subvec_rotr<2>(s);
    s = s + sfpi::subvec_rotr<1>(s);

    // Broadcast the column-0 sum across all eight SFPU columns (the
    // original's LREG_MASK build + masked rotate-add tree, stages 1/2/4).
    // Column-0 predicate: sfpi::lane_col() == 0 (M2).
    sfpi::vFloat mask = 0.0f;
    v_if (sfpi::lane_col() == 0) {
        mask = 1.0f;
    }
    v_endif;
    s = s * mask;
    s = s + sfpi::subvec_rotr<1>(s);
    s = s + sfpi::subvec_rotr<2>(s);
    s = s + sfpi::subvec_rotr<4>(s);

    // 1 / sum — the original's SFPARECIP seed + 2 correction MADs + clamp,
    // op for op.  SFPARECIP has no typed wrapper (direct builtin, mod1
    // RECIP=0, SFPARECIP.md).
    sfpi::vFloat seed = sfpi::vFloat(__builtin_rvtt_sfparecip(s.get(), 0));
    sfpi::vFloat e = -s * seed + 1.0f;      // MAD, NEGATE_VA
    sfpi::vFloat e3 = e * e + e;            // MAD
    e3 = e3 * e + e;                        // MAD
    e3 = sfpi::min(e3, sfpi::vFloat(1.0f)); // SFPSWAP VEC_MIN_MAX vs LCONST_1
    sfpi::vFloat recip = e3 * seed + seed;  // MAD

    // Normalize both column groups and write the completed softmax to row 0.
    sfpi::vFloat r0 = sfpi::dst_reg[0];
    sfpi::vFloat r1 = sfpi::dst_reg[1];
    r0 = r0 * recip;
    r1 = r1 * recip;
    sfpi::dst_reg[0] = r0;
    sfpi::dst_reg[1] = r1;
}

}  // namespace crosslane
}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
