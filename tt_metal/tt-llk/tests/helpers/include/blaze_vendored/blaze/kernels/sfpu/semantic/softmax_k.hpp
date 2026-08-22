// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SEMANTIC LIFT of softmax_k (lane EX, 2026-08-21).  Original
// (byte-untouched):
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_softmax_k.h
//
// EW's census refused this kernel on the SFPCONFIG-LREG11 per-instance
// lane-mask choreography.  The lift DISSOLVES that choreography instead of
// bridging it: LReg[11]'s role here is purely to materialize the per-column
// predicate "SFPU column == (k-1)/2" (SFPCONFIG.md: Imm16-lane-mask bit
// (Lane&7)*2 selects the column; the value is vertically broadcast) — which
// the type system can spell directly from the tile-id constant
// (LTILEID == 2*lane_id, so column == lane%8 == (TileId&15)==0 tests).
// Everything else is plain typed sfpi plus TWO bridges: SFPARECIP (no typed
// wrapper) and the production calculate_exponential (already typed, reused
// by identity — the original calls the same function).
//
// Documented intentional differences (value-identical, see SEMANTIC-LIFT.md):
//  S1 the LREG11 SFPCONFIG program/restore pair is dropped entirely: the lift
//     never touches LReg[11], so the original's restore of the -1.0 default
//     has nothing to restore.  The zeroing predicate is computed from
//     vConstTileId instead (same lane set, proven above).
//  S2 the dead second-half fold is dropped: the original loads LREG4 = 0 and
//     runs horizontal_reduce's L4/L5 leg on it; the result is never read.
//  S3 the CC window is spelled v_if/v_endif (PUSHC/SETCC/POPC) instead of the
//     original's bare SFPSETCC..SFPENCC; the enabled-lane set inside the
//     window and the all-enabled state after it are identical.
//  S4 SFPNOP scheduling filler is not spelled (the compiler owns scheduling;
//     BH stalls dynamically — swarm-established).

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"  // production typed exp (calculate_exponential, exp_init)
#include "blaze/kernels/sfpu/semantic/sfpu_bridge.hpp"

namespace ckernel {
namespace sfpu {
namespace semantic {

sfpi_inline void _init_semantic_softmax_k_()
{
    // Identical to the original's _init_softmax_k_.
    sfpu::exp_init<false, 0x3F800000, true, DST_ACCUM_MODE>();
}

// For odd k, the final valid even lane predicates its paired odd tail lane;
// clear that extra exponential before it contributes to the row sum.
// Original mechanism: LReg[11] = 1.0 on SFPU column (k-1)/2 via SFPCONFIG
// imm-lane-mask, then SFPSETCC on LReg[11] != 0.  Lifted mechanism: the same
// column predicate from the tile-id constant.
template <int k>
sfpi_inline void _semantic_zero_paired_odd_tail_lane_()
{
    if constexpr ((k & 1) && k < 16) {
        constexpr int tail_column = (k - 1) / 2;
        sfpi::vInt col = (sfpi::vInt(sfpi::vConstTileId) >> 1) & 7;
        v_if (col == tail_column) {
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
    // dead LREG4=0 companion fold is dropped).
    sfpi::vFloat s = sfpi::dst_reg[0];
    s = s + sfpi::vFloat(sfpi::dst_reg[1]);
    {
        // horizontal_reduce phases: rotate by 4 / 2 / 1, adding.
        sfpi::vFloat t = ror1(s);
        ror1_ip(t);
        ror1_ip(t);
        ror1_ip(t);
        s = s + t;
        t = ror1(s);
        ror1_ip(t);
        s = s + t;
        t = ror1(s);
        s = s + t;
    }

    // Broadcast the column-0 sum across all eight SFPU columns (the
    // original's LREG_MASK build + masked rotate-add tree, stages 1/2/4).
    // Column-0 predicate: LTILEID == 2*lane, so (TileId & 15) == 0 <=>
    // lane % 8 == 0 (exactly _build_lane_mask_col0_'s shift-by-28 test).
    sfpi::vFloat mask = 0.0f;
    v_if ((sfpi::vInt(sfpi::vConstTileId) & 15) == 0) {
        mask = 1.0f;
    }
    v_endif;
    s = s * mask;
    {
        sfpi::vFloat t = ror1(s);
        s = s + t;
        t = ror1(s);
        ror1_ip(t);
        s = s + t;
        t = ror1(s);
        ror1_ip(t);
        ror1_ip(t);
        ror1_ip(t);
        s = s + t;
    }

    // 1 / sum — the original's SFPARECIP seed + 2 correction MADs + clamp,
    // op for op.  SFPARECIP has no typed wrapper (bridge builtin, mod1
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

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
