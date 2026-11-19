// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// SFPU sweeps of the fused iHC site (ihc_site.cpp), math thread only. Everything the site computes on values that
// must stay fp32 runs here: the streams never pass through SrcA / SrcB (19-bit), only through DEST and the SFPU's
// fp32 registers.
//
// DEST is used as 8 fp32 page slots (full sync, never cleared by the packer):
//   0..3  the 4 streams pages of a batch (a batch is 4 chunks of 32 hidden columns), or the exchange pages
//   4     the rows page of the batch: the sublayer output F going in, the collapsed input y coming out
//   5     scratch: the projection result, the partial sums, the parametrisation
//   6     the coefficient page: unit 2 o + g = coefficient o (pre 0..3, post 4..7) of token rows 4 g .. 4 g + 3,
//         each row's value on all 8 lanes
//   7     the site's constants (layout.py const_page)
// Addresses are sfpi::dst_reg indices ("units", 2 DEST rows each): a slot is 32 units, a face 8. One unit is one
// SFPU register: 4 DEST rows x 8 lanes (layout.py). A face of a streams / rows page holds token rows 0..7 for the
// first 16 columns in DEST rows 0..7 and for the last 16 in rows 8..15, so unit 2 (2 h + g) + p of a face is token
// rows 4 g .. 4 g + 3, column half h, lane set p.

#pragma once

#ifdef TRISC_MATH

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_sqrt.h"
#include "sfpi.h"

namespace ihc {

constexpr int SLOT = 32;
constexpr int FACE = 8;

constexpr int U_ROWS = 4 * SLOT;
constexpr int U_SCR = 5 * SLOT;
constexpr int U_COEF = 6 * SLOT;
constexpr int U_CONST = 7 * SLOT;

// scratch slot
constexpr int U_D = U_SCR;                    // face 0: the projection, [16, 16]
constexpr int U_PART = U_SCR + FACE;          // face 1 rows 0..7: mixes g0, ss g0, mixes g1, ss g1 (sent to every core)
constexpr int U_ACCS = U_SCR + 2 * FACE;      // + g: lane sums of the streams' squares
constexpr int U_ACCY = U_SCR + 2 * FACE + 2;  // + g: lane sums of y^2 (face 2 rows 4..7, sent to the root core)
// With the root exchange (ihc_site.cpp) the root core receives up to 6 exchange pages into slots 0..5, so what it
// derives from them lives outside the scratch slot: in the free half of the coefficient slot.
constexpr bool ROOT_EXCHANGE = get_named_compile_time_arg_val("root_exchange") == 1;
constexpr int U_MIX = ROOT_EXCHANGE ? U_COEF + 2 * FACE : U_SCR + 3 * FACE;  // + g: the normalised mixes
constexpr int U_G = U_MIX + 2;                                               // + g: pre | post, lane o = output o

// constant page (layout.py CONST_*)
constexpr int U_SCALE = U_CONST + 0;
constexpr int U_BASE = U_CONST + 1;
constexpr int U_GAIN = U_CONST + 2;
constexpr int U_EPS = U_CONST + 3;
constexpr int U_NORM_EPS = U_CONST + 4;
constexpr int U_INV_N = U_CONST + 5;
constexpr int U_INV_C = U_CONST + 6;
constexpr int U_LANE = U_CONST + 8;

// statistic page while the root core builds it (slot 1; slot 4 with the root exchange, whose sums of y^2 can fill
// slots 0..2): 1 / RMS(y) of token rows 4 g .. at unit 2 g, all lanes
constexpr int STAT_SLOT = ROOT_EXCHANGE ? 4 : 1;
constexpr int U_STAT = STAT_SLOT * SLOT;

// unit of token-row group g, column half h, lane set p inside a face
constexpr int cell(int g, int h, int p) { return 2 * (2 * h + g) + p; }

// Sum of the 8 lanes of every row, left on all 8 lanes (three rotate-and-add stages). A register with one non-zero
// lane per row comes out as that lane's value on every lane, exactly.
sfpi_inline sfpi::vFloat hsum8(sfpi::vFloat v) {
    sfpi::vFloat r = sfpi::subvec_shflror1(v);
    v = v + r;
    r = sfpi::subvec_shflror1(v);
    r = sfpi::subvec_shflror1(r);
    v = v + r;
    r = sfpi::subvec_shflror1(v);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    v = v + r;
    return v;
}

// A whole page from DEST slot FROM to slot TO (the low_unpack fallback of ihc_site.cpp: pages are unpacked into
// slots 0..3 only and moved up by the SFPU).
template <int FROM, int TO>
inline void copy_page() {
#pragma GCC unroll 32
    for (int u = 0; u < SLOT; u++) {
        sfpi::vFloat v = sfpi::dst_reg[FROM * SLOT + u];
        sfpi::dst_reg[TO * SLOT + u] = v;
    }
}

// The running sums start at zero; the projection accumulates into face 0 of the scratch slot.
inline void zero_state() {
    sfpi::vFloat z = 0.0f;
#pragma GCC unroll 8
    for (int u = 0; u < FACE; u++) {
        sfpi::dst_reg[U_D + u] = z;
    }
    sfpi::dst_reg[U_ACCS + 0] = z;
    sfpi::dst_reg[U_ACCS + 1] = z;
    sfpi::dst_reg[U_ACCY + 0] = z;
    sfpi::dst_reg[U_ACCY + 1] = z;
}

// The running sums of y^2 alone (root exchange: an exchange page may have passed through the scratch slot).
inline void zero_y2() {
    sfpi::vFloat z = 0.0f;
    sfpi::dst_reg[U_ACCY + 0] = z;
    sfpi::dst_reg[U_ACCY + 1] = z;
}

// Post-mix of one batch for token rows 4 G ..: stream j of slot k += post_j * F (slot 4, face k), in place; with
// WITH_SS the squares of the updated streams are summed per lane for the next pre-mix.
template <int G, bool WITH_SS>
inline void post_rows() {
    sfpi::vFloat p0 = sfpi::dst_reg[U_COEF + 2 * 4 + G];
    sfpi::vFloat p1 = sfpi::dst_reg[U_COEF + 2 * 5 + G];
    sfpi::vFloat p2 = sfpi::dst_reg[U_COEF + 2 * 6 + G];
    sfpi::vFloat p3 = sfpi::dst_reg[U_COEF + 2 * 7 + G];
    sfpi::vFloat acc = sfpi::dst_reg[U_ACCS + G];
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++) {
#pragma GCC unroll 4
        for (int hp = 0; hp < 4; hp++) {
            const int c = cell(G, hp >> 1, hp & 1);
            sfpi::vFloat f = sfpi::dst_reg[U_ROWS + FACE * k + c];
            {
                sfpi::vFloat x = sfpi::dst_reg[SLOT * k + 0 * FACE + c];
                x = f * p0 + x;
                sfpi::dst_reg[SLOT * k + 0 * FACE + c] = x;
                if constexpr (WITH_SS) {
                    acc = x * x + acc;
                }
            }
            {
                sfpi::vFloat x = sfpi::dst_reg[SLOT * k + 1 * FACE + c];
                x = f * p1 + x;
                sfpi::dst_reg[SLOT * k + 1 * FACE + c] = x;
                if constexpr (WITH_SS) {
                    acc = x * x + acc;
                }
            }
            {
                sfpi::vFloat x = sfpi::dst_reg[SLOT * k + 2 * FACE + c];
                x = f * p2 + x;
                sfpi::dst_reg[SLOT * k + 2 * FACE + c] = x;
                if constexpr (WITH_SS) {
                    acc = x * x + acc;
                }
            }
            {
                sfpi::vFloat x = sfpi::dst_reg[SLOT * k + 3 * FACE + c];
                x = f * p3 + x;
                sfpi::dst_reg[SLOT * k + 3 * FACE + c] = x;
                if constexpr (WITH_SS) {
                    acc = x * x + acc;
                }
            }
        }
    }
    if constexpr (WITH_SS) {
        sfpi::dst_reg[U_ACCS + G] = acc;
    }
}

// Lane sums of the squares of the 4 streams of one batch (a pre-mix that no post-mix precedes).
template <int G>
inline void ss_rows() {
    sfpi::vFloat acc = sfpi::dst_reg[U_ACCS + G];
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++) {
#pragma GCC unroll 4
        for (int hp = 0; hp < 4; hp++) {
            const int c = cell(G, hp >> 1, hp & 1);
#pragma GCC unroll 4
            for (int j = 0; j < 4; j++) {
                sfpi::vFloat x = sfpi::dst_reg[SLOT * k + j * FACE + c];
                acc = x * x + acc;
            }
        }
    }
    sfpi::dst_reg[U_ACCS + G] = acc;
}

// Pre-mix of one batch for token rows 4 G ..: y (slot 4, face k) = sum_j pre_j * stream j of slot k; y^2 summed per
// lane for the RMSNorm statistic.
template <int G>
inline void mix_rows() {
    sfpi::vFloat c0 = sfpi::dst_reg[U_COEF + 2 * 0 + G];
    sfpi::vFloat c1 = sfpi::dst_reg[U_COEF + 2 * 1 + G];
    sfpi::vFloat c2 = sfpi::dst_reg[U_COEF + 2 * 2 + G];
    sfpi::vFloat c3 = sfpi::dst_reg[U_COEF + 2 * 3 + G];
    sfpi::vFloat acc = sfpi::dst_reg[U_ACCY + G];
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++) {
#pragma GCC unroll 4
        for (int hp = 0; hp < 4; hp++) {
            const int c = cell(G, hp >> 1, hp & 1);
            sfpi::vFloat x = sfpi::dst_reg[SLOT * k + 0 * FACE + c];
            sfpi::vFloat y = x * c0;
            x = sfpi::dst_reg[SLOT * k + 1 * FACE + c];
            y = x * c1 + y;
            x = sfpi::dst_reg[SLOT * k + 2 * FACE + c];
            y = x * c2 + y;
            x = sfpi::dst_reg[SLOT * k + 3 * FACE + c];
            y = x * c3 + y;
            sfpi::dst_reg[U_ROWS + FACE * k + c] = y;
            acc = y * y + acc;
        }
    }
    sfpi::dst_reg[U_ACCY + G] = acc;
}

// This core's partial sums for the exchange. The projection of token row r onto output o is DEST [r][.] (first
// column half of every chunk) + DEST [8 + r][.] (second half) of the projection face, and fn^T is laid out
// (layout.py fn_to_pages) so that output o of half h is lane o of lane set h: the two blocks are lane aligned.
inline void partial_cells() {
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
        sfpi::vFloat a = sfpi::dst_reg[U_D + cell(g, 0, 0)];
        sfpi::vFloat b = sfpi::dst_reg[U_D + cell(g, 1, 1)];
        sfpi::dst_reg[U_PART + 2 * g] = a + b;
        sfpi::vFloat s = sfpi::dst_reg[U_ACCS + g];
        sfpi::dst_reg[U_PART + 2 * g + 1] = s;
    }
}

// mm32 (fallback projection): one full [32, 32] x [32, 32] matmul per page accumulated in slot 0, whose faces 0
// (streams 0, 1) and 3 (streams 2, 3) hold the blocks partial_cells reads from the projection face.
inline void zero_slot0() {
    sfpi::vFloat z = 0.0f;
#pragma GCC unroll 32
    for (int u = 0; u < SLOT; u++) {
        sfpi::dst_reg[u] = z;
    }
}

inline void partial_cells_mm32() {
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
        sfpi::vFloat a = sfpi::dst_reg[cell(g, 0, 0)];
        sfpi::vFloat b = sfpi::dst_reg[cell(g, 1, 1)];
        a = a + b;
        sfpi::vFloat c = sfpi::dst_reg[3 * FACE + cell(g, 0, 0)];
        sfpi::vFloat d = sfpi::dst_reg[3 * FACE + cell(g, 1, 1)];
        c = c + d;
        sfpi::dst_reg[U_PART + 2 * g] = a + c;
        sfpi::vFloat s = sfpi::dst_reg[U_ACCS + g];
        sfpi::dst_reg[U_PART + 2 * g + 1] = s;
    }
}

// SFPU projection (sfpu_proj in ihc_site.cpp): no value of the streams passes through the FPU's 19-bit operands.
// One streams page sits in slot 4, its four fn pages (layout.py fn_to_row_pages, one per stream) in slots 0..3.
// The running sums, per output o and token-row group g, are 16 cells of the constant page's free half (zero when
// the page is unpacked): lane sums over the columns, fp32.
constexpr int U_PACC = U_CONST + 2 * FACE;  // + 2 o + g

// Outputs 2 OP and 2 OP + 1 of one streams page: sum += streams cell * fn cell, both token-row groups.
template <int OP>
inline void proj_page() {
    constexpr int o0 = 2 * OP;
    constexpr int o1 = 2 * OP + 1;
    sfpi::vFloat a00 = sfpi::dst_reg[U_PACC + 2 * o0 + 0];
    sfpi::vFloat a01 = sfpi::dst_reg[U_PACC + 2 * o0 + 1];
    sfpi::vFloat a10 = sfpi::dst_reg[U_PACC + 2 * o1 + 0];
    sfpi::vFloat a11 = sfpi::dst_reg[U_PACC + 2 * o1 + 1];
#pragma GCC unroll 4
    for (int j = 0; j < 4; j++) {
#pragma GCC unroll 4
        for (int hp = 0; hp < 4; hp++) {
            sfpi::vFloat xa = sfpi::dst_reg[U_ROWS + j * FACE + cell(0, hp >> 1, hp & 1)];
            sfpi::vFloat xb = sfpi::dst_reg[U_ROWS + j * FACE + cell(1, hp >> 1, hp & 1)];
            sfpi::vFloat w = sfpi::dst_reg[j * SLOT + FACE * hp + o0];
            a00 = xa * w + a00;
            a01 = xb * w + a01;
            w = sfpi::dst_reg[j * SLOT + FACE * hp + o1];
            a10 = xa * w + a10;
            a11 = xb * w + a11;
        }
    }
    sfpi::dst_reg[U_PACC + 2 * o0 + 0] = a00;
    sfpi::dst_reg[U_PACC + 2 * o0 + 1] = a01;
    sfpi::dst_reg[U_PACC + 2 * o1 + 0] = a10;
    sfpi::dst_reg[U_PACC + 2 * o1 + 1] = a11;
}

// This core's partial for the exchange from the SFPU projection: each output's lane sums added up and placed on
// lane o of the mixes cell.
inline void partial_cells_sfpu() {
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
        sfpi::vFloat out = 0.0f;
#pragma GCC unroll 8
        for (int o = 0; o < 8; o++) {
            sfpi::vFloat v = sfpi::dst_reg[U_PACC + 2 * o + g];
            v = hsum8(v);
            sfpi::vFloat e = sfpi::dst_reg[U_LANE + o];
            out = v * e + out;
        }
        sfpi::dst_reg[U_PART + 2 * g] = out;
        sfpi::vFloat s = sfpi::dst_reg[U_ACCS + g];
        sfpi::dst_reg[U_PART + 2 * g + 1] = s;
    }
}

// Sum the K cores' partials (exchange pages in slots 0.., 4 units per core: mixes g0, ss g0, mixes g1, ss g1) and
// normalise the mixes: mixes * rsqrt(sum of squares / (n C) + eps). Needs sqrt_init<false>().
template <int K>
inline void param_norm() {
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
        sfpi::vFloat m = sfpi::dst_reg[2 * g];
        sfpi::vFloat s = sfpi::dst_reg[2 * g + 1];
#pragma GCC unroll 64
        for (int k = 1; k < K; k++) {
            sfpi::vFloat a = sfpi::dst_reg[4 * k + 2 * g];
            m = m + a;
            sfpi::vFloat b = sfpi::dst_reg[4 * k + 2 * g + 1];
            s = s + b;
        }
        s = hsum8(s);
        sfpi::vFloat inv_n = sfpi::dst_reg[U_INV_N];
        sfpi::vFloat eps = sfpi::dst_reg[U_NORM_EPS];
        s = s * inv_n + eps;
        sfpi::vFloat rs = ckernel::sfpu::_calculate_sqrt_body_<false, true, false>(s);
        sfpi::dst_reg[U_MIX + g] = m * rs;
    }
}

// pre = sigmoid(a_pre m + b) + eps on lanes 0..3, post = magnitude * sigmoid(a_post m + b) + eps on lanes 4..7.
// Needs sfpu_reciprocal_init<false>().
inline void param_sigmoid() {
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
        sfpi::vFloat m = sfpi::dst_reg[U_MIX + g];
        sfpi::vFloat a = sfpi::dst_reg[U_SCALE];
        sfpi::vFloat b = sfpi::dst_reg[U_BASE];
        sfpi::vFloat z = m * a + b;
        sfpi::vFloat sg = ckernel::sfpu::_sfpu_sigmoid_<true>(z);
        sfpi::vFloat gain = sfpi::dst_reg[U_GAIN];
        sfpi::vFloat eps = sfpi::dst_reg[U_EPS];
        sfpi::dst_reg[U_G + g] = sg * gain + eps;
    }
}

// Every coefficient on all 8 lanes of its own cell of the coefficient page.
inline void param_broadcast() {
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
#pragma GCC unroll 8
        for (int o = 0; o < 8; o++) {
            sfpi::vFloat v = sfpi::dst_reg[U_G + g];
            sfpi::vFloat e = sfpi::dst_reg[U_LANE + o];
            sfpi::dst_reg[U_COEF + 2 * o + g] = hsum8(v * e);
        }
    }
}

// Root core: 1 / RMS(y) per token row from the K cores' lane sums of y^2 (exchange page in slot 0, core k at units
// 2 k + g). Needs sqrt_init<false>().
template <int K>
inline void stat_cells() {
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
        sfpi::vFloat s = sfpi::dst_reg[g];
#pragma GCC unroll 64
        for (int k = 1; k < K; k++) {
            sfpi::vFloat b = sfpi::dst_reg[2 * k + g];
            s = s + b;
        }
        s = hsum8(s);
        sfpi::vFloat inv_c = sfpi::dst_reg[U_INV_C];
        sfpi::vFloat eps = sfpi::dst_reg[U_NORM_EPS];
        s = s * inv_c + eps;
        sfpi::dst_reg[U_STAT + 2 * g] = ckernel::sfpu::_calculate_sqrt_body_<false, true, false>(s);
    }
}

// Probe (test_probe_lanes): what one SFPU register holds and what the lane rotation does. In: a page in slot 0.
// Out (slot 6): unit 0 = the register loaded from unit 0, 1 = that register rotated once, 2 = the register loaded
// from unit 1, 3 = hsum8 of the unit 0 register, 4 = unit 0 register * 3 + its own value (one multiply-add),
// 5 = unit 0 of slot 7 (the page unpacked there), 6 = rsqrt, 7 = sigmoid of the unit 0 register, 8..10 = the
// unit 0 register after an SFPU store to and load from slots 7, 5, 4.
inline void probe_cells() {
    sfpi::vFloat a = sfpi::dst_reg[0];
    sfpi::dst_reg[U_COEF + 0] = a;
    sfpi::vFloat r = sfpi::subvec_shflror1(a);
    sfpi::dst_reg[U_COEF + 1] = r;
    sfpi::vFloat b = sfpi::dst_reg[1];
    sfpi::dst_reg[U_COEF + 2] = b;
    sfpi::dst_reg[U_COEF + 3] = hsum8(a);
    sfpi::vFloat three = 3.0f;
    sfpi::dst_reg[U_COEF + 4] = a * three + a;
    sfpi::vFloat c = sfpi::dst_reg[U_CONST];
    sfpi::dst_reg[U_COEF + 5] = c;
    // units 8..10: the unit 0 register stored to slots 7, 5 and 4 by the SFPU and read back (SFPU access to the
    // upper slots, apart from whether the unpacker reaches them)
    sfpi::dst_reg[U_CONST + 2 * FACE] = a;
    sfpi::dst_reg[U_SCR] = a;
    sfpi::dst_reg[U_ROWS] = a;
    sfpi::vFloat b7 = sfpi::dst_reg[U_CONST + 2 * FACE];
    sfpi::dst_reg[U_COEF + 8] = b7;
    sfpi::vFloat b5 = sfpi::dst_reg[U_SCR];
    sfpi::dst_reg[U_COEF + 9] = b5;
    sfpi::vFloat b4 = sfpi::dst_reg[U_ROWS];
    sfpi::dst_reg[U_COEF + 10] = b4;
}

// Second DEST section of the probe: slots 6 and 7 as the first section left them (nothing clears DEST in between).
inline void probe_persist() {
    sfpi::vFloat a = sfpi::dst_reg[U_COEF + 0];
    sfpi::dst_reg[U_STAT + 0] = a;
    sfpi::vFloat c = sfpi::dst_reg[U_CONST + 1];
    sfpi::dst_reg[U_STAT + 1] = c;
}

inline void probe_rsqrt() {
    sfpi::dst_reg[U_COEF + 6] = ckernel::sfpu::_calculate_sqrt_body_<false, true, false>(sfpi::dst_reg[0]);
}

inline void probe_sigmoid() { sfpi::dst_reg[U_COEF + 7] = ckernel::sfpu::_sfpu_sigmoid_<true>(sfpi::dst_reg[0]); }

}  // namespace ihc

#endif  // TRISC_MATH
