// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14c kda_rec v2 (REC_V2): the KDA decode recurrence of one item (user b, head h) as fused SFPI passes over the
// item's 16 fp32 state tiles in DST, instead of ~29 broadcast tile-ops per 4-tile column block (v1).
//
// DST (8 fp32 tiles, dst_full_sync): slots 0..3 = S[mi][ni] for the current key tile mi (slot = value tile ni);
// slot 4 = the item's compact vector tile (VEC); slot 5 = the contraction accumulators (ACC; the norm weight in the
// norm window); slot 6 = per-band broadcasts + lane masks (BC); slot 7 = o and the delta patterns (OD).
// A DST "plane" p of slot s is sfpi::dst_reg[32 s + p]: 32 lanes = 4 sub-rows i x 8 SFPU columns c; in a state tile
// plane p = (2 fr + fc) * 8 + 2 band + par holds tile rows 16 fr + 4 band + i, columns 16 fc + 2 c + par
// (kda_rec_common.hpp plane_off, the mhc_pre plane layout). Column group cg = 2 fc + par; row band bb = 4 fr + band.
// Compact vector planes (VEC, written by the v2 reader):
//   key side (q, k, g): value r of the head (0..127) -> plane BASE + r / 32, lane (i = r % 4, c = (r / 4) % 8): the 4
//       rows of band bb of key tile mi sit in SFPU column bb of plane BASE + mi, one per sub-row;
//   value side ("u-layout": v, gate_out, the o_norm weight, o, u, w): value j -> plane BASE + j / 32, lane (sub-row
//       cg(j), column (j % 32 / 2) % 8): the 8 columns of column group cg of value tile ni in sub-row cg.
// Math per item (reference kda_rec.py reference_step; fp32, FMA rounding):
//   d = exp(g); qn = q / sqrt(|q|^2 + 1e-6) / sqrt(128); kn = k / sqrt(|k|^2 + 1e-6); qk = qn . kn
//   u_j = sum_i (kn_i d_i) S_ij; w_j = sum_i (qn_i d_i) S_ij;  delta = (v - u) * beta;  o = w + qk delta
//   S'_ij = S_ij d_i + kn_i delta_j;  on = o / sqrt(mean(o^2) + eps) * w_norm * sigmoid(gate_out)
// (o = qn . S' expanded: qn . (d S) + (qn . kn) delta.) Cross-lane steps: a sub-row sum / spread by the cyclic
// rotate-and-add over the 8 SFPU columns (SFPSHFT2 subvec rotate), a sum over sub-rows by the 4 x 4 sub-row
// transpose of 4 registers (SFPTRANSP).
#pragma once

// the plane map (every TRISC compiles it: the compute kernel names these outside MATH(...))
namespace ckernel::sfpu::kdav2 {
constexpr int PT = 32;  // planes per DST tile
constexpr int VEC = 4 * PT, ACC = 5 * PT, BC = 6 * PT, OD = 7 * PT;
// VEC planes: inputs (reader) q 0..3, k 4..7, g 8..11, v 12..15, gate_out 16..19, beta 20 (every lane); prep
// outputs qd = qn d (0..3), kd = kn d (4..7), d (8..11), kn 21..24, qk 25 (every lane)
constexpr int VQ = 0, VK = 4, VG = 8, VV = 12, VGO = 16, VBETA = 20, VKN = 21, VQK = 25;
constexpr int BC_A = 0, BC_B = 8, BC_MASK = 16;  // BC: broadcasts A / B of the current key tile, column masks
constexpr int OD_O = 0, OD_D = 4;                // OD: o (u-layout, per ni), delta pattern of (ni, cg) at 4 + 4 ni + cg
constexpr int OD_U = 28, OD_W = 29, OD_DL = 30;  // OD scratch planes: u / w of the current ni (the norm's 1/rms), delta
constexpr int VT0 = 26, VT1 = 27;                // VEC scratch planes: 1 / |q| * scale, 1 / |k|
// state-tile plane of row band bb (0..7) and column group cg (0..3)
constexpr int sp(int bb, int cg) { return (2 * (bb >> 2) + (cg >> 1)) * 8 + 2 * (bb & 3) + (cg & 1); }
// the plane holding tile row 0 of column group cg
constexpr int row0(int cg) { return (cg >> 1) * 8 + (cg & 1); }
}  // namespace ckernel::sfpu::kdav2

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat kdav2_ror(sfpi::vFloat v) { return sfpi::vFloat(sfpi::subvec_shflror1(v)); }

// every lane of a sub-row <- the sum of that sub-row's 8 lanes (cyclic doubling; also spreads a single non-zero lane)
sfpi_inline sfpi::vFloat kdav2_rowsum8(sfpi::vFloat v) {
    v = v + kdav2_ror(v);
    v = v + kdav2_ror(kdav2_ror(v));
    const sfpi::vFloat t = kdav2_ror(kdav2_ror(v));
    v = v + kdav2_ror(kdav2_ror(t));
    return v;
}

// SFPTRANSP transposes BOTH register groups L0..L3 and L4..L7, but sfpi models only its 4 operands: a value the
// compiler keeps in L4..L7 across a subvec_transp is silently corrupted. Rule for every function below: across a
// transpose nothing is live but its 4 operands — anything needed afterwards is stored to a DST plane before and
// reloaded after (checked on the disassembly: no L4..L7 read after an sfptransp before a write).

// DST plane P <- every lane = the sum of all 32 lanes of v
sfpi_inline void kdav2_total(const int P, sfpi::vFloat v) {
    v = kdav2_rowsum8(v);
    sfpi::vFloat a = v, b = v, c = v, d = v;
    sfpi::subvec_transp(a, b, c, d);
    sfpi::dst_reg[P] = (a + b) + (c + d);
}

// DST plane P <- every lane = the sum over the 4 sub-rows of the planes P0..P0+3 (lane c of sub-row i of the result =
// sum_k lane c of sub-row i of the transposed set: the column-group sums of the u-layout)
sfpi_inline void kdav2_subrow_sum(const int P0, const int P) {
    sfpi::vFloat a = sfpi::dst_reg[P0 + 0], b = sfpi::dst_reg[P0 + 1];
    sfpi::vFloat c = sfpi::dst_reg[P0 + 2], d = sfpi::dst_reg[P0 + 3];
    sfpi::subvec_transp(a, b, c, d);
    sfpi::dst_reg[P] = (a + b) + (c + d);
}

// DST planes Q0..Q3 <- the 4 x 4 sub-row transpose of 4 copies of plane P (Qk: sub-row k of P in every sub-row)
sfpi_inline void kdav2_spread4(const int P, const int Q0, const int Q1, const int Q2, const int Q3) {
    sfpi::vFloat a = sfpi::dst_reg[P], b = sfpi::dst_reg[P], c = sfpi::dst_reg[P], d = sfpi::dst_reg[P];
    sfpi::subvec_transp(a, b, c, d);
    sfpi::dst_reg[Q0] = a;
    sfpi::dst_reg[Q1] = b;
    sfpi::dst_reg[Q2] = c;
    sfpi::dst_reg[Q3] = d;
}

// 1 / sqrt(x), x > 0: bit-trick seed + 3 Newton steps (literal constants only: no programmable constant register is
// touched, so no init ordering against the other SFPU phases of the kernel)
sfpi_inline sfpi::vFloat kdav2_rsqrt(const sfpi::vFloat x) {
    const sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f3759df) - i);
    const sfpi::vFloat hx = x * 0.5f;
    y = y * (1.5f - hx * y * y);
    y = y * (1.5f - hx * y * y);
    y = y * (1.5f - hx * y * y);
    return y;
}

sfpi_inline sfpi::vFloat kdav2_recip(const sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    sfpi::vFloat t = 2.0f - x * y;
    y = y * t;
    t = 2.0f - x * y;
    return y * t;
}

sfpi_inline sfpi::vFloat kdav2_sigmoid(const sfpi::vFloat z) {
    const sfpi::vFloat e = _sfpu_exp_fp32_accurate_<false>(sfpi::min(-z, 88.0f));
    return kdav2_recip(e + 1.0f);
}

// BC column masks: plane BC_MASK + c = 1.0 in SFPU column c of every sub-row, 0 elsewhere (vConstTileId = 2 x lane)
inline void kda_rec_v2_masks() {
    using namespace kdav2;
    const sfpi::vInt col = (sfpi::vConstTileId >> 1) & 7;
#pragma GCC unroll 8
    for (int c = 0; c < 8; c++) {
        sfpi::vFloat m = 0.0f;
        v_if(col == c) { m = 1.0f; }
        v_endif;
        sfpi::dst_reg[BC + BC_MASK + c] = m;
    }
}

// prep on the VEC tile: d, the q / k l2 norms, qd / kd / kn, qk; then the column masks
template <uint32_t SCALE_BITS>
inline void kda_rec_v2_prep() {
    using namespace kdav2;
#pragma GCC unroll 4
    for (int m = 0; m < 4; m++) {
        sfpi::dst_reg[VEC + VG + m] = _sfpu_exp_fp32_accurate_<false>(sfpi::dst_reg[VEC + VG + m]);
    }
    {
        sfpi::vFloat sq = 0.0f;
#pragma GCC unroll 4
        for (int m = 0; m < 4; m++) {
            const sfpi::vFloat q = sfpi::dst_reg[VEC + VQ + m];
            sq = sq + q * q;
        }
        kdav2_total(VEC + VT0, sq);
    }
    {
        sfpi::vFloat sk = 0.0f;
#pragma GCC unroll 4
        for (int m = 0; m < 4; m++) {
            const sfpi::vFloat k = sfpi::dst_reg[VEC + VK + m];
            sk = sk + k * k;
        }
        kdav2_total(VEC + VT1, sk);
    }
    sfpi::dst_reg[VEC + VT0] =
        kdav2_rsqrt(sfpi::vFloat(sfpi::dst_reg[VEC + VT0]) + 1e-6f) * Converter::as_float(SCALE_BITS);
    sfpi::dst_reg[VEC + VT1] = kdav2_rsqrt(sfpi::vFloat(sfpi::dst_reg[VEC + VT1]) + 1e-6f);
    {
        const sfpi::vFloat iq = sfpi::dst_reg[VEC + VT0];
        const sfpi::vFloat ik = sfpi::dst_reg[VEC + VT1];
        sfpi::vFloat qk = 0.0f;
#pragma GCC unroll 4
        for (int m = 0; m < 4; m++) {
            const sfpi::vFloat qn = sfpi::dst_reg[VEC + VQ + m] * iq;
            const sfpi::vFloat kn = sfpi::dst_reg[VEC + VK + m] * ik;
            const sfpi::vFloat d = sfpi::dst_reg[VEC + VG + m];
            qk = qk + qn * kn;
            sfpi::dst_reg[VEC + VKN + m] = kn;
            sfpi::dst_reg[VEC + VQ + m] = qn * d;
            sfpi::dst_reg[VEC + VK + m] = kn * d;
        }
        sfpi::dst_reg[VEC + VQK] = qk;
    }
    kdav2_total(VEC + VQK, sfpi::dst_reg[VEC + VQK]);
    kda_rec_v2_masks();
}

// BC_A / BC_B <- the per-band broadcasts of VEC planes A + MI / B + MI (key tile MI): plane bb holds, in every lane of
// sub-row i, the value of row 4 (bb % 4) + i of face row bb / 4 (spread of SFPU column bb)
template <int MI, int A, int B>
inline void kda_rec_v2_bcast() {
    using namespace kdav2;
#pragma GCC unroll 8
    for (int bb = 0; bb < 8; bb++) {
        const sfpi::vFloat m = sfpi::dst_reg[BC + BC_MASK + bb];
        sfpi::dst_reg[BC + BC_A + bb] = kdav2_rowsum8(sfpi::dst_reg[VEC + A + MI] * m);
        sfpi::dst_reg[BC + BC_B + bb] = kdav2_rowsum8(sfpi::dst_reg[VEC + B + MI] * m);
    }
}

// pass 1 over key tile mi (slots 0..3): ACC[4 ni + cg] (+)= sum_bands S * kd, ACC[16 + 4 ni + cg] (+)= sum_bands S * qd
// (per lane: sub-row = row in band, column = column in group)
template <bool FIRST>
inline void kda_rec_v2_pass1() {
    using namespace kdav2;
#pragma GCC unroll 4
    for (int ni = 0; ni < 4; ni++) {
#pragma GCC unroll 4
        for (int cg = 0; cg < 4; cg++) {
            sfpi::vFloat au = 0.0f, aw = 0.0f;
#pragma GCC unroll 8
            for (int bb = 0; bb < 8; bb++) {
                const sfpi::vFloat s = sfpi::dst_reg[ni * PT + sp(bb, cg)];
                au = au + s * sfpi::dst_reg[BC + BC_A + bb];
                aw = aw + s * sfpi::dst_reg[BC + BC_B + bb];
            }
            if constexpr (FIRST) {
                sfpi::dst_reg[ACC + 4 * ni + cg] = au;
                sfpi::dst_reg[ACC + 16 + 4 * ni + cg] = aw;
            } else {
                sfpi::dst_reg[ACC + 4 * ni + cg] = sfpi::dst_reg[ACC + 4 * ni + cg] + au;
                sfpi::dst_reg[ACC + 16 + 4 * ni + cg] = sfpi::dst_reg[ACC + 16 + 4 * ni + cg] + aw;
            }
        }
    }
}

// after pass 1: u, w (u-layout) by the sub-row transpose, delta, o -> OD; the delta pattern of each (ni, cg) (its 8
// values in every sub-row) -> OD 4 + 4 ni + cg
inline void kda_rec_v2_mid() {
    using namespace kdav2;
#pragma GCC unroll 4
    for (int ni = 0; ni < 4; ni++) {
        kdav2_subrow_sum(ACC + 4 * ni, OD + OD_U);
        kdav2_subrow_sum(ACC + 16 + 4 * ni, OD + OD_W);
        {
            const sfpi::vFloat v = sfpi::dst_reg[VEC + VV + ni], u = sfpi::dst_reg[OD + OD_U];
            const sfpi::vFloat w = sfpi::dst_reg[OD + OD_W];
            const sfpi::vFloat dl = (v - u) * sfpi::dst_reg[VEC + VBETA];
            sfpi::dst_reg[OD + OD_O + ni] = w + sfpi::dst_reg[VEC + VQK] * dl;
            sfpi::dst_reg[OD + OD_DL] = dl;
        }
        kdav2_spread4(
            OD + OD_DL, OD + OD_D + 4 * ni + 0, OD + OD_D + 4 * ni + 1, OD + OD_D + 4 * ni + 2, OD + OD_D + 4 * ni + 3);
    }
}

// pass 2 over key tile mi (slots 0..3; BC_A = d, BC_B = kn broadcasts of mi): S <- S d + kn delta
inline void kda_rec_v2_pass2() {
    using namespace kdav2;
#pragma GCC unroll 4
    for (int ni = 0; ni < 4; ni++) {
#pragma GCC unroll 4
        for (int cg = 0; cg < 4; cg++) {
            const sfpi::vFloat dl = sfpi::dst_reg[OD + OD_D + 4 * ni + cg];
#pragma GCC unroll 8
            for (int bb = 0; bb < 8; bb++) {
                sfpi::vFloat s = sfpi::dst_reg[ni * PT + sp(bb, cg)];
                s = s * sfpi::dst_reg[BC + BC_A + bb];
                s = s + sfpi::dst_reg[BC + BC_B + bb] * dl;
                sfpi::dst_reg[ni * PT + sp(bb, cg)] = s;
            }
        }
    }
}

// gated norm: on = o / sqrt(mean(o^2) + eps) * w * sigmoid(gate_out) (o: OD 0..3, w: slot 5 planes 0..3, gate_out: VEC
// 16..19, all u-layout) -> row 0 of slots 0..3 (value tile ni in slot ni; the other rows are don't-care)
template <uint32_t EPS_BITS, uint32_t INV_V_BITS>
inline void kda_rec_v2_norm() {
    using namespace kdav2;
    {
        sfpi::vFloat ss = 0.0f;
#pragma GCC unroll 4
        for (int ni = 0; ni < 4; ni++) {
            const sfpi::vFloat o = sfpi::dst_reg[OD + OD_O + ni];
            ss = ss + o * o;
        }
        kdav2_total(OD + OD_U, ss);
    }
    sfpi::dst_reg[OD + OD_U] = kdav2_rsqrt(
        sfpi::vFloat(sfpi::dst_reg[OD + OD_U]) * Converter::as_float(INV_V_BITS) + Converter::as_float(EPS_BITS));
#pragma GCC unroll 4
    for (int ni = 0; ni < 4; ni++) {
        {
            sfpi::vFloat on = sfpi::vFloat(sfpi::dst_reg[OD + OD_O + ni]) * sfpi::dst_reg[OD + OD_U];
            on = on * sfpi::dst_reg[ACC + ni];
            on = on * kdav2_sigmoid(sfpi::dst_reg[VEC + VGO + ni]);
            sfpi::dst_reg[OD + OD_DL] = on;
        }
        kdav2_spread4(OD + OD_DL, ni * PT + row0(0), ni * PT + row0(1), ni * PT + row0(2), ni * PT + row0(3));
    }
}

// REC_V2_PROBE (diagnostics, numerics invalid): slot 0 planes 0..7 <- the cross-lane primitives applied to the raw v
// planes VEC 12..15 (host-controlled): 0 ror(v0), 1 rowsum8(v0), 2..5 transpose(v0..v3), 6 the column-3 mask, 7 spread
// of column 3 of v0 — packed as state tile (0, 0) of the item and compared on the host with the emulator's primitives
inline void kda_rec_v2_probe() {
    using namespace kdav2;
    const sfpi::vFloat a = sfpi::dst_reg[VEC + VV + 0];
    sfpi::dst_reg[0] = kdav2_ror(a);
    sfpi::dst_reg[1] = kdav2_rowsum8(a);
    sfpi::vFloat t0 = sfpi::dst_reg[VEC + VV + 0], t1 = sfpi::dst_reg[VEC + VV + 1];
    sfpi::vFloat t2 = sfpi::dst_reg[VEC + VV + 2], t3 = sfpi::dst_reg[VEC + VV + 3];
    sfpi::subvec_transp(t0, t1, t2, t3);
    sfpi::dst_reg[2] = t0;
    sfpi::dst_reg[3] = t1;
    sfpi::dst_reg[4] = t2;
    sfpi::dst_reg[5] = t3;
    sfpi::dst_reg[6] = sfpi::vFloat(sfpi::dst_reg[BC + BC_MASK + 3]);
    // after the transpose: reload (a value kept in L4..L7 across SFPTRANSP is corrupted)
    sfpi::dst_reg[7] = kdav2_rowsum8(sfpi::dst_reg[VEC + VV + 0] * sfpi::dst_reg[BC + BC_MASK + 3]);
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
