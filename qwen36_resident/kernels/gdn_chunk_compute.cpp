// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core compute of the chunked verify: the Gated DeltaNet step of one value head for the R rows of
// one user (positions p .. p + R - 1) in one pass, in the WY / chunked form on 32x32 tiles whose rows 0..R-1 are
// the R rows (the sequential form is gdn_head_compute.cpp). With S the committed state, k_r, q_r the normalized
// key / scaled query rows, g_r the decays, Gamma_r = g_0 .. g_r, beta_r = sigmoid(b_r):
//   G[r][j]  = Gamma_r / Gamma_j for j <= r (else 0)      = exp(c_r - c_j), c the cumulative log decays
//   N        = beta (x) (G strictly below the diagonal) (x) K K^T
//   Delta    = (I + N)^-1 beta (x) (V - Gamma (x) (K S))   ((I + N)^-1 = (I - N)(I + N^2)(I + N^4), N^8 = 0)
//   O        = Gamma (x) (Q S) + (G (x) Q K^T) Delta
// and the state after row a is S_a = Gamma_a S + sum_{j <= a} G[a][j] k_j^T delta_j. The step does not know a:
// it writes S (the new base) and the pending block [K^T, Delta, G, Gamma]; the next step, given a, commits
// S_a = Gamma_a S + (K^T (x) w) Delta (w: column j = G[a][j], from the reader) before its own rows. Then the gated
// RMSNorm and silu(z) as in the sequential form. State scaling runs on the SFPU in fp32 (UnpackToDestFp32 CBs), the
// matmuls and the other products on the FPU with fp32 accumulation; the cumulative sums of the log decays are split
// into a bf16 part and an fp32 remainder so that the 0/1 matmuls keep them to fp32 precision.
//
// CBs (reader -> compute): 0 q, 1 k, 2 v, 3 z (rows 0..R-1 of bf16 tiles), 4 norm weight, 5 base state (fp32,
// UnpackToDestFp32), 6 gates [a rows, a columns, b rows, dt_bias, neg_exp_A] (fp32, UnpackToDestFp32), 11 Gamma_a
// fill (UnpackToDestFp32), 12 w, 13 / 14 the pending K^T / Delta, 15 constants [ones, Tri, Tri^T, I, strict lower
// mask] (bf16), 16 causal additive mask (fp32, UnpackToDestFp32). Compute -> writer: 18 the new base state, 30 the
// output rows (bf16), 31 the pending block [K^T x4, Delta x4, G, Gamma]. Internal: 7 the layer's kept tiles [beta,
// Gamma, G, M, (I + N)^-1], 8 / 25 / 26 / 28 the log decays (fp32 remainder, fp32 UnpackToDestFp32, bf16 part,
// cumulative), 17 the committed state for the matmuls, 19 / 20 / 21 q, k, k^T (fp32), 23 / 24 / 27 4-tile
// scratch, 29 a FIFO of single tiles. Tiles are read at an index from a CB's read pointer, which the unpacker does
// not wrap: every CB that is read that way takes exactly its size per layer (29: 14 single tiles).
//
// Compile-time args: Nk, Nv, Kt, Vt, group, scale, l2_eps, rms_eps, inv_dv (floats as bits), GDN layers, R

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q_in = 0, cb_k_in = 1, cb_v_in = 2, cb_z_in = 3, cb_norm_w = 4, cb_s_base = 5, cb_gates = 6;
constexpr uint32_t cb_keep = 7, cb_ldlo = 8, cb_gamma_a = 11, cb_w = 12, cb_kt_prev = 13, cb_d_prev = 14;
constexpr uint32_t cb_consts = 15, cb_maskinc = 16, cb_s_mm = 17, cb_s_out = 18, cb_qn = 19, cb_kn = 20, cb_kt = 21;
constexpr uint32_t cb_x4a = 23, cb_x4b = 24, cb_ld = 25, cb_ldhi = 26, cb_x4c = 27, cb_c = 28, cb_tmp = 29;
constexpr uint32_t cb_out = 30, cb_pend = 31;
// constants
constexpr uint32_t kOnes = 0, kTri = 1, kTriT = 2, kEye = 3, kStrict = 4;
// gates
constexpr uint32_t kARow = 0, kACol = 1, kBRow = 2, kDt = 3, kNegA = 4;
// kept tiles
constexpr uint32_t kBeta = 0, kGamma = 1, kG = 2, kM = 3, kAinv = 4, kKept = 5;
// pending block
constexpr uint32_t kPendKt = 0, kPendDelta = 4, kPendG = 8, kPendGamma = 9, kPend = 10;

constexpr uint32_t kOneBits = 0x3f800000u;
constexpr uint32_t kSoftplusThreshold = 0x41a00000u;  // 20.0f

enum class Bin { Mul, Sub, Add, MulBcastRows };

// c += n tiles: a[ia + t] OP b[ib + (b_single ? 0 : t)], on the FPU. Does not pop.
__attribute__((noinline)) void ew(
    Bin op, uint32_t cb_a, uint32_t ia, uint32_t cb_b, uint32_t ib, bool b_single, uint32_t n, uint32_t cb_c) {
    CircularBuffer(cb_a).wait_front(ia + n);
    CircularBuffer(cb_b).wait_front(ib + (b_single ? 1 : n));
    CircularBuffer(cb_c).reserve_back(n);
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_c);
    if (op == Bin::Mul) {
        mul_init(cb_a, cb_b);
    } else if (op == Bin::Sub) {
        sub_init(cb_a, cb_b);
    } else if (op == Bin::Add) {
        add_init(cb_a, cb_b);
    } else {
        mul_bcast_rows_init(cb_a, cb_b);
    }
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t tb = ib + (b_single ? 0 : t);
        tile_regs_acquire();
        if (op == Bin::Mul) {
            mul_tiles(cb_a, cb_b, ia + t, tb, 0);
        } else if (op == Bin::Sub) {
            sub_tiles(cb_a, cb_b, ia + t, tb, 0);
        } else if (op == Bin::Add) {
            add_tiles(cb_a, cb_b, ia + t, tb, 0);
        } else {
            mul_tiles_bcast_rows(cb_a, cb_b, ia + t, tb, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c);
        tile_regs_release();
    }
    CircularBuffer(cb_c).push_back(n);
}

inline void matmul_setup(uint32_t in0, uint32_t in1, uint32_t out) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
    pack_reconfig_data_format(out);
}

// c += n_out tiles: out[j] = sum_k a[ia + k] @ b[ib + k * n_out + j]; also packed to pend[pend_at + j] when
// pend_at >= 0. Does not pop.
__attribute__((noinline)) void mm(
    uint32_t cb_a,
    uint32_t ia,
    uint32_t cb_b,
    uint32_t ib,
    uint32_t n_k,
    uint32_t n_out,
    uint32_t cb_c,
    int32_t pend_at = -1) {
    CircularBuffer(cb_a).wait_front(ia + n_k);
    CircularBuffer(cb_b).wait_front(ib + n_k * n_out);
    CircularBuffer(cb_c).reserve_back(n_out);
    matmul_setup(cb_a, cb_b, cb_c);
    for (uint32_t j = 0; j < n_out; j++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < n_k; k++) {
            matmul_tiles(cb_a, cb_b, ia + k, ib + k * n_out + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c);
        if (pend_at >= 0) {
            pack_tile<true>(0, cb_pend, pend_at + j);
        }
        tile_regs_release();
    }
    CircularBuffer(cb_c).push_back(n_out);
}

inline void copy_to_dst(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

// tmp += inv = post * rsqrt(pre * sum_cols(x^2) + eps) per row (replicated over the columns), x: n tiles of cb_x
// (squares in cb_sq). Does not pop x.
__attribute__((noinline)) void inv_norm(
    uint32_t cb_x, uint32_t n, uint32_t cb_sq, uint32_t pre, uint32_t eps, uint32_t post) {
    ew(Bin::Mul, cb_x, 0, cb_x, 0, false, n, cb_sq);
    CircularBuffer(cb_sq).wait_front(n);
    CircularBuffer(cb_tmp).reserve_back(1);
    matmul_setup(cb_sq, cb_consts, cb_tmp);
    tile_regs_acquire();
    for (uint32_t t = 0; t < n; t++) {
        matmul_tiles(cb_sq, cb_consts, t, kOnes, 0);
    }
    binop_with_scalar_tile_init();
    if (pre != kOneBits) {
        mul_unary_tile(0, pre);
    }
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if (post != kOneBits) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, post);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_tmp);
    tile_regs_release();
    CircularBuffer(cb_tmp).push_back(1);
    CircularBuffer(cb_sq).pop_front(n);
}

// out = in (x) inv_norm(in) per row; pops in
__attribute__((noinline)) void l2_normalize(uint32_t cb_in, uint32_t n, uint32_t eps, uint32_t post, uint32_t cb_c) {
    inv_norm(cb_in, n, cb_x4c, kOneBits, eps, post);
    ew(Bin::Mul, cb_in, 0, cb_tmp, 0, true, n, cb_c);
    CircularBuffer(cb_in).pop_front(n);
    CircularBuffer(cb_tmp).pop_front(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Vt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(5);
    constexpr uint32_t l2_eps_bits = get_compile_time_arg_val(6);
    constexpr uint32_t rms_eps_bits = get_compile_time_arg_val(7);
    constexpr uint32_t inv_dv_bits = get_compile_time_arg_val(8);
    constexpr uint32_t layers = get_compile_time_arg_val(9);
    constexpr uint32_t st = Kt * Vt;
    static_assert(Kt == 4 && Vt == 4, "128-wide heads");

    compute_kernel_hw_startup(cb_q_in, cb_consts, cb_tmp);
    CircularBuffer(cb_consts).wait_front(5);
    CircularBuffer(cb_maskinc).wait_front(1);
    for (uint32_t layer = 0; layer < layers; layer++) {
        CircularBuffer(cb_pend).reserve_back(kPend);
        // ---- 0. commit the previous step's accepted rows: S = Gamma_a S_base + (K^T (x) w) Delta
        ew(Bin::Mul, cb_kt_prev, 0, cb_w, 0, true, Kt, cb_x4a);
        CircularBuffer(cb_kt_prev).pop_front(Kt);
        CircularBuffer(cb_w).pop_front(1);
        {
            constexpr uint32_t kWin = 3, kScale = 3;
            CircularBuffer(cb_s_base).wait_front(st);
            CircularBuffer(cb_gamma_a).wait_front(1);
            CircularBuffer(cb_x4a).wait_front(Kt);
            CircularBuffer(cb_d_prev).wait_front(Vt);
            CircularBuffer(cb_s_mm).reserve_back(st);
            CircularBuffer(cb_s_out).reserve_back(st);
            for (uint32_t t0 = 0; t0 < st; t0 += kWin) {
                const uint32_t n = (st - t0) < kWin ? (st - t0) : kWin;
                tile_regs_acquire();
                reconfig_data_format_srca(cb_s_base);
                copy_init(cb_s_base);
                for (uint32_t i = 0; i < n; i++) {
                    copy_tile(cb_s_base, t0 + i, i);
                }
                copy_to_dst(cb_gamma_a, 0, kScale);
                mul_binary_tile_init();
                for (uint32_t i = 0; i < n; i++) {
                    mul_binary_tile(i, kScale, i);
                }
                matmul_setup(cb_x4a, cb_d_prev, cb_s_mm);
                for (uint32_t i = 0; i < n; i++) {
                    const uint32_t t = t0 + i;
                    matmul_tiles(cb_x4a, cb_d_prev, t / Vt, t % Vt, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < n; i++) {
                    pack_tile(i, cb_s_mm, t0 + i);
                    pack_tile(i, cb_s_out, t0 + i);
                }
                tile_regs_release();
            }
            CircularBuffer(cb_s_mm).push_back(st);
            CircularBuffer(cb_s_out).push_back(st);
            CircularBuffer(cb_s_base).pop_front(st);
            CircularBuffer(cb_gamma_a).pop_front(1);
            CircularBuffer(cb_x4a).pop_front(Kt);
            CircularBuffer(cb_d_prev).pop_front(Vt);
        }

        // ---- 1. normalized query (with scale) and key rows; K^T (also pending)
        l2_normalize(cb_q_in, Kt, l2_eps_bits, scale_bits, cb_qn);
        l2_normalize(cb_k_in, Kt, l2_eps_bits, kOneBits, cb_kn);
        CircularBuffer(cb_kn).wait_front(Kt);
        CircularBuffer(cb_kt).reserve_back(Kt);
        reconfig_data_format_srca(cb_kn);
        pack_reconfig_data_format(cb_kt);
        transpose_init(cb_kn);
        for (uint32_t t = 0; t < Kt; t++) {
            tile_regs_acquire();
            transpose_tile(cb_kn, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_kt);
            pack_tile<true>(0, cb_pend, kPendKt + t);
            tile_regs_release();
        }
        CircularBuffer(cb_kt).push_back(Kt);

        // ---- 2. gates: log decays ld = neg_exp_A * softplus(a + dt_bias) as rows and as columns (fp32 and its
        //         bf16 part), beta = sigmoid(b) rows
        CircularBuffer(cb_gates).wait_front(5);
        CircularBuffer(cb_ld).reserve_back(2);
        CircularBuffer(cb_ldhi).reserve_back(2);
        for (uint32_t side = 0; side < 2; side++) {
            tile_regs_acquire();
            copy_to_dst(cb_gates, side == 0 ? kARow : kACol, 0);
            copy_to_dst(cb_gates, kDt, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            softplus_tile_init();
            softplus_tile(0, kOneBits, kOneBits, kSoftplusThreshold);
            copy_to_dst(cb_gates, kNegA, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_ld);
            pack_tile(0, cb_ld);
            pack_reconfig_data_format(cb_ldhi);
            pack_tile(0, cb_ldhi);
            tile_regs_release();
        }
        CircularBuffer(cb_ld).push_back(2);
        CircularBuffer(cb_ldhi).push_back(2);
        CircularBuffer(cb_keep).reserve_back(1);
        tile_regs_acquire();
        copy_to_dst(cb_gates, kBRow, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_keep);
        pack_tile(0, cb_keep);
        tile_regs_release();
        CircularBuffer(cb_keep).push_back(1);  // kBeta
        CircularBuffer(cb_gates).pop_front(5);
        // fp32 remainders ld - bf16(ld)
        CircularBuffer(cb_ld).wait_front(2);
        CircularBuffer(cb_ldhi).wait_front(2);
        CircularBuffer(cb_ldlo).reserve_back(2);
        for (uint32_t side = 0; side < 2; side++) {
            tile_regs_acquire();
            copy_to_dst(cb_ld, side, 0);
            copy_to_dst(cb_ldhi, side, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_ldlo);
            pack_tile(0, cb_ldlo);
            tile_regs_release();
        }
        CircularBuffer(cb_ldlo).push_back(2);
        CircularBuffer(cb_ld).pop_front(2);
        // cumulative log decays: c rows = Tri @ ld rows, c columns = ld columns @ Tri^T (bf16 part + remainder)
        CircularBuffer(cb_ldlo).wait_front(2);
        CircularBuffer(cb_c).reserve_back(2);
        tile_regs_acquire();
        matmul_setup(cb_consts, cb_ldhi, cb_c);
        matmul_tiles(cb_consts, cb_ldhi, kTri, 0, 0);
        matmul_setup(cb_consts, cb_ldlo, cb_c);
        matmul_tiles(cb_consts, cb_ldlo, kTri, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c);
        tile_regs_release();
        tile_regs_acquire();
        matmul_setup(cb_ldhi, cb_consts, cb_c);
        matmul_tiles(cb_ldhi, cb_consts, 1, kTriT, 0);
        matmul_setup(cb_ldlo, cb_consts, cb_c);
        matmul_tiles(cb_ldlo, cb_consts, 1, kTriT, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c);
        tile_regs_release();
        CircularBuffer(cb_c).push_back(2);
        CircularBuffer(cb_ldhi).pop_front(2);
        CircularBuffer(cb_ldlo).pop_front(2);
        // Gamma = exp(c rows); G = exp(c_r - c_j + causal mask) (fp32 SFPU); both also pending
        CircularBuffer(cb_c).wait_front(2);
        CircularBuffer(cb_keep).reserve_back(2);
        tile_regs_acquire();
        copy_to_dst(cb_c, 0, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_keep);
        pack_tile(0, cb_keep);  // kGamma
        pack_tile<true>(0, cb_pend, kPendGamma);
        tile_regs_release();
        tile_regs_acquire();
        copy_to_dst(cb_c, 0, 0);
        copy_to_dst(cb_c, 1, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        copy_to_dst(cb_maskinc, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_keep);  // kG
        pack_tile<true>(0, cb_pend, kPendG);
        tile_regs_release();
        CircularBuffer(cb_keep).push_back(2);
        CircularBuffer(cb_c).pop_front(2);

        // ---- 3. K K^T, Q K^T; N = beta (x) strict (x) G (x) K K^T, M = G (x) Q K^T
        mm(cb_kn, 0, cb_kt, 0, Kt, 1, cb_tmp);  // tmp: [KK]
        // (Q K^T: q tile k with K^T tile k)
        {
            CircularBuffer(cb_qn).wait_front(Kt);
            CircularBuffer(cb_kt).wait_front(Kt);
            CircularBuffer(cb_tmp).reserve_back(1);
            matmul_setup(cb_qn, cb_kt, cb_tmp);
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_tiles(cb_qn, cb_kt, k, k, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            CircularBuffer(cb_tmp).push_back(1);  // tmp: [KK, QK]
        }
        ew(Bin::Mul, cb_tmp, 0, cb_keep, kG, true, 1, cb_tmp);         // [KK, QK, G KK]
        ew(Bin::Mul, cb_tmp, 2, cb_consts, kStrict, true, 1, cb_tmp);  // [.., strict G KK]
        ew(Bin::Mul, cb_tmp, 3, cb_keep, kBeta, true, 1, cb_tmp);      // [.., N]
        ew(Bin::Mul, cb_tmp, 1, cb_keep, kG, true, 1, cb_keep);        // kM
        CircularBuffer(cb_tmp).pop_front(4);                           // [N]

        // ---- 4. (I + N)^-1 = (I - N)(I + N^2)(I + N^4)
        mm(cb_tmp, 0, cb_tmp, 0, 1, 1, cb_tmp);                     // [N, N2]
        mm(cb_tmp, 1, cb_tmp, 1, 1, 1, cb_tmp);                     // [N, N2, N4]
        ew(Bin::Sub, cb_consts, kEye, cb_tmp, 0, true, 1, cb_tmp);  // [N, N2, N4, I-N]
        ew(Bin::Add, cb_consts, kEye, cb_tmp, 1, true, 1, cb_tmp);  // [.., I+N2]
        CircularBuffer(cb_tmp).pop_front(2);                        // [N4, I-N, I+N2]
        mm(cb_tmp, 1, cb_tmp, 2, 1, 1, cb_tmp);                     // [N4, I-N, I+N2, T]
        ew(Bin::Add, cb_consts, kEye, cb_tmp, 0, true, 1, cb_tmp);  // [.., T, I+N4]
        mm(cb_tmp, 3, cb_tmp, 4, 1, 1, cb_keep);                    // kAinv
        CircularBuffer(cb_tmp).pop_front(5);

        // ---- 5. Delta = (I + N)^-1 (beta (x) (V - (Gamma (x) K) S)) (also pending)
        ew(Bin::Mul, cb_kn, 0, cb_keep, kGamma, true, Kt, cb_x4a);
        mm(cb_x4a, 0, cb_s_mm, 0, Kt, Vt, cb_x4b);
        CircularBuffer(cb_x4a).pop_front(Kt);
        CircularBuffer(cb_kn).pop_front(Kt);
        CircularBuffer(cb_kt).pop_front(Kt);
        ew(Bin::Sub, cb_v_in, 0, cb_x4b, 0, false, Vt, cb_x4a);
        CircularBuffer(cb_v_in).pop_front(Vt);
        CircularBuffer(cb_x4b).pop_front(Vt);
        ew(Bin::Mul, cb_x4a, 0, cb_keep, kBeta, true, Vt, cb_x4b);
        CircularBuffer(cb_x4a).pop_front(Vt);
        // Delta tile v = Ainv @ B tile v
        {
            CircularBuffer(cb_keep).wait_front(kKept);
            CircularBuffer(cb_x4b).wait_front(Vt);
            CircularBuffer(cb_x4a).reserve_back(Vt);
            matmul_setup(cb_keep, cb_x4b, cb_x4a);
            for (uint32_t v = 0; v < Vt; v++) {
                tile_regs_acquire();
                matmul_tiles(cb_keep, cb_x4b, kAinv, v, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_x4a);
                pack_tile<true>(0, cb_pend, kPendDelta + v);
                tile_regs_release();
            }
            CircularBuffer(cb_x4a).push_back(Vt);
            CircularBuffer(cb_x4b).pop_front(Vt);
        }

        // ---- 6. O = (Gamma (x) Q) S + M Delta
        ew(Bin::Mul, cb_qn, 0, cb_keep, kGamma, true, Kt, cb_x4b);
        CircularBuffer(cb_qn).pop_front(Kt);
        {
            CircularBuffer(cb_x4b).wait_front(Kt);
            CircularBuffer(cb_s_mm).wait_front(st);
            CircularBuffer(cb_x4a).wait_front(Vt);
            CircularBuffer(cb_x4c).reserve_back(Vt);
            for (uint32_t v = 0; v < Vt; v++) {
                tile_regs_acquire();
                matmul_setup(cb_x4b, cb_s_mm, cb_x4c);
                for (uint32_t k = 0; k < Kt; k++) {
                    matmul_tiles(cb_x4b, cb_s_mm, k, k * Vt + v, 0);
                }
                matmul_setup(cb_keep, cb_x4a, cb_x4c);
                matmul_tiles(cb_keep, cb_x4a, kM, v, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_x4c);
                tile_regs_release();
            }
            CircularBuffer(cb_x4c).push_back(Vt);
            CircularBuffer(cb_x4b).pop_front(Kt);
            CircularBuffer(cb_s_mm).pop_front(st);
            CircularBuffer(cb_x4a).pop_front(Vt);
        }
        CircularBuffer(cb_keep).pop_front(kKept);
        CircularBuffer(cb_pend).push_back(kPend);

        // ---- 7. gated RMSNorm: out = o * rsqrt(mean(o^2) + eps) * norm_w * silu(z)
        inv_norm(cb_x4c, Vt, cb_x4a, inv_dv_bits, rms_eps_bits, kOneBits);
        ew(Bin::Mul, cb_x4c, 0, cb_tmp, 0, true, Vt, cb_x4b);
        CircularBuffer(cb_x4c).pop_front(Vt);
        CircularBuffer(cb_tmp).pop_front(1);
        ew(Bin::MulBcastRows, cb_x4b, 0, cb_norm_w, 0, false, Vt, cb_x4a);
        CircularBuffer(cb_x4b).pop_front(Vt);
        CircularBuffer(cb_norm_w).pop_front(Vt);
        CircularBuffer(cb_z_in).wait_front(Vt);
        CircularBuffer(cb_x4b).reserve_back(Vt);
        pack_reconfig_data_format(cb_x4b);
        for (uint32_t t = 0; t < Vt; t++) {
            tile_regs_acquire();
            copy_to_dst(cb_z_in, t, 0);
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_x4b);
            tile_regs_release();
        }
        CircularBuffer(cb_x4b).push_back(Vt);
        CircularBuffer(cb_z_in).pop_front(Vt);
        ew(Bin::Mul, cb_x4a, 0, cb_x4b, 0, false, Vt, cb_out);
        CircularBuffer(cb_x4a).pop_front(Vt);
        CircularBuffer(cb_x4b).pop_front(Vt);
    }
    CircularBuffer(cb_consts).pop_front(5);
    CircularBuffer(cb_maskinc).pop_front(1);
}
