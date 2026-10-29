// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core compute: the Gated DeltaNet decode step of one value head, repeated for every
// layer and user of the run (T=1; row 0 of every tile is live, the users one after another). The reader lands the
// head's q/k/v/z/a/b from the streamers and the layer's state; the writer keeps the updated state and returns the
// output.
//
//   q, k  = l2norm(q) * scale, l2norm(k)                   (key head kh = h / group)
//   decay = exp(neg_exp_A * softplus(a + dt_bias)),  beta = sigmoid(b)
//   delta = beta * (v - (decay * k) @ S)                    (decay is a per-head scalar)
//   S'    = decay * S + k^T (x) delta                        (written back in place)
//   o     = q @ S'
//   out   = rmsnorm(o) * norm_w * silu(z)
//
// State decay/update and the gates run on SFPU in fp32 (UnpackToDestFp32 CBs); matmuls and the
// row-sum reductions run on the FPU with fp32 accumulation, matching the unfused ttnn graph.
//
// Compile-time args: Nk, Nv, Kt, Vt, group, scale, l2_eps, rms_eps, inv_dv (floats as bits), layers x users

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

constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_z_in = 3;
constexpr uint32_t cb_norm_w = 4;
constexpr uint32_t cb_s_in = 5;
constexpr uint32_t cb_a_full = 6;
constexpr uint32_t cb_b_full = 7;
constexpr uint32_t cb_ones = 8;
constexpr uint32_t cb_row_mask = 9;
constexpr uint32_t cb_scratch = 10;
constexpr uint32_t cb_sq = 11;
constexpr uint32_t cb_inv = 12;
constexpr uint32_t cb_qn = 13;
constexpr uint32_t cb_kn = 14;
constexpr uint32_t cb_decay = 15;
constexpr uint32_t cb_beta = 16;
constexpr uint32_t cb_s_mm = 17;
constexpr uint32_t cb_vr = 18;
constexpr uint32_t cb_beta_row0 = 19;
constexpr uint32_t cb_delta = 20;
constexpr uint32_t cb_kt = 21;
constexpr uint32_t cb_s_new_mm = 23;
constexpr uint32_t cb_s_new_out = 24;
constexpr uint32_t cb_o = 25;
constexpr uint32_t cb_on = 26;
constexpr uint32_t cb_sz = 27;
constexpr uint32_t cb_out = 28;
constexpr uint32_t cb_neg_a_full = 30;
constexpr uint32_t cb_dt_bias_full = 31;

constexpr uint32_t kOneBits = 0x3f800000u;

enum class Bin { Mul, Sub, MulBcastRows };

// out[t] = a[t] OP b[b_single ? 0 : t] on the FPU, t in [0, n). Does not pop inputs.
__attribute__((noinline)) void fpu_binary(
    Bin op, uint32_t cb_a, uint32_t cb_b, bool b_single, uint32_t n, uint32_t cb_c) {
    CircularBuffer(cb_a).wait_front(n);
    CircularBuffer(cb_b).wait_front(b_single ? 1 : n);
    CircularBuffer(cb_c).reserve_back(n);
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_c);
    if (op == Bin::Mul) {
        mul_init(cb_a, cb_b);
    } else if (op == Bin::Sub) {
        sub_init(cb_a, cb_b);
    } else {
        mul_bcast_rows_init(cb_a, cb_b);
    }
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t tb = b_single ? 0 : t;
        tile_regs_acquire();
        if (op == Bin::Mul) {
            mul_tiles(cb_a, cb_b, t, tb, 0);
        } else if (op == Bin::Sub) {
            sub_tiles(cb_a, cb_b, t, tb, 0);
        } else {
            mul_tiles_bcast_rows(cb_a, cb_b, t, tb, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c, t);
        tile_regs_release();
    }
    CircularBuffer(cb_c).push_back(n);
}

inline void matmul_setup(uint32_t in0, uint32_t in1, uint32_t out) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
    pack_reconfig_data_format(out);
}

// out[j] = sum_k a[k] @ b[k * n_out + j] for a row-tile vector a (n_k tiles) and a
// [n_k x n_out] tile matrix b. Does not pop inputs.
__attribute__((noinline)) void row_matmul(uint32_t cb_a, uint32_t cb_b, uint32_t n_k, uint32_t n_out, uint32_t cb_c) {
    CircularBuffer(cb_a).wait_front(n_k);
    CircularBuffer(cb_b).wait_front(n_k * n_out);
    CircularBuffer(cb_c).reserve_back(n_out);
    matmul_setup(cb_a, cb_b, cb_c);
    for (uint32_t j = 0; j < n_out; j++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < n_k; k++) {
            matmul_tiles(cb_a, cb_b, k, k * n_out + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c, j);
        tile_regs_release();
    }
    CircularBuffer(cb_c).push_back(n_out);
}

// inv = post * rsqrt(pre * sum_cols(x^2) + eps), replicated across all 32 columns of one tile
// (row sum via x^2 @ ones). Does not pop x.
__attribute__((noinline)) void inv_norm(uint32_t cb_x, uint32_t n, uint32_t pre, uint32_t eps, uint32_t post) {
    fpu_binary(Bin::Mul, cb_x, cb_x, false, n, cb_sq);
    CircularBuffer(cb_sq).wait_front(n);
    CircularBuffer(cb_inv).reserve_back(1);
    matmul_setup(cb_sq, cb_ones, cb_inv);
    tile_regs_acquire();
    for (uint32_t t = 0; t < n; t++) {
        matmul_tiles(cb_sq, cb_ones, t, 0, 0);
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
    pack_tile(0, cb_inv);
    tile_regs_release();
    CircularBuffer(cb_inv).push_back(1);
    CircularBuffer(cb_sq).pop_front(n);
}

// L2-normalised copy of an input row vector: out = in * inv_norm(in). Padding rows are not masked
// here; they are removed from the rank-1 state update by the row-masked beta.
__attribute__((noinline)) void l2_normalize(uint32_t cb_in, uint32_t n, uint32_t eps, uint32_t post, uint32_t cb_c) {
    inv_norm(cb_in, n, kOneBits, eps, post);
    fpu_binary(Bin::Mul, cb_in, cb_inv, true, n, cb_c);
    CircularBuffer(cb_in).pop_front(n);
    CircularBuffer(cb_inv).pop_front(1);
}

inline void copy_to_dst(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Vt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(5);
    constexpr uint32_t l2_eps_bits = get_compile_time_arg_val(6);
    constexpr uint32_t rms_eps_bits = get_compile_time_arg_val(7);
    constexpr uint32_t inv_dv_bits = get_compile_time_arg_val(8);
    constexpr uint32_t st = Kt * Vt;
    constexpr uint32_t kSoftplusThreshold = 0x41a00000u;  // 20.0f

    constexpr uint32_t layers = get_compile_time_arg_val(9);
    compute_kernel_hw_startup(cb_q_in, cb_row_mask, cb_scratch);
    CircularBuffer(cb_ones).wait_front(1);
    CircularBuffer(cb_row_mask).wait_front(1);
    for (uint32_t layer = 0; layer < layers; layer++) {
        // 1. Normalised query (with scale) and key.
        l2_normalize(cb_q_in, Kt, l2_eps_bits, scale_bits, cb_qn);
        l2_normalize(cb_k_in, Kt, l2_eps_bits, kOneBits, cb_kn);
        // 2. Gates (fp32 SFPU): decay = exp(neg_exp_A * softplus(a + dt_bias)), beta = sigmoid(b).
        CircularBuffer(cb_a_full).wait_front(1);
        CircularBuffer(cb_dt_bias_full).wait_front(1);
        CircularBuffer(cb_neg_a_full).wait_front(1);
        CircularBuffer(cb_decay).reserve_back(1);
        CircularBuffer(cb_inv).reserve_back(1);
        pack_reconfig_data_format(cb_decay);
        tile_regs_acquire();
        copy_to_dst(cb_a_full, 0, 0);
        copy_to_dst(cb_dt_bias_full, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        softplus_tile_init();
        softplus_tile(0, kOneBits, kOneBits, kSoftplusThreshold);
        copy_to_dst(cb_neg_a_full, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_decay);
        pack_tile(0, cb_inv);  // FPU-side copy of decay
        tile_regs_release();
        CircularBuffer(cb_decay).push_back(1);
        CircularBuffer(cb_inv).push_back(1);
        CircularBuffer(cb_a_full).pop_front(1);
        CircularBuffer(cb_dt_bias_full).pop_front(1);
        CircularBuffer(cb_neg_a_full).pop_front(1);

        CircularBuffer(cb_b_full).wait_front(1);
        CircularBuffer(cb_beta).reserve_back(1);
        pack_reconfig_data_format(cb_beta);
        tile_regs_acquire();
        copy_to_dst(cb_b_full, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_beta);
        tile_regs_release();
        CircularBuffer(cb_beta).push_back(1);
        CircularBuffer(cb_b_full).pop_front(1);
        // beta restricted to row 0: zeroes the padding rows of delta, so they drop out of k^T (x) delta.
        fpu_binary(Bin::Mul, cb_beta, cb_row_mask, true, 1, cb_beta_row0);
        CircularBuffer(cb_beta).pop_front(1);
        // 3. delta = beta_row0 * (v - (decay * k) @ S).
        fpu_binary(Bin::Mul, cb_kn, cb_inv, true, Kt, cb_scratch);
        CircularBuffer(cb_inv).pop_front(1);
        row_matmul(cb_scratch, cb_s_mm, Kt, Vt, cb_vr);
        CircularBuffer(cb_scratch).pop_front(Kt);
        CircularBuffer(cb_s_mm).pop_front(st);
        fpu_binary(Bin::Sub, cb_v_in, cb_vr, false, Vt, cb_scratch);
        CircularBuffer(cb_v_in).pop_front(Vt);
        CircularBuffer(cb_vr).pop_front(Vt);
        fpu_binary(Bin::Mul, cb_scratch, cb_beta_row0, true, Vt, cb_delta);
        CircularBuffer(cb_scratch).pop_front(Vt);
        CircularBuffer(cb_beta_row0).pop_front(1);
        // 4. k^T tiles (column vectors) for the rank-1 update.
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
            pack_tile(0, cb_kt, t);
            tile_regs_release();
        }
        CircularBuffer(cb_kt).push_back(Kt);
        CircularBuffer(cb_kn).pop_front(Kt);
        // 5. S' = decay * S + k^T (x) delta. Per DEST window: three state tiles and decay are copied in
        //    (fp32), scaled on SFPU, then the rank-1 tile k^T[kt] @ delta[vt] is accumulated onto them
        //    by the FPU (only column 0 of k^T and row 0 of delta are live). Packed twice: for the
        //    query matmul and for the writer.
        constexpr uint32_t kWin = 3;
        constexpr uint32_t kDecaySlot = 3;
        CircularBuffer(cb_kt).wait_front(Kt);
        CircularBuffer(cb_delta).wait_front(Vt);
        CircularBuffer(cb_s_in).wait_front(st);
        CircularBuffer(cb_decay).wait_front(1);
        CircularBuffer(cb_s_new_mm).reserve_back(st);
        CircularBuffer(cb_s_new_out).reserve_back(st);
        for (uint32_t t0 = 0; t0 < st; t0 += kWin) {
            const uint32_t n = (st - t0) < kWin ? (st - t0) : kWin;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_s_in);
            copy_init(cb_s_in);
            for (uint32_t i = 0; i < n; i++) {
                copy_tile(cb_s_in, t0 + i, i);
            }
            copy_to_dst(cb_decay, 0, kDecaySlot);
            mul_binary_tile_init();
            for (uint32_t i = 0; i < n; i++) {
                mul_binary_tile(i, kDecaySlot, i);
            }
            matmul_setup(cb_kt, cb_delta, cb_s_new_mm);
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t t = t0 + i;
                matmul_tiles(cb_kt, cb_delta, t / Vt, t % Vt, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; i++) {
                pack_tile(i, cb_s_new_mm, t0 + i);
                pack_tile(i, cb_s_new_out, t0 + i);
            }
            tile_regs_release();
        }
        CircularBuffer(cb_s_new_mm).push_back(st);
        CircularBuffer(cb_s_new_out).push_back(st);
        CircularBuffer(cb_s_in).pop_front(st);
        CircularBuffer(cb_decay).pop_front(1);
        CircularBuffer(cb_kt).pop_front(Kt);
        CircularBuffer(cb_delta).pop_front(Vt);
        // 6. o = q @ S'.
        row_matmul(cb_qn, cb_s_new_mm, Kt, Vt, cb_o);
        CircularBuffer(cb_qn).pop_front(Kt);
        CircularBuffer(cb_s_new_mm).pop_front(st);
        // 7. Gated RMSNorm: on = o * rsqrt(mean(o^2) + eps) * norm_w.
        inv_norm(cb_o, Vt, inv_dv_bits, rms_eps_bits, kOneBits);
        fpu_binary(Bin::Mul, cb_o, cb_inv, true, Vt, cb_on);
        CircularBuffer(cb_o).pop_front(Vt);
        CircularBuffer(cb_inv).pop_front(1);
        fpu_binary(Bin::MulBcastRows, cb_on, cb_norm_w, false, Vt, cb_scratch);
        CircularBuffer(cb_on).pop_front(Vt);
        CircularBuffer(cb_norm_w).pop_front(Vt);
        // 8. out = on * silu(z).
        CircularBuffer(cb_z_in).wait_front(Vt);
        CircularBuffer(cb_sz).reserve_back(Vt);
        pack_reconfig_data_format(cb_sz);
        for (uint32_t t = 0; t < Vt; t++) {
            tile_regs_acquire();
            copy_to_dst(cb_z_in, t, 0);
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sz, t);
            tile_regs_release();
        }
        CircularBuffer(cb_sz).push_back(Vt);
        CircularBuffer(cb_z_in).pop_front(Vt);
        fpu_binary(Bin::Mul, cb_scratch, cb_sz, false, Vt, cb_out);
        CircularBuffer(cb_scratch).pop_front(Vt);
        CircularBuffer(cb_sz).pop_front(Vt);
    }
    CircularBuffer(cb_ones).pop_front(1);
    CircularBuffer(cb_row_mask).pop_front(1);
}
