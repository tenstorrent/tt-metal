// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The gradient arithmetic of one block pair of the cyclic SDPA backward pass.
//
// Given a resident column block (K_j, V_j) and a row packet
// (Q_i, dO_i, L_i, D_i), this computes
//
//   S  = Q_i K_j^T / sqrt(d)      (masked when i == j)
//   P  = exp(S - L_i)
//   dP = dO_i V_j^T
//   dS = P * (dP - D_i) / sqrt(d)
//   dQ_i += dS K_j        dV_j += P^T dO_i        dK_j += dS^T Q_i
//
// Five matmuls per pair. tt-train's sdpa_bw computes dQ in a separate kernel
// that recomputes S, P and dP, which is seven; fusing them is the point of
// this schedule, and the fusion is what is new here. Everything else is
// sdpa_bw's helpers, which already produce dS^T and P^T in DST alongside dS.
//
// COMPUTE_STAGE builds the chain up one step at a time, each stage writing
// its last intermediate to a probe tile so a numerical failure names the step
// that produced it rather than just the gradients:
//
//   1 S    2 P    3 dP    4 dS    5 the three gradients
//
// There is deliberately no stage that reads dS^T or P^T back. Packing a
// fourth tile out of the transpose region is fragile -- it returned zeros and
// in one arrangement hung -- and those two operands have exactly one consumer
// each, dK and dV, which check them well enough.
//
// Single block pair, so B = 32 (one tile row) and the accumulators start from
// this pair rather than accumulating across timesteps. Accumulation across a
// streak and a column residency is the next step.

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp"

#ifndef COMPUTE_STAGE
#define COMPUTE_STAGE 5
#endif

namespace {

constexpr uint32_t qWt = get_compile_time_arg_val(0);          // Q/K width in tiles
constexpr uint32_t vWt = get_compile_time_arg_val(1);          // V/dO width in tiles
constexpr uint32_t scaler_bits = get_compile_time_arg_val(2);  // 1/sqrt(d), float32 bits
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(3);
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(4);
constexpr uint32_t block_size = get_compile_time_arg_val(5);

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
constexpr uint32_t cb_lse = tt::CBIndex::c_4;       // L_i, one FP32 tile, value in column 0
constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;  // D_i, likewise
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;

constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;      // P
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_11;      // dP
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_12;            // dS
constexpr uint32_t cb_grad_scores_transposed = tt::CBIndex::c_13;  // dS^T
constexpr uint32_t cb_attn_weights_transposed = tt::CBIndex::c_14;  // P^T

constexpr uint32_t cb_probe = tt::CBIndex::c_15;       // the stage's intermediate
constexpr uint32_t cb_grad_query = tt::CBIndex::c_16;  // dQ_i
constexpr uint32_t cb_grad_key = tt::CBIndex::c_17;    // dK_j
constexpr uint32_t cb_grad_value = tt::CBIndex::c_18;  // dV_j

// Copy one tile from a CB to the probe output, leaving the source in place.
//
// Only valid before the transposes. transpose_dest_init reprograms the math
// MOP and addr-mods, and copy_init does not restore enough of that state: a
// copy_tile issued after a transpose_dest packs zeros, and worse, it leaves
// the matmuls that follow it disturbed -- which is how a probe placed between
// the transposes and the gradient matmuls corrupted dK while dQ and dV came
// out right. Anything wanted after the transposes is packed from inside their
// own acquire instead. The same hazard is documented for mm_init_short in
// sdpa_bw's optimisation notes.
void probe_tile(uint32_t cb_source) {
    cb_wait_front(cb_source, onetile);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_source);
    copy_init(cb_source);
    copy_tile(cb_source, /* tile_idx */ 0, /* register idx */ 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_probe, onetile);
    pack_reconfig_data_format(cb_probe);
    pack_tile(0, cb_probe);
    tile_regs_release();
    cb_push_back(cb_probe, onetile);
}

// dS, and the two transposed operands the column gradients need, out of one
// DST region. sdpa_bw's version packs only dS^T and P^T, because its dQ lives
// in another kernel; the fused schedule needs dS itself as well, and it is
// already sitting in DST at that point.
// dS goes to the probe from the same acquire region that holds it.
//
// dS, dS^T and P^T all come out of one acquire, and they have to. dS must not
// be written to a CB and read back to be transposed: the transposed CBs are
// Float32 but unpack in Default mode, because matmul Src registers do not
// support Float32 unpack, so a copy_tile out of one lands in DST in a layout
// the 32-bit transpose_dest then scrambles. That is what made dK wrong while
// dQ and dV looked fine. So dS is duplicated inside DST with
// copy_dest_values, one copy is transposed and one is not, and all three
// operands are packed from registers.
void compute_grad_scores_with_transposes() {
    cb_wait_front(cb_grad_attn_weights, onetile);
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_u_scalar, onetile);

    constexpr uint32_t grad_reg = 0;      // dS, then dS^T
    constexpr uint32_t attn_reg = 1;      // P, then P^T
    constexpr uint32_t grad_keep_reg = 2;  // dS, left untransposed for dQ

    tile_regs_acquire();
    reconfig_data_format(cb_grad_attn_weights, cb_u_scalar);
    sub_bcast_cols_init(cb_grad_attn_weights, cb_u_scalar);
    sub_tiles_bcast_cols(cb_grad_attn_weights, cb_u_scalar, 0, 0, grad_reg);

    // P is copied to DST, which is why cb_attention_weights is the one CB
    // declared UnpackToDestFp32: it keeps the elementwise chain at full FP32.
    reconfig_data_format_srca(cb_grad_attn_weights, cb_attention_weights);
    copy_init(cb_attention_weights);
    copy_tile(cb_attention_weights, 0, attn_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_reg, grad_reg);

    binop_with_scalar_tile_init();
    mul_unary_tile(grad_reg, scaler_bits);

    // Keep dS as it is for dQ before transposing the other copy.
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(grad_reg, grad_keep_reg);

    transpose_dest_init</* is_32bit */ true>(cb_attention_weights);
    transpose_dest</* is_32bit */ true>(grad_reg);  // dS -> dS^T
    transpose_dest</* is_32bit */ true>(attn_reg);  // P  -> P^T

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_grad_scores, onetile);
    cb_reserve_back(cb_grad_scores_transposed, onetile);
    cb_reserve_back(cb_attn_weights_transposed, onetile);
    pack_reconfig_data_format(cb_grad_attn_weights, cb_grad_scores);
    pack_tile(grad_keep_reg, cb_grad_scores);
    pack_reconfig_data_format(cb_grad_scores, cb_grad_scores_transposed);
    pack_tile(grad_reg, cb_grad_scores_transposed);
    pack_reconfig_data_format(cb_grad_scores_transposed, cb_attn_weights_transposed);
    pack_tile(attn_reg, cb_attn_weights_transposed);
    cb_reserve_back(cb_probe, onetile);
    pack_reconfig_data_format(cb_probe);
    pack_tile(grad_keep_reg, cb_probe);
    cb_push_back(cb_probe, onetile);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);
    cb_push_back(cb_grad_scores_transposed, onetile);
    cb_push_back(cb_attn_weights_transposed, onetile);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_query, cb_key, cb_attention_weights);
    copy_init(cb_query);
    matmul_init(cb_query, cb_key);

    cb_wait_front(cb_query, qWt);
    cb_wait_front(cb_key, qWt);
    cb_wait_front(cb_value, vWt);
    cb_wait_front(cb_grad_output, vWt);
    cb_wait_front(cb_lse, onetile);

    // ---- S = Q K^T / sqrt(d), masked on the diagonal
    constexpr uint32_t scores_reg = 0;
    reconfig_data_format(cb_query, cb_key);
    matmul_init(cb_query, cb_key, /* transpose */ 1);
    tile_regs_acquire();
    for (uint32_t k = 0; k < qWt; ++k) {
        matmul_tiles(cb_query, cb_key, k, k, scores_reg);
    }
#ifdef DIAGONAL_BLOCK
    apply_mask_on_reg(scores_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
#else
    binop_with_scalar_tile_init();
    mul_unary_tile(scores_reg, scaler_bits);
#endif

#if COMPUTE_STAGE == 1
    // Stop at the scores.
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_probe, onetile);
    pack_reconfig_data_format(cb_probe);
    pack_tile(scores_reg, cb_probe);
    tile_regs_release();
    cb_push_back(cb_probe, onetile);
    return;
#else
    // ---- P = exp(S - L_i), fused on DST at full FP32
    apply_softmax_statistics_on_dst(scores_reg, cb_lse);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_attention_weights, onetile);
    pack_reconfig_data_format(cb_attention_weights);
    pack_tile(scores_reg, cb_attention_weights);
    tile_regs_release();
    cb_push_back(cb_attention_weights, onetile);

#if COMPUTE_STAGE == 2
    probe_tile(cb_attention_weights);
    return;
#else
    // ---- dP = dO V^T
    compute_grad_attn_weights(
        cb_grad_output, cb_value, vWt, cb_grad_attn_weights, cb_attention_weights, scaler_bits);

#if COMPUTE_STAGE == 3
    probe_tile(cb_grad_attn_weights);
    return;
#else
    // ---- dS = P * (dP - D_i) / sqrt(d), with dS^T and P^T alongside
    compute_grad_scores_with_transposes();

#if COMPUTE_STAGE == 4
    return;
#else
    // ---- the three gradients
    update_grad_query(cb_grad_scores, cb_key, cb_grad_query, qWt, block_size, /* accumulate */ false);
    cb_wait_front(cb_grad_query, qWt);

    update_grad_value(
        cb_attn_weights_transposed, cb_grad_output, cb_grad_value, vWt, block_size, false);
    cb_wait_front(cb_grad_value, vWt);

    update_grad_key(
        cb_grad_scores_transposed,
        cb_query,
        cb_grad_key,
        qWt,
        block_size,
        /* cb_prev_pack */ cb_grad_value,
        /* cb_prev_srca */ cb_grad_output,
        false);
    cb_wait_front(cb_grad_key, qWt);
#endif
#endif
#endif
#endif
}
