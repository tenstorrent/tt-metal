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
// B = 32, one tile row per block. NUM_PAIRS > 1 runs several pairs in
// sequence with L1 accumulation into the same gradient buffers, which is what
// a streak does for dQ and a column residency does for dK and dV: the first
// pair reserves, the rest accumulate in place.

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
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp"

#ifndef COMPUTE_STAGE
#define COMPUTE_STAGE 5
#endif

#ifndef NUM_PAIRS
#define NUM_PAIRS 1
#endif

// SEED_DQ: the accumulator starts from a previous dQ rather than from zero,
// which is what Algorithm 2 needs at every timestep and the relay needs at
// every streak start. The reader puts that value in its own buffer and this
// kernel copies it into the accumulator with pack_tiles_to_output.
//
// The copy is not a formality. sdpa_bw's accumulate path packs without
// reserving, which relies on the packer's write pointer already being where
// the previous cycle left it -- and only a reserve/push cycle *by this kernel*
// does that. A push from the reader does not: the write pointers are per
// RISC. Packing straight onto reader-pushed tiles leaves the packer stalled
// and the result zero.
#ifndef SEED_DQ
#define SEED_DQ 0
#endif

namespace {

constexpr uint32_t qWt = get_compile_time_arg_val(0);          // Q/K width in tiles
constexpr uint32_t vWt = get_compile_time_arg_val(1);          // V/dO width in tiles
constexpr uint32_t scaler_bits = get_compile_time_arg_val(2);  // 1/sqrt(d), float32 bits
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(3);
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(4);
constexpr uint32_t block_size = get_compile_time_arg_val(5);
// Row-tiles per block: B = Bt * 32. Bt > 1 is what lets the score matmuls of
// one row issue back to back, so the matmul pipeline latency is paid once per
// row instead of once per tile. Profiling put that latency at ~1.4 us against
// 0.076 us for each additional tile issued behind it.
constexpr uint32_t Bt = get_compile_time_arg_val(6);
constexpr uint32_t score_tiles = Bt * Bt;

// Score tiles live in even registers because both apply_mask_on_reg and the
// softmax step need a scratch register next to the one they work on, and
// several score tiles are live at once. Bt <= 4 fits the eight FP32 registers.
constexpr uint32_t score_reg(uint32_t b) {
    return 2u * b;
}

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
constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_19;  // previous dQ_i

// The same fused FP32 softmax as sdpa_bw's apply_softmax_statistics_on_dst,
// but reading L from a chosen tile and taking the scratch register as an
// argument: that helper uses scores_reg + 1, which is another score tile
// here. At Bt = 1 with tmp = scores_reg + 1 this is the same sequence.
void softmax_on_dst(
    const uint32_t scores_reg,
    const uint32_t tmp_reg,
    const uint32_t cb_statistics,
    const uint32_t stat_tile) {
    reconfig_data_format_srcb(cb_statistics);
    UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_statistics)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          ckernel::DataCopyType::B2D,
          DST_ACCUM_MODE,
          BroadcastType::COL>(cb_statistics)));
    unary_bcast<BroadcastType::COL>(cb_statistics, stat_tile, tmp_reg);

    sub_binary_tile_init();
    sub_binary_tile(scores_reg, tmp_reg, scores_reg);

    sdpa_exp_tile_init();
    sdpa_exp_tile(scores_reg);
}

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
void compute_grad_scores_with_transposes(bool want_probe, uint32_t a = 0, uint32_t b = 0) {
    // The score tile of this (row, column) tile pair, and the row's D.
    const uint32_t score_tile = a * Bt + b;
    const uint32_t transposed_tile = b * Bt + a;

    constexpr uint32_t grad_reg = 0;      // dS, then dS^T
    constexpr uint32_t attn_reg = 1;      // P, then P^T
    constexpr uint32_t grad_keep_reg = 2;  // dS, left untransposed for dQ

    tile_regs_acquire();
    reconfig_data_format(cb_grad_attn_weights, cb_u_scalar);
    sub_bcast_cols_init(cb_grad_attn_weights, cb_u_scalar);
    sub_tiles_bcast_cols(cb_grad_attn_weights, cb_u_scalar, score_tile, a, grad_reg);

    // P is copied to DST, which is why cb_attention_weights is the one CB
    // declared UnpackToDestFp32: it keeps the elementwise chain at full FP32.
    reconfig_data_format_srca(cb_grad_attn_weights, cb_attention_weights);
    copy_init(cb_attention_weights);
    copy_tile(cb_attention_weights, score_tile, attn_reg);

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
    // Out-of-order packing, because dS is indexed by (row, column) and its
    // transpose by (column, row): one of the two cannot be sequential.
    pack_reconfig_data_format(cb_grad_attn_weights, cb_grad_scores);
    pack_tile</* out_of_order */ true>(grad_keep_reg, cb_grad_scores, score_tile);
    pack_reconfig_data_format(cb_grad_scores, cb_grad_scores_transposed);
    pack_tile</* out_of_order */ true>(grad_reg, cb_grad_scores_transposed, transposed_tile);
    pack_reconfig_data_format(cb_grad_scores_transposed, cb_attn_weights_transposed);
    pack_tile</* out_of_order */ true>(attn_reg, cb_attn_weights_transposed, transposed_tile);
    // The probe holds one tile, so only the last pair leaves its dS there.
    if (want_probe) {
        cb_reserve_back(cb_probe, onetile);
        pack_reconfig_data_format(cb_probe);
        pack_tile(grad_keep_reg, cb_probe);
        cb_push_back(cb_probe, onetile);
    }
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_query, cb_key, cb_attention_weights);
    copy_init(cb_query);
    matmul_init(cb_query, cb_key);

#if SEED_DQ
    pack_tiles_to_output(cb_grad_query_seed, cb_grad_query, Bt * qWt);
#endif

#if COMPUTE_STAGE < 5
    // The staged probes look at one tile, so they stay a single-tile tool.
    static_assert(Bt == 1u, "the intermediate stages are only defined for Bt == 1");
    for (uint32_t pair = 0; pair < NUM_PAIRS; ++pair) {
    const bool accumulate = pair > 0;
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
    compute_grad_scores_with_transposes(/* want_probe */ pair + 1u == NUM_PAIRS);

#if COMPUTE_STAGE == 4
    return;
#else
    // ---- the three gradients
    update_grad_query(
        cb_grad_scores, cb_key, cb_grad_query, qWt, block_size, accumulate || (SEED_DQ != 0));
    cb_wait_front(cb_grad_query, qWt);

    update_grad_value(
        cb_attn_weights_transposed, cb_grad_output, cb_grad_value, vWt, block_size, accumulate);
    cb_wait_front(cb_grad_value, vWt);

    update_grad_key(
        cb_grad_scores_transposed,
        cb_query,
        cb_grad_key,
        qWt,
        block_size,
        /* cb_prev_pack */ cb_grad_value,
        /* cb_prev_srca */ cb_grad_output,
        accumulate);
    cb_wait_front(cb_grad_key, qWt);

    // The operands of this pair are done with; the next pair's are behind them.
    cb_pop_front(cb_query, qWt);
    cb_pop_front(cb_key, qWt);
    cb_pop_front(cb_value, vWt);
    cb_pop_front(cb_grad_output, vWt);
    cb_pop_front(cb_lse, onetile);
    cb_pop_front(cb_u_scalar, onetile);
    cb_pop_front(cb_attention_weights, onetile);
    cb_pop_front(cb_grad_attn_weights, onetile);
#endif
#endif
#endif
#endif
    }
#else
    for (uint32_t pair = 0; pair < NUM_PAIRS; ++pair) {
        DeviceZoneScopedN("PAIR");
        const bool accumulate = pair > 0;
        cb_wait_front(cb_query, Bt * qWt);
        cb_wait_front(cb_key, Bt * qWt);
        cb_wait_front(cb_value, Bt * vWt);
        cb_wait_front(cb_grad_output, Bt * vWt);
        cb_wait_front(cb_lse, Bt);
        cb_wait_front(cb_u_scalar, Bt);

        // ---- S = Q K^T / sqrt(d), then P = exp(S - L), a row of tiles at a
        // time. Every column tile of a row is issued into DST before any of
        // them is read back, which is the whole point of Bt > 1: the matmul
        // pipeline latency is paid once for the row instead of once per tile.
        cb_reserve_back(cb_attention_weights, score_tiles);
        for (uint32_t a = 0; a < Bt; ++a) {
            reconfig_data_format(cb_query, cb_key);
            matmul_init(cb_query, cb_key, /* transpose */ 1);
            tile_regs_acquire();
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k = 0; k < qWt; ++k) {
                    matmul_tiles(cb_query, cb_key, a * qWt + k, b * qWt + k, score_reg(b));
                }
            }
            for (uint32_t b = 0; b < Bt; ++b) {
#ifdef DIAGONAL_BLOCK
                if (b == a) {
                    // The one tile the triangle actually cuts through.
                    apply_mask_on_reg(
                        score_reg(b), cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
                } else {
                    binop_with_scalar_tile_init();
                    mul_unary_tile(score_reg(b), scaler_bits);
                }
#else
                binop_with_scalar_tile_init();
                mul_unary_tile(score_reg(b), scaler_bits);
#endif
                softmax_on_dst(score_reg(b), score_reg(b) + 1u, cb_lse, a);
#ifdef DIAGONAL_BLOCK
                if (b > a) {
                    // Wholly above the diagonal. P is zero there, and zeroing
                    // it here is what lets every later sum run over all tiles
                    // without knowing anything about the triangle: dS
                    // inherits the zero through its P factor.
                    binop_with_scalar_tile_init();
                    mul_unary_tile(score_reg(b), /* 0.0f */ 0u);
                }
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_attention_weights);
            for (uint32_t b = 0; b < Bt; ++b) {
                pack_tile</* out_of_order */ true>(
                    score_reg(b), cb_attention_weights, a * Bt + b);
            }
            tile_regs_release();
        }
        cb_push_back(cb_attention_weights, score_tiles);
        cb_wait_front(cb_attention_weights, score_tiles);

        // ---- dP = dO V^T, the same row-at-a-time shape
        cb_reserve_back(cb_grad_attn_weights, score_tiles);
        for (uint32_t a = 0; a < Bt; ++a) {
            reconfig_data_format(cb_grad_output, cb_value);
            matmul_init(cb_grad_output, cb_value, /* transpose */ 1);
            tile_regs_acquire();
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k = 0; k < vWt; ++k) {
                    matmul_tiles(cb_grad_output, cb_value, a * vWt + k, b * vWt + k, b);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_attention_weights, cb_grad_attn_weights);
            for (uint32_t b = 0; b < Bt; ++b) {
                pack_tile</* out_of_order */ true>(b, cb_grad_attn_weights, a * Bt + b);
            }
            tile_regs_release();
        }
        cb_push_back(cb_grad_attn_weights, score_tiles);
        cb_wait_front(cb_grad_attn_weights, score_tiles);

        // ---- dS, and dS^T and P^T alongside, one tile pair at a time. This
        // stage is elementwise and transposes, so there is no latency to
        // amortize and three DST registers per tile to respect.
        cb_reserve_back(cb_grad_scores, score_tiles);
        cb_reserve_back(cb_grad_scores_transposed, score_tiles);
        cb_reserve_back(cb_attn_weights_transposed, score_tiles);
        for (uint32_t a = 0; a < Bt; ++a) {
            for (uint32_t b = 0; b < Bt; ++b) {
                compute_grad_scores_with_transposes(
                    /* want_probe */ (pair + 1u == NUM_PAIRS) && a == 0u && b == 0u, a, b);
            }
        }
        cb_push_back(cb_grad_scores, score_tiles);
        cb_push_back(cb_grad_scores_transposed, score_tiles);
        cb_push_back(cb_attn_weights_transposed, score_tiles);
        cb_wait_front(cb_grad_scores, score_tiles);
        cb_wait_front(cb_grad_scores_transposed, score_tiles);
        cb_wait_front(cb_attn_weights_transposed, score_tiles);

        // ---- dQ_i += dS K_j. The sum that used to be one tile product is
        // now over the block's column tiles, and it accumulates in DST before
        // anything is packed, so the extra depth costs no extra packs.
        const bool seed_dq = accumulate || (SEED_DQ != 0);
        pack_reconfig_data_format(cb_attn_weights_transposed, cb_grad_query);
        if (!seed_dq) {
            cb_reserve_back(cb_grad_query, Bt * qWt);
        } else {
            pack_reconfig_l1_acc(true);
        }
        for (uint32_t a = 0; a < Bt; ++a) {
            for (uint32_t k0 = 0; k0 < qWt; k0 += block_size) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_grad_query, cb_key);
                matmul_init(cb_grad_scores, cb_key, /* transpose */ 0);
                for (uint32_t bi = 0; bi < block_size; ++bi) {
                    for (uint32_t b = 0; b < Bt; ++b) {
                        matmul_tiles(
                            cb_grad_scores, cb_key, a * Bt + b, b * qWt + k0 + bi, bi);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t bi = 0; bi < block_size; ++bi) {
                    pack_tile(bi, cb_grad_query);
                }
                tile_regs_release();
            }
        }
        if (seed_dq) {
            pack_reconfig_l1_acc(false);
            cb_pop_front(cb_grad_query, Bt * qWt);
            cb_reserve_back(cb_grad_query, Bt * qWt);
        }
        cb_push_back(cb_grad_query, Bt * qWt);
        cb_wait_front(cb_grad_query, Bt * qWt);

        // ---- dV_j += P^T dO_i, summed over the block's row tiles
        pack_reconfig_data_format(cb_grad_query, cb_grad_value);
        if (!accumulate) {
            cb_reserve_back(cb_grad_value, Bt * vWt);
        } else {
            pack_reconfig_l1_acc(true);
        }
        for (uint32_t b = 0; b < Bt; ++b) {
            for (uint32_t k0 = 0; k0 < vWt; k0 += block_size) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_grad_value, cb_grad_output);
                matmul_init(cb_attn_weights_transposed, cb_grad_output, /* transpose */ 0);
                for (uint32_t bi = 0; bi < block_size; ++bi) {
                    for (uint32_t a = 0; a < Bt; ++a) {
                        matmul_tiles(
                            cb_attn_weights_transposed,
                            cb_grad_output,
                            b * Bt + a,
                            a * vWt + k0 + bi,
                            bi);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t bi = 0; bi < block_size; ++bi) {
                    pack_tile(bi, cb_grad_value);
                }
                tile_regs_release();
            }
        }
        if (accumulate) {
            pack_reconfig_l1_acc(false);
            cb_pop_front(cb_grad_value, Bt * vWt);
            cb_reserve_back(cb_grad_value, Bt * vWt);
        }
        cb_push_back(cb_grad_value, Bt * vWt);
        cb_wait_front(cb_grad_value, Bt * vWt);

        // ---- dK_j += dS^T Q_i, likewise. The reconfig arguments name what
        // the previous operation left in the packer and in SrcA, which here
        // is the dV update just above.
        pack_reconfig_data_format(cb_grad_value, cb_grad_key);
        if (!accumulate) {
            cb_reserve_back(cb_grad_key, Bt * qWt);
        } else {
            pack_reconfig_l1_acc(true);
        }
        for (uint32_t b = 0; b < Bt; ++b) {
            for (uint32_t k0 = 0; k0 < qWt; k0 += block_size) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_grad_output, cb_query);
                matmul_init(cb_grad_scores_transposed, cb_query, /* transpose */ 0);
                for (uint32_t bi = 0; bi < block_size; ++bi) {
                    for (uint32_t a = 0; a < Bt; ++a) {
                        matmul_tiles(
                            cb_grad_scores_transposed,
                            cb_query,
                            b * Bt + a,
                            a * qWt + k0 + bi,
                            bi);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t bi = 0; bi < block_size; ++bi) {
                    pack_tile(bi, cb_grad_key);
                }
                tile_regs_release();
            }
        }
        if (accumulate) {
            pack_reconfig_l1_acc(false);
            cb_pop_front(cb_grad_key, Bt * qWt);
            cb_reserve_back(cb_grad_key, Bt * qWt);
        }
        cb_push_back(cb_grad_key, Bt * qWt);
        cb_wait_front(cb_grad_key, Bt * qWt);

        // This pair's operands are done with; the next pair's are behind them.
        cb_pop_front(cb_query, Bt * qWt);
        cb_pop_front(cb_key, Bt * qWt);
        cb_pop_front(cb_value, Bt * vWt);
        cb_pop_front(cb_grad_output, Bt * vWt);
        cb_pop_front(cb_lse, Bt);
        cb_pop_front(cb_u_scalar, Bt);
        cb_pop_front(cb_attention_weights, score_tiles);
        cb_pop_front(cb_grad_attn_weights, score_tiles);
        cb_pop_front(cb_grad_scores, score_tiles);
        cb_pop_front(cb_grad_scores_transposed, score_tiles);
        cb_pop_front(cb_attn_weights_transposed, score_tiles);
    }
#endif
}
