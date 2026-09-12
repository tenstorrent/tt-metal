// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The cyclic backward pass on one core: the schedule's T + 1 block pairs,
// each contributing to one row gradient and two column gradients.
//
// Every operand of a timestep arrives from DRAM and every result goes back to
// DRAM, including the column gradients. The paper keeps (K_j, V_j, dK_j,
// dV_j) resident on the owning core and changes columns twice per core, which
// is where its DRAM-traffic argument comes from; this step deliberately does
// not, and reloads them each timestep instead. The arithmetic and the
// schedule are unaffected, and the residency rules are validated separately
// by the transport probe -- which poisons the column pages in DRAM so that a
// first visit reading them is caught. Restoring residency belongs with the
// relay, which is where it pays.
//
// What that buys is a compute kernel with no state carried between
// timesteps: each timestep is the single block pair of
// cyclic_pair_compute.cpp, which is separately tested, wrapped in a loop.
// Every gradient uses the same three buffers -- a seed from the reader, an
// accumulator, and an output the writer takes -- because
//
//   * sdpa_bw's accumulate path packs without reserving, and only a
//     reserve/push cycle by *this* kernel leaves the packer's write pointer
//     where that expects; a push from the reader does not, since the write
//     pointers are per RISC. The copy from seed to accumulator is what
//     establishes it;
//   * a buffer the reader fills and this kernel accumulates onto would let
//     the writer's cb_wait_front be satisfied by the reader's push, and it
//     would write the value back unaccumulated. The copy out gives the writer
//     a buffer pushed exactly once per result.
//
// Only dQ needs the chip-wide barrier. Each column belongs to exactly one
// core, so dK and dV are never contended.

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp"

// COLUMN_RESIDENT: keep the whole column state across a residency interval
// instead of taking it fresh every timestep, which is the paper's
// column-state management. Each core changes its resident column exactly
// twice over T + 1 timesteps, always at a diagonal block, giving three
// intervals: its first column, its other column, then the first again.
//
// K_j and V_j are read once per interval, and this kernel releases the
// storage by popping at the change. dK_j and dV_j accumulate in L1 for the
// whole interval and are handed over once, at its end, for the write kernel
// to store. At an interval start they either begin from nothing -- a first
// visit, where the first update overwrites rather than accumulates, so the
// zeros never come from DRAM -- or from the value in DRAM, on the one
// revisit each core makes.
//
// Algorithm 2's reader still supplies the column per timestep, so the two
// have to agree on who pops and when, hence a switch rather than a change.
#ifndef COLUMN_RESIDENT
#define COLUMN_RESIDENT 0
#endif

#ifndef RELEASE_TOKEN
#define RELEASE_TOKEN 0
#endif

// FOLD_SCALE_INTO_KEY: where the softmax's 1/sqrt(d) lives.
//
// It appears twice in the arithmetic, on the scores and again on dS, and both
// can go by scaling K instead:
//
//     S  = Q (aK)^T = a Q K^T     the score path then needs no scale at all,
//     dQ = G (aK)   = a G K       with dS left unscaled,
//     dK = G^T Q                  short by a -- but dK is an accumulator, so
//                                 one scale at its handover fixes it,
//     dV = P^T dO                 untouched.
//
// K is resident for a whole residency interval, so scaling it costs one pass
// per interval, twice per core over the run, against Bt * Bt passes on dS and
// Bt on the statistic every timestep. At Bt = 4 that removes 20 SFPU passes
// per timestep.
//
// Only where it is exact: K is bfloat16, so a * K rounds unless a is a power
// of two -- d = 64 and 256 yes, d = 128 no -- and the host enables this only
// then. The other path keeps the scale folded into the exponential instead.
#ifndef FOLD_SCALE_INTO_KEY
#define FOLD_SCALE_INTO_KEY 0
#endif

// RELEASE_TOKEN: publish a token once this timestep's packet slot has been
// popped, so the relay reader knows the slot is free and can hand its
// producer the credit at the release -- the paper's timing -- instead of two
// timesteps later when it reserves the slot itself. The reader cannot see the
// pop any other way: the release happens on this RISC, and a circular
// buffer's acked count is not something the other side can read.

namespace {

constexpr uint32_t kCores = get_compile_time_arg_val(0);
constexpr uint32_t qWt = get_compile_time_arg_val(1);
constexpr uint32_t vWt = get_compile_time_arg_val(2);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(4);
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(5);
constexpr uint32_t block_size = get_compile_time_arg_val(6);
// Row-tiles per block: B = Bt * 32. What it buys is in the block-pair
// kernel's commit message and in docs/overlaps.md -- at Bt = 1 the score
// stages are one tile each and the matmul pipeline latency has nothing to
// hide behind.
constexpr uint32_t Bt = get_compile_time_arg_val(7);
// sqrt(d): the exponential applies the softmax scale to its whole argument,
// so the statistic being subtracted is divided by it first -- once per row of
// score tiles instead of a scale pass on each of the Bt tiles.
constexpr uint32_t inv_scaler_bits = get_compile_time_arg_val(8);
constexpr uint32_t score_tiles = Bt * Bt;

// Score tiles are contiguous, with two shared scratch registers above them:
// one for the row's broadcast statistic, one for the causal mask. Bt + 2
// registers where per-tile scratch would need 2 * Bt, and it is what lets the
// row's statistic be broadcast once rather than once per score tile.
constexpr uint32_t score_reg(uint32_t b) {
    return b;
}
constexpr uint32_t stat_reg = Bt;
constexpr uint32_t mask_reg = Bt + 1u;

// Operands, all per timestep.
constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_key_scaled = tt::CBIndex::c_27;  // a * K, when exact
#if FOLD_SCALE_INTO_KEY
constexpr uint32_t cb_key_operand = cb_key_scaled;
#else
constexpr uint32_t cb_key_operand = cb_key;
#endif
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
constexpr uint32_t cb_lse = tt::CBIndex::c_4;
constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;

// Intermediates.
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_11;
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_12;
constexpr uint32_t cb_grad_scores_transposed = tt::CBIndex::c_13;
constexpr uint32_t cb_attn_weights_transposed = tt::CBIndex::c_14;

// Each gradient: what the reader loaded, the accumulator, the writer's copy.
constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
constexpr uint32_t cb_grad_query_accum = tt::CBIndex::c_16;
constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
constexpr uint32_t cb_grad_key_accum = tt::CBIndex::c_19;
constexpr uint32_t cb_grad_key_out = tt::CBIndex::c_20;
constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;
constexpr uint32_t cb_grad_value_accum = tt::CBIndex::c_22;
constexpr uint32_t cb_grad_value_out = tt::CBIndex::c_23;

// dS, dS^T and P^T out of one acquire region. dS must not be written to a CB
// and read back to be transposed: the transposed buffers unpack in Default
// mode, because matmul Src registers do not support Float32 unpack, so a
// copy_tile out of one lands in DST in a layout the 32-bit transpose_dest
// scrambles. copy_dest_values duplicates dS inside DST instead.
#if FOLD_SCALE_INTO_KEY
// pack_tiles_to_output with a multiply on the way through.
//
// dK needs it at both ends of its life. Folding the scale into K leaves the
// accumulator short by a factor of a, so the handover multiplies by a -- but
// a revisit then reads that already-scaled value back from DRAM as its seed,
// so the seed is divided by a again first. Both are Bt * qWt tiles once per
// residency interval, which is twice per core over the whole run.
void pack_tiles_scaled(
    const uint32_t cb_source, const uint32_t cb_output, const uint32_t num_tiles, const uint32_t scale_bits) {
    cb_wait_front(cb_source, num_tiles);
    cb_reserve_back(cb_output, num_tiles);

    pack_reconfig_data_format(cb_output);
    reconfig_data_format(cb_source, cb_source);

    copy_init(cb_source);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tile_regs_acquire();
        copy_tile(cb_source, tile_idx, /* register idx */ 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(/* register idx */ 0, scale_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/* register idx */ 0, cb_output);
        tile_regs_release();
    }
    cb_push_back(cb_output, num_tiles);
    cb_pop_front(cb_source, num_tiles);
}
#endif

// sdpa_bw's apply_mask_on_reg with the scratch register named rather than
// assumed to be the next one along: score tiles are contiguous here, so the
// register after one is another score tile.
void apply_mask_at(
    const uint32_t scores_reg,
    const uint32_t mask_register,
    const uint32_t cb_mask,
    const uint32_t minus_one,
    const uint32_t custom_inf) {
    copy_init(cb_mask);
    copy_tile(cb_mask, /* tile_idx */ 0, mask_register);

    mask_tile_init();
    mask_tile(scores_reg, mask_register);

    // No scale here: the exponential applies it, and the mask's minus
    // infinity survives being scaled either way.
    binop_with_scalar_tile_init();
    add_unary_tile(mask_register, minus_one);
    mul_unary_tile(mask_register, custom_inf);

    add_binary_tile_init();
    add_binary_tile(scores_reg, mask_register, scores_reg);
}

// Broadcast a row's statistic into DST. Split out of the softmax step because
// it is the same for every column tile of the row: sdpa_bw folds the two
// together, which is right when a row has one score tile and wasteful when it
// has Bt of them.
void broadcast_statistic_to_dst(
    const uint32_t tmp_reg, const uint32_t cb_statistics, const uint32_t stat_tile) {
    reconfig_data_format_srcb(cb_statistics);
    UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_statistics)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          ckernel::DataCopyType::B2D,
          DST_ACCUM_MODE,
          BroadcastType::COL>(cb_statistics)));
    unary_bcast<BroadcastType::COL>(cb_statistics, stat_tile, tmp_reg);
}

// P = exp(a(S - L/a)) for every score tile of one row, with the scale folded
// into the exponential rather than applied to S beforehand.
//
// sdpa_exp_tile_scaled folds the whole FP32 scale into LREG12 at init time on
// Blackhole -- one SFPU pass per score tile that no longer happens -- and
// pre-multiplies by a bfloat16 scale on Wormhole. It computes exp(a * x), so
// the caller supplies L/a and the identity exp(a(S - L/a)) = exp(aS - L) does
// the rest. sdpa_fw already works this way, keeping its scores and its
// running maximum unscaled.
//
// Both SFPU programs are configured once for the row rather than once per
// tile, which is only possible because the subtracts and the exponentials are
// no longer interleaved: an init between them would reprogram the unit.
void subtract_and_exp_row(const uint32_t first_reg, const uint32_t count, const uint32_t broadcast_reg) {
    sub_binary_tile_init();
    for (uint32_t b = 0; b < count; ++b) {
        sub_binary_tile(first_reg + b, broadcast_reg, first_reg + b);
    }

#if FOLD_SCALE_INTO_KEY
    // S already carries the scale, having come out of Q (aK)^T.
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ false>();
#else
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ true, scaler_bits>();
#endif
    for (uint32_t b = 0; b < count; ++b) {
        sdpa_exp_tile(first_reg + b);
    }
}

// One tile pair of the block. The caller reserves and pushes the three
// output buffers around the whole block, so this only packs.
void grad_scores_and_transposes(uint32_t a, uint32_t b) {
    const uint32_t score_tile = a * Bt + b;
    const uint32_t transposed_tile = b * Bt + a;

    constexpr uint32_t grad_reg = 0;
    constexpr uint32_t attn_reg = 1;
    constexpr uint32_t grad_keep_reg = 2;

    tile_regs_acquire();
    reconfig_data_format(cb_grad_attn_weights, cb_u_scalar);
    sub_bcast_cols_init(cb_grad_attn_weights, cb_u_scalar);
    sub_tiles_bcast_cols(cb_grad_attn_weights, cb_u_scalar, score_tile, a, grad_reg);

    reconfig_data_format_srca(cb_grad_attn_weights, cb_attention_weights);
    copy_init(cb_attention_weights);
    copy_tile(cb_attention_weights, score_tile, attn_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_reg, grad_reg);
#if !FOLD_SCALE_INTO_KEY
    binop_with_scalar_tile_init();
    mul_unary_tile(grad_reg, scaler_bits);
#endif

    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(grad_reg, grad_keep_reg);

    transpose_dest_init</* is_32bit */ true>(cb_attention_weights);
    transpose_dest</* is_32bit */ true>(grad_reg);
    transpose_dest</* is_32bit */ true>(attn_reg);

    tile_regs_commit();
    tile_regs_wait();
    // dS is indexed (row, column) and its transpose (column, row), so one of
    // the two cannot be sequential: both transposed operands go out of order.
    pack_reconfig_data_format(cb_grad_attn_weights, cb_grad_scores);
    pack_tile</* out_of_order */ true>(grad_keep_reg, cb_grad_scores, score_tile);
    pack_reconfig_data_format(cb_grad_scores, cb_grad_scores_transposed);
    pack_tile</* out_of_order */ true>(grad_reg, cb_grad_scores_transposed, transposed_tile);
    pack_reconfig_data_format(cb_grad_scores_transposed, cb_attn_weights_transposed);
    pack_tile</* out_of_order */ true>(attn_reg, cb_attn_weights_transposed, transposed_tile);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores);
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;

#if COLUMN_RESIDENT
    // Which columns this core owns, whether each has been resident before --
    // a first visit starts the gradients from nothing, a revisit from DRAM --
    // and whether the current interval's updates accumulate or overwrite.
    const auto owned = sched.owned_columns(my_core);
    bool visited[2] = {false, false};
    bool column_accumulating = false;
#endif

    static_assert(
        FOLD_SCALE_INTO_KEY == 0 || COLUMN_RESIDENT == 1,
        "folding the scale into K assumes the resident column path: without residency dK is handed "
        "over every timestep and its seed re-read every timestep, so the scale would compound");

    compute_kernel_hw_startup(cb_query, cb_key, cb_attention_weights);
    copy_init(cb_query);
    matmul_init(cb_query, cb_key);
    cb_wait_front(cb_attn_mask, onetile);

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        const bool diagonal = pair.i == pair.j;

#if COLUMN_RESIDENT
        // Popped only when the column changes, which releases the storage for
        // the next column. Waiting every timestep is free once it is there.
        const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != pair.j);
        const bool column_ends =
            (t + 1u == kTimesteps) || (sched.pair(my_core, t + 1u).j != pair.j);
        const uint32_t owned_slot = (pair.j == owned.first) ? 0u : 1u;
        if (column_changed && t > 0u) {
            cb_pop_front(cb_key, Bt * qWt);
            cb_pop_front(cb_value, Bt * vWt);
#if FOLD_SCALE_INTO_KEY
            cb_pop_front(cb_key_scaled, Bt * qWt);
#endif
        }
#if FOLD_SCALE_INTO_KEY
        // a * K, once for the residency interval. Everything that reads K --
        // the scores and dQ -- reads this copy instead.
        if (column_changed) {
            cb_wait_front(cb_key, Bt * qWt);
            cb_reserve_back(cb_key_scaled, Bt * qWt);
            reconfig_data_format_srca(cb_key);
            copy_init(cb_key);
            pack_reconfig_data_format(cb_key_scaled);
            for (uint32_t t0 = 0; t0 < Bt * qWt; ++t0) {
                tile_regs_acquire();
                copy_tile(cb_key, t0, 0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scaler_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_key_scaled);
                tile_regs_release();
            }
            cb_push_back(cb_key_scaled, Bt * qWt);
            cb_wait_front(cb_key_scaled, Bt * qWt);
        }
#endif
        if (column_changed) {
            if (visited[owned_slot]) {
                // A revisit: the interval starts from what is in DRAM.
#if FOLD_SCALE_INTO_KEY
                pack_tiles_scaled(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt, inv_scaler_bits);
#else
                pack_tiles_to_output(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt);
#endif
                pack_tiles_to_output(cb_grad_value_seed, cb_grad_value_accum, Bt * vWt);
                column_accumulating = true;
            } else {
                // A first visit: the first update writes rather than adds, so
                // the gradients start at zero without reading zeros.
                column_accumulating = false;
                visited[owned_slot] = true;
            }
        }
#endif
        {
            DeviceZoneScopedN("WAIT-PACKET");
            cb_wait_front(cb_query, Bt * qWt);
            cb_wait_front(cb_key, Bt * qWt);
            cb_wait_front(cb_value, Bt * vWt);
            cb_wait_front(cb_grad_output, Bt * vWt);
            cb_wait_front(cb_lse, Bt);
            cb_wait_front(cb_u_scalar, Bt);
        }

        // ---- S = Q K^T / sqrt(d), masked when i == j, then P = exp(S - L).
        // A row of tiles at a time: every column tile of the row is issued
        // into DST before any is read back, so the matmul pipeline latency is
        // paid once for the row rather than once per tile. At Bt = 1 that is
        // one tile and the latency -- measured at 1.4 us against 0.076 for
        // each tile issued behind it -- is entirely exposed.
        {
        DeviceZoneScopedN("SCORES");
        cb_reserve_back(cb_attention_weights, score_tiles);
        for (uint32_t a = 0; a < Bt; ++a) {
            reconfig_data_format(cb_query, cb_key_operand);
            matmul_init(cb_query, cb_key_operand, /* transpose */ 1);
            tile_regs_acquire();
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k = 0; k < qWt; ++k) {
                    matmul_tiles(
                        cb_query, cb_key_operand, a * qWt + k, b * qWt + k, score_reg(b));
                }
            }
            if (diagonal) {
                apply_mask_at(
                    score_reg(a), mask_reg, cb_attn_mask, minus_one_bits, custom_inf_bits);
            }

            // One broadcast for the row, one scale of it, and one
            // configuration of each SFPU program -- all per row rather than
            // per score tile.
            broadcast_statistic_to_dst(stat_reg, cb_lse, a);
#if !FOLD_SCALE_INTO_KEY
            // The exponential carries the scale, so what it subtracts must be
            // divided by it first.
            binop_with_scalar_tile_init();
            mul_unary_tile(stat_reg, inv_scaler_bits);
#endif
            subtract_and_exp_row(score_reg(0), Bt, stat_reg);
            for (uint32_t b = 0; b < Bt; ++b) {
                if (diagonal && b > a) {
                    // Wholly above the diagonal. Zeroing P there lets every
                    // later sum run over all of the block's tiles without
                    // knowing about the triangle: dS inherits the zero
                    // through its P factor.
                    binop_with_scalar_tile_init();
                    mul_unary_tile(score_reg(b), /* 0.0f */ 0u);
                }
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
        }

        // ---- dP = dO V^T, the same shape
        {
            DeviceZoneScopedN("GRAD-WEIGHTS");
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
        }

        // ---- dS, with dS^T and P^T alongside, one tile pair at a time:
        // elementwise work with three registers per tile and no latency to
        // amortize.
        {
            DeviceZoneScopedN("GRAD-SCORES");
            cb_reserve_back(cb_grad_scores, score_tiles);
            cb_reserve_back(cb_grad_scores_transposed, score_tiles);
            cb_reserve_back(cb_attn_weights_transposed, score_tiles);
            for (uint32_t a = 0; a < Bt; ++a) {
                for (uint32_t b = 0; b < Bt; ++b) {
                    grad_scores_and_transposes(a, b);
                }
            }
            cb_push_back(cb_grad_scores, score_tiles);
            cb_push_back(cb_grad_scores_transposed, score_tiles);
            cb_push_back(cb_attn_weights_transposed, score_tiles);
            cb_wait_front(cb_grad_scores, score_tiles);
            cb_wait_front(cb_grad_scores_transposed, score_tiles);
            cb_wait_front(cb_attn_weights_transposed, score_tiles);
        }


        // ---- dQ_i = (dQ_i from DRAM) + dS K_j. The sum runs over the
        // block's column tiles as well as the head dimension, and it
        // accumulates in DST, so the extra depth costs no extra packs. The
        // loops replace update_grad_query, which cannot express the inner
        // sum, but keep its reconfig arguments and its L1-accumulate dance.
        {
            DeviceZoneScopedN("SEED-DQ");
            pack_tiles_to_output(cb_grad_query_seed, cb_grad_query_accum, Bt * qWt);
        }
        {
            DeviceZoneScopedN("UPDATE-DQ");
            pack_reconfig_data_format(cb_grad_scores, cb_grad_query_accum);
            pack_reconfig_l1_acc(true);
            for (uint32_t a = 0; a < Bt; ++a) {
                for (uint32_t k0 = 0; k0 < qWt; k0 += block_size) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_grad_query_accum, cb_key_operand);
                    matmul_init(cb_grad_scores, cb_key_operand, /* transpose */ 0);
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        for (uint32_t b = 0; b < Bt; ++b) {
                            matmul_tiles(
                                cb_grad_scores,
                                cb_key_operand,
                                a * Bt + b,
                                b * qWt + k0 + bi,
                                bi);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        pack_tile(bi, cb_grad_query_accum);
                    }
                    tile_regs_release();
                }
            }
            pack_reconfig_l1_acc(false);
            cb_pop_front(cb_grad_query_accum, Bt * qWt);
            cb_reserve_back(cb_grad_query_accum, Bt * qWt);
            cb_push_back(cb_grad_query_accum, Bt * qWt);
            cb_wait_front(cb_grad_query_accum, Bt * qWt);
        }
        {
            DeviceZoneScopedN("EMIT-DQ");
            pack_tiles_to_output(cb_grad_query_accum, cb_grad_query_out, Bt * qWt);
        }

        // ---- dV_j += P^T dO_i, summed over the block's row tiles
        {
            DeviceZoneScopedN("UPDATE-DV");
#if COLUMN_RESIDENT
            const bool dv_accumulate = column_accumulating;
#else
            pack_tiles_to_output(cb_grad_value_seed, cb_grad_value_accum, Bt * vWt);
            const bool dv_accumulate = true;
#endif
            pack_reconfig_data_format(cb_attn_weights_transposed, cb_grad_value_accum);
            if (!dv_accumulate) {
                cb_reserve_back(cb_grad_value_accum, Bt * vWt);
            } else {
                pack_reconfig_l1_acc(true);
            }
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k0 = 0; k0 < vWt; k0 += block_size) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_grad_value_accum, cb_grad_output);
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
                        pack_tile(bi, cb_grad_value_accum);
                    }
                    tile_regs_release();
                }
            }
            if (dv_accumulate) {
                pack_reconfig_l1_acc(false);
                cb_pop_front(cb_grad_value_accum, Bt * vWt);
                cb_reserve_back(cb_grad_value_accum, Bt * vWt);
            }
            cb_push_back(cb_grad_value_accum, Bt * vWt);
            cb_wait_front(cb_grad_value_accum, Bt * vWt);
#if !COLUMN_RESIDENT
            pack_tiles_to_output(cb_grad_value_accum, cb_grad_value_out, Bt * vWt);
#endif
        }

        // ---- dK_j += dS^T Q_i. The reconfig arguments must name what the
        // previous operation actually left in the packer and in SrcA, because
        // those reconfigs are conditional and skip when the formats already
        // match: naming cb_grad_output as the previous SrcA -- which is what
        // sdpa_bw's own call site does, its previous operation being a
        // different one -- leaves the unpacker in Float32 while this matmul
        // needs Q in bfloat16, and dK comes out wrong while dQ and dV are
        // fine. Here the previous operation is the dV update just above.
        {
            DeviceZoneScopedN("UPDATE-DK");
#if COLUMN_RESIDENT
            const bool dk_accumulate = column_accumulating;
#else
            pack_tiles_to_output(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt);
            const bool dk_accumulate = true;
#endif
#if COLUMN_RESIDENT
            // The previous operation is the dV update, so its accumulator is
            // what the packer was last set from and dO what SrcA was.
            constexpr uint32_t cb_prev_pack = cb_grad_value_accum;
            constexpr uint32_t cb_prev_srca = cb_grad_output;
#else
            // Here the previous operation is the dK seed copy just above,
            // which packed to the accumulator and read the seed. Naming dO as
            // the previous SrcA -- correct in the resident path -- makes the
            // reconfig below look unnecessary, because dO and Q are both
            // bfloat16, and leaves the unpacker in Float32 where the seed copy
            // put it. dK then comes out wrong while dQ and dV are fine.
            constexpr uint32_t cb_prev_pack = cb_grad_key_accum;
            constexpr uint32_t cb_prev_srca = cb_grad_key_seed;
#endif
            pack_reconfig_data_format(cb_prev_pack, cb_grad_key_accum);
            if (!dk_accumulate) {
                cb_reserve_back(cb_grad_key_accum, Bt * qWt);
            } else {
                pack_reconfig_l1_acc(true);
            }
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k0 = 0; k0 < qWt; k0 += block_size) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_prev_srca, cb_query);
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
                        pack_tile(bi, cb_grad_key_accum);
                    }
                    tile_regs_release();
                }
            }
            if (dk_accumulate) {
                pack_reconfig_l1_acc(false);
                cb_pop_front(cb_grad_key_accum, Bt * qWt);
                cb_reserve_back(cb_grad_key_accum, Bt * qWt);
            }
            cb_push_back(cb_grad_key_accum, Bt * qWt);
            cb_wait_front(cb_grad_key_accum, Bt * qWt);
#if COLUMN_RESIDENT
            column_accumulating = true;
            // Hand both column gradients over once, at the end of the interval.
            if (column_ends) {
                pack_tiles_to_output(cb_grad_value_accum, cb_grad_value_out, Bt * vWt);
#if FOLD_SCALE_INTO_KEY
                pack_tiles_scaled(cb_grad_key_accum, cb_grad_key_out, Bt * qWt, scaler_bits);
#else
                pack_tiles_to_output(cb_grad_key_accum, cb_grad_key_out, Bt * qWt);
#endif
            }
#else
            pack_tiles_to_output(cb_grad_key_accum, cb_grad_key_out, Bt * qWt);
#endif
        }

        cb_pop_front(cb_query, Bt * qWt);
#if !COLUMN_RESIDENT
        cb_pop_front(cb_key, Bt * qWt);
        cb_pop_front(cb_value, Bt * vWt);
#endif
        cb_pop_front(cb_grad_output, Bt * vWt);
        cb_pop_front(cb_lse, Bt);
        cb_pop_front(cb_u_scalar, Bt);
        cb_pop_front(cb_attention_weights, score_tiles);
        cb_pop_front(cb_grad_attn_weights, score_tiles);
        // The helpers used to pop these; the loops above read them by index
        // and leave them alone, so they are released here with the rest.
        cb_pop_front(cb_grad_scores, score_tiles);
        cb_pop_front(cb_grad_scores_transposed, score_tiles);
        cb_pop_front(cb_attn_weights_transposed, score_tiles);

#if RELEASE_TOKEN
        // Slot t mod 2 is free now: every read of it is done.
        cb_reserve_back(cb_slot_release, 1);
        cb_push_back(cb_slot_release, 1);
#endif
    }
}
