// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The cyclic forward pass on one core (tt-flash-attn, Algorithm 8): the
// schedule's block pairs, each folding one key/value column into the
// online-softmax state of one query row block.
//
// Everything is computed in the transposed orientation of the backward
// kernel: S^T = K Q^T, P^T = exp(a (S^T - m)), and the accumulator is
// O^T = V^T P^T, every tile transposed within itself, so that the matmuls
// need no transposed score tile and the per-query statistics m and l are
// row broadcasts (one value per column of S^T, in row 0 of a tile).
//
// Per timestep, for the pair (i, j):
//
//   1. S^T for every live key tile, packed to L1 twice: rounded to the 19
//      bits the FPU reads for the column maximum, and exact for the
//      subtraction;
//   2. per query tile: m_new = max(m_old, colmax S^T), r = exp(a (m_old - m_new));
//   3. P^T = exp(a (S^T - m_new)), the subtraction on the SFPU from the
//      exact copy, packed rounded for the matmul;
//   4. l_new = r l_old + colsum P^T;
//   5. O^T <- r O^T + V^T P^T, the rescale on the SFPU from the exact seed,
//      the products accumulated by the FPU on top;
//   6. at the row's last visit: O = (O^T / l)^T in bfloat16 and
//      lse = a m + ln l in column layout, for the write kernel.
//
// The statistics' representation. The column reductions produce a row-layout
// tile (one value per query in row 0; the other rows are masked to zero,
// since the reduction leaves partial results there). m travels as a *full*
// tile, every row the same, made once per visit by the FPU as ones x tile
// (a matmul against the all-ones tile, which reads the row-layout block
// maximum through a source register at 19 bits -- exact, since it is a
// maximum of 19-bit-rounded scores); a full tile needs no broadcast, so
// S^T - m, r = exp(a (m_old - m_new)) and the rescales of O^T and l are
// plain SFPU operations on exact copies, and the running sum l stays exact.
// l itself travels in row layout; the only 19-bit read of it is the FPU
// broadcast of 1/l at the row's last visit, which the bfloat16 output does
// not notice. All SFPU work is on the math thread: the SFPU's programmable
// constants are shared between the threads, and a pack-thread exponential
// was corrupted by the math thread's reciprocal and logarithm (measured:
// the exponent came out scaled by 11).
//
// The state travels in the packet: read through the seed views (c_13 m,
// c_14 l, c_15 O^T), packed back through the out views (c_18, c_19, c_17)
// onto the same memory. At a row's first streak there is no state and the
// first update writes.

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"

#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif

#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

namespace {

constexpr uint32_t kCores = get_compile_time_arg_val(0);
constexpr uint32_t qWt = get_compile_time_arg_val(1);
constexpr uint32_t vWt = get_compile_time_arg_val(2);
// The softmax scale a = 1/sqrt(d): folded into the exponential's constant
// (it computes exp(a x)) and applied to m in the lse.
constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
constexpr uint32_t block_size = get_compile_time_arg_val(4);
constexpr uint32_t Bt = get_compile_time_arg_val(5);
constexpr uint32_t score_tiles = Bt * Bt;

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_value_t = tt::CBIndex::c_16;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
constexpr uint32_t kMaskTriangle = 0;
constexpr uint32_t cb_zero_tile = tt::CBIndex::c_8;
constexpr uint32_t cb_transpose_fence = tt::CBIndex::c_8;
constexpr uint32_t cb_ones_column = tt::CBIndex::c_28;
constexpr uint32_t cb_ones_row = tt::CBIndex::c_29;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_27;
constexpr uint32_t cb_scores = tt::CBIndex::c_10;        // S^T, 19-bit rounded: the reduce's operand
constexpr uint32_t cb_scores_exact = tt::CBIndex::c_11;  // S^T, exact: the subtraction's source
constexpr uint32_t cb_probs = tt::CBIndex::c_12;         // P^T, 19-bit rounded
constexpr uint32_t cb_rescale = tt::CBIndex::c_20;       // r, full tile, exact
constexpr uint32_t cb_block_max = tt::CBIndex::c_23;     // colmax S^T, row layout (scratch)
constexpr uint32_t cb_max_seed = tt::CBIndex::c_13;      // m, full tile, exact
constexpr uint32_t cb_sum_seed = tt::CBIndex::c_14;      // l, exact (unpack to dest)
constexpr uint32_t cb_sum_plain = tt::CBIndex::c_26;     // l, the same memory, for the FPU broadcast
constexpr uint32_t cb_out_seed = tt::CBIndex::c_15;
constexpr uint32_t cb_max_out = tt::CBIndex::c_18;
constexpr uint32_t cb_sum_out = tt::CBIndex::c_19;
constexpr uint32_t cb_out_out = tt::CBIndex::c_17;
constexpr uint32_t cb_output = tt::CBIndex::c_21;
constexpr uint32_t cb_lse = tt::CBIndex::c_22;
constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;

// Math fidelity per matmul, as the backward measured them: the scores are
// neutral at HiFi3; the products with a 19-bit operand in SrcA run at HiFi4.
#ifndef FID_S
#define FID_S 3
#endif
#ifndef FID_O
#define FID_O 4
#endif
constexpr MathFidelity fid(int phases) {
    return phases == 2 ? MathFidelity::HiFi2 : phases == 3 ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
}
constexpr MathFidelity kFidS = fid(FID_S), kFidO = fid(FID_O);
template <MathFidelity MF>
void mm_init(uint32_t in0, uint32_t in1, uint32_t transpose) {
    MATH((llk_math_matmul_init<MF, MM_THROTTLE>(in0, in1, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose)));
}
template <MathFidelity MF>
void mm_tiles(uint32_t in0, uint32_t in1, uint32_t t0, uint32_t t1, uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, t0, t1)));
    MATH((llk_math_matmul<MF, MM_THROTTLE>(idst)));
}

// The packer's "round to a 10-bit mantissa" control: on, a Float32 pack
// lands the 19 bits the Src registers keep, rounded to nearest rather than
// truncated on the way back in. Off for everything the SFPU reads exactly.
void pack_rounding(const bool on) {
    PACK((cfg_reg_rmw_tensix<
          PCK_DEST_RD_CTRL_Round_10b_mant_ADDR32,
          PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT,
          PCK_DEST_RD_CTRL_Round_10b_mant_MASK>(on ? 1u : 0u)));
}

// The unpacker waits for everything the pack thread has packed so far (see
// the backward kernel): before a dest-register transpose's neighbours, and
// before reading back what was just packed.
void unpacker_fence() {
    cb_reserve_back(cb_transpose_fence, 1);
    cb_push_back(cb_transpose_fence, 1);
    cb_wait_front(cb_transpose_fence, 1);
    cb_pop_front(cb_transpose_fence, 1);
}

// V_j^T from the resident V_j: Bt x vWt bf16 tiles in, vWt x Bt out, each
// tile transposed by the unpacker on the way into DST. Once per residency
// interval; it is what lets O^T be a plain matmul with P^T as its second operand.
void transpose_value_block() {
    cb_wait_front(cb_value, Bt * vWt);
    cb_reserve_back(cb_value_t, Bt * vWt);
    pack_reconfig_data_format(cb_value_t);
    reconfig_data_format_srca(cb_value);
    transpose_init(cb_value);
    for (uint32_t e = 0; e < vWt; ++e) {
        for (uint32_t b = 0; b < Bt; ++b) {
            tile_regs_acquire();
            transpose_tile(cb_value, b * vWt + e, /* register idx */ 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(/* register idx */ 0, cb_value_t);  // tile e * Bt + b
            tile_regs_release();
        }
    }
    cb_push_back(cb_value_t, Bt * vWt);
    cb_wait_front(cb_value_t, Bt * vWt);
}

// A row-layout statistic broadcast down the rows of a DST tile: ones x tile,
// one matmul (HiFi4; the statistic goes through SrcA at 19 bits, see the
// top of the file). Leaves the matmul configured for (ones, cb).
void broadcast_rows(const uint32_t cb_stat, const uint32_t tile, const uint32_t idst) {
    reconfig_data_format(cb_stat, cb_reduce_scaler);
    mm_init<MathFidelity::HiFi4>(cb_reduce_scaler, cb_stat, /* transpose */ 0);
    mm_tiles<MathFidelity::HiFi4>(cb_reduce_scaler, cb_stat, 0, tile, idst);
}

// exp(a x) in place on a DST tile, on the math thread's SFPU.
void exp_scaled(const uint32_t idst) {
    binop_with_scalar_tile_init();
    mul_unary_tile(idst, scaler_bits);
    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(idst);
}



}  // namespace

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t slice_count = get_arg_val<uint32_t>(1);

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    using ttml::metal::ops::cyclic_sdpa_bw::kNoCore;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();

    compute_kernel_hw_startup(cb_query, cb_key, cb_scores);
    copy_init(cb_query);
    matmul_init(cb_key, cb_query);
    cb_wait_front(cb_attn_mask, 2);
    cb_wait_front(cb_ones_column, 1);
    cb_wait_front(cb_ones_row, 1);
    cb_wait_front(cb_reduce_scaler, 1);

    for (uint32_t s = 0; s < slice_count; ++s) {
    for (uint32_t t = 0; t < kTimesteps; ++t) {
        DeviceZoneScopedN("T-STEP");
        const auto pair = sched.pair(my_core, t);
        const uint32_t g = s * kTimesteps + t;
        const bool diagonal = (DENSE_MODE == 0) && (pair.i == pair.j);
        const bool skip_masked = (Bt > 1u) && diagonal;
        // Where the row stands: a first streak start has no state to seed
        // from; the row's last visit finishes it.
        const bool fresh = !sched.producer(my_core, t).internal && !sched.is_later_streak_start(pair.i, t);
        const bool final = sched.next_consumer(pair.i, t) == kNoCore && !sched.has_later_active(pair.i, t);
        // Live key tiles of query tile a on a diagonal pair: b <= a.
        const auto n_live = [&](uint32_t a) { return skip_masked ? a + 1u : Bt; };

        // ---- column state
        const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != pair.j);
        if (column_changed && g > 0u) {
            cb_pop_front(cb_key, Bt * qWt);
            cb_pop_front(cb_value, Bt * vWt);
            cb_pop_front(cb_value_t, Bt * vWt);
        }
        if (column_changed) {
            DeviceZoneScopedN("COL-CHANGE");
            transpose_value_block();
        }
        {
            DeviceZoneScopedN("WAIT-PACKET");
            cb_wait_front(cb_query, Bt * qWt);
            cb_wait_front(cb_key, Bt * qWt);
            cb_wait_front(cb_max_seed, Bt);
            cb_wait_front(cb_sum_seed, Bt);
            cb_wait_front(cb_sum_plain, Bt);
            cb_wait_front(cb_out_seed, Bt * qWt);
        }
        cb_reserve_back(cb_max_out, Bt);
        cb_reserve_back(cb_sum_out, Bt);
        cb_reserve_back(cb_out_out, Bt * qWt);

        // ---- 1. S^T = K Q^T, a column of the score grid at a time, packed twice.
        {
            DeviceZoneScopedN("SCORES");
            cb_reserve_back(cb_scores, score_tiles);
            cb_reserve_back(cb_scores_exact, score_tiles);
            pack_reconfig_data_format(cb_scores);
            for (uint32_t a = 0; a < Bt; ++a) {
                const uint32_t live = n_live(a);
                tile_regs_acquire();
                reconfig_data_format(cb_query, cb_key);
                mm_init<kFidS>(cb_key, cb_query, /* transpose */ 1);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    for (uint32_t k = 0; k < qWt; ++k) {
                        mm_tiles<kFidS>(cb_key, cb_query, b * qWt + k, a * qWt + k, b);
                    }
                }
                if (diagonal) {
                    // The triangle, -inf where the key index exceeds the query index.
                    reconfig_data_format(cb_attn_mask, cb_zero_tile);
                    add_init(cb_attn_mask, cb_zero_tile, /* acc_to_dest */ true);
                    add_tiles(cb_attn_mask, cb_zero_tile, kMaskTriangle, 0, a);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    pack_rounding(true);
                    pack_tile</* out_of_order */ true>(b, cb_scores, b * Bt + a);
                    pack_rounding(false);
                    pack_tile</* out_of_order */ true>(b, cb_scores_exact, b * Bt + a);
                }
                tile_regs_release();
            }
            cb_push_back(cb_scores, score_tiles);
            cb_push_back(cb_scores_exact, score_tiles);
            cb_wait_front(cb_scores, score_tiles);
            cb_wait_front(cb_scores_exact, score_tiles);
        }

        // ---- 2. The block maximum per query tile, row layout, to scratch.
        {
            DeviceZoneScopedN("MAX");
            constexpr uint32_t kMaxReg = 0, kRowMaskReg = 1;
            cb_reserve_back(cb_block_max, Bt);
            for (uint32_t a = 0; a < Bt; ++a) {
                const uint32_t live = n_live(a);
                tile_regs_acquire();
                reconfig_data_format(cb_scores, cb_reduce_scaler);
                reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_scores, cb_reduce_scaler, cb_block_max);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_scores, cb_reduce_scaler, b * Bt + a, 0, kMaxReg);
                }
                reduce_uninit();
                // Row 0 only: the reduction leaves partial results in the other
                // rows (measured). mask_tile takes its mask in the register above.
                reconfig_data_format_srca(cb_scores, cb_ones_row);
                copy_init(cb_ones_row);
                copy_tile(cb_ones_row, 0, kRowMaskReg);
                mask_tile_init();
                mask_tile(kMaxReg, kRowMaskReg);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_block_max);
                pack_tile</* out_of_order */ true>(kMaxReg, cb_block_max, a);
                tile_regs_release();
            }
            cb_push_back(cb_block_max, Bt);
            cb_wait_front(cb_block_max, Bt);
        }

        // ---- 2b. m_new = max(m_old, colmax S^T) as a full tile, and
        // r = exp(a (m_old - m_new)), exact.
        {
            DeviceZoneScopedN("RESCALE");
            constexpr uint32_t kNewReg = 0, kOldReg = 1, kDiffReg = 2;
            if (!fresh) {
                cb_reserve_back(cb_rescale, Bt);
            }
            for (uint32_t a = 0; a < Bt; ++a) {
                tile_regs_acquire();
                broadcast_rows(cb_block_max, a, kNewReg);  // the block maximum down every row
                if (!fresh) {
                    reconfig_data_format_srca(cb_block_max, cb_max_seed);
                    copy_init(cb_max_seed);
                    copy_tile(cb_max_seed, a, kOldReg);
                    copy_tile(cb_max_seed, a, kDiffReg);
                    binary_max_tile_init();
                    binary_max_tile(kNewReg, kOldReg, kNewReg);
                    sub_binary_tile_init();
                    sub_binary_tile(kDiffReg, kNewReg, kDiffReg);
                    exp_scaled(kDiffReg);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_max_out);
                pack_tile</* out_of_order */ true>(kNewReg, cb_max_out, a);
                if (!fresh) {
                    pack_reconfig_data_format(cb_rescale);
                    pack_tile</* out_of_order */ true>(kDiffReg, cb_rescale, a);
                }
                tile_regs_release();
            }
            if (!fresh) {
                cb_push_back(cb_rescale, Bt);
                cb_wait_front(cb_rescale, Bt);
            }
            // m_new is read back below through the seed view of the same memory.
            unpacker_fence();
        }

        // ---- 3. P^T = exp(a (S^T - m_new)), the subtraction exact on the SFPU.
        {
            DeviceZoneScopedN("PROBS");
            cb_reserve_back(cb_probs, score_tiles);
            constexpr uint32_t kMaxReg = Bt;  // above the Bt score tiles
            for (uint32_t a = 0; a < Bt; ++a) {
                const uint32_t live = n_live(a);
                tile_regs_acquire();
                reconfig_data_format_srca(cb_scores, cb_max_seed);
                copy_init(cb_max_seed);
                copy_tile(cb_max_seed, a, kMaxReg);
                reconfig_data_format_srca(cb_max_seed, cb_scores_exact);
                copy_init(cb_scores_exact);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    copy_tile(cb_scores_exact, b * Bt + a, b);
                }
                sub_binary_tile_init();
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    sub_binary_tile(b, kMaxReg, b);
                }
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    exp_scaled(b);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_probs);
                pack_rounding(true);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    pack_tile</* out_of_order */ true>(b, cb_probs, b * Bt + a);
                }
                pack_rounding(false);
                tile_regs_release();
            }
            cb_push_back(cb_probs, score_tiles);
            cb_wait_front(cb_probs, score_tiles);
        }

        // ---- 4. l_new = r l_old + colsum P^T.
        {
            DeviceZoneScopedN("SUM");
            constexpr uint32_t kSumReg = 0, kOldReg = 1, kRReg = 2;
            for (uint32_t a = 0; a < Bt; ++a) {
                const uint32_t live = n_live(a);
                tile_regs_acquire();
                reconfig_data_format(cb_probs, cb_reduce_scaler);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_probs, cb_reduce_scaler, cb_sum_out);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_probs, cb_reduce_scaler, b * Bt + a, 0, kSumReg);
                }
                reduce_uninit();
                reconfig_data_format_srca(cb_probs, cb_ones_row);
                copy_init(cb_ones_row);
                copy_tile(cb_ones_row, 0, kSumReg + 1u);
                mask_tile_init();
                mask_tile(kSumReg, kSumReg + 1u);
                if (!fresh) {
                    // r l_old + l_blk in row 0, all exact.
                    reconfig_data_format_srca(cb_ones_row, cb_rescale);
                    copy_init(cb_rescale);
                    copy_tile(cb_rescale, a, kRReg);
                    reconfig_data_format_srca(cb_rescale, cb_sum_seed);
                    copy_init(cb_sum_seed);
                    copy_tile(cb_sum_seed, a, kOldReg);
                    mul_binary_tile_init();
                    mul_binary_tile(kOldReg, kRReg, kOldReg);
                    add_binary_tile_init();
                    add_binary_tile(kSumReg, kOldReg, kSumReg);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_sum_out);
                pack_tile</* out_of_order */ true>(kSumReg, cb_sum_out, a);
                tile_regs_release();
            }
        }

        // ---- 5. O^T <- r O^T + V^T P^T, on the packet where it lies.
        {
            DeviceZoneScopedN("UPDATE-O");
            constexpr uint32_t kRReg = qWt;  // above the qWt output tiles of a query tile
            for (uint32_t a = 0; a < Bt; ++a) {
                const uint32_t live = n_live(a);
                tile_regs_acquire();
                if (!fresh) {
                    reconfig_data_format_srca(cb_probs, cb_rescale);
                    copy_init(cb_rescale);
                    copy_tile(cb_rescale, a, kRReg);
                    reconfig_data_format_srca(cb_rescale, cb_out_seed);
                    copy_init(cb_out_seed);
                    for (uint32_t k = 0; k < qWt; ++k) {
                        copy_tile(cb_out_seed, a * qWt + k, k);
                    }
                    mul_binary_tile_init();
                    for (uint32_t k = 0; k < qWt; ++k) {
                        mul_binary_tile(k, kRReg, k);
                    }
                }
                // V^T is the first operand (bf16, SrcB), P^T the second (Float32, SrcA).
                reconfig_data_format(cb_probs, cb_value_t);
                mm_init<kFidO>(cb_value_t, cb_probs, /* transpose */ 0);
                for (uint32_t k = 0; k < qWt; ++k) {
                    for (uint32_t b = 0; b < Bt; ++b) {
                        if (b >= live) {
                            break;
                        }
                        mm_tiles<kFidO>(cb_value_t, cb_probs, k * Bt + b, b * Bt + a, k);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_out_out);
                for (uint32_t k = 0; k < qWt; ++k) {
                    pack_tile</* out_of_order */ true>(k, cb_out_out, a * qWt + k);
                }
                tile_regs_release();
            }
        }

        // ---- 6. The finished row: O = (O^T / l)^T in bfloat16, lse = a m + ln l.
        //
        // A dest-register transpose clears the source registers when it is
        // done, so an operand the unpacker has already delivered for the next
        // copy is lost (see the backward kernel; measured here: the lse of
        // every query tile but the first came out without its m). The
        // transposes are therefore kept in loops of their own -- copy,
        // transpose, pack -- with the unpacker fenced after every tile, which
        // holds the next copy's unpack behind this tile's pack.
        if (final) {
            DeviceZoneScopedN("FINISH");
            // The sums and the accumulator packed above are read back through
            // the views of the same memory.
            unpacker_fence();
            // O^T / l, back onto the packet (dead after this).
            {
                constexpr uint32_t kInvReg = qWt;
                for (uint32_t a = 0; a < Bt; ++a) {
                    tile_regs_acquire();
                    broadcast_rows(cb_sum_plain, a, kInvReg);
                    recip_tile_init</* legacy_compat */ false>();
                    recip_tile</* legacy_compat */ false>(kInvReg);
                    reconfig_data_format_srca(cb_sum_plain, cb_out_seed);
                    copy_init(cb_out_seed);
                    for (uint32_t k = 0; k < qWt; ++k) {
                        copy_tile(cb_out_seed, a * qWt + k, k);
                    }
                    mul_binary_tile_init();
                    for (uint32_t k = 0; k < qWt; ++k) {
                        mul_binary_tile(k, kInvReg, k);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_out_out);
                    for (uint32_t k = 0; k < qWt; ++k) {
                        pack_tile</* out_of_order */ true>(k, cb_out_out, a * qWt + k);
                    }
                    tile_regs_release();
                }
            }
            unpacker_fence();
            // Every tile transposed within itself, into the bfloat16 output.
            cb_reserve_back(cb_output, Bt * qWt);
            pack_reconfig_data_format(cb_output);
            reconfig_data_format_srca(cb_out_seed, cb_out_seed);
            for (uint32_t t_idx = 0; t_idx < Bt * qWt; ++t_idx) {
                tile_regs_acquire();
                copy_init(cb_out_seed);
                copy_tile(cb_out_seed, t_idx, 0);
                transpose_dest_init</* is_32bit */ true>(cb_out_seed);
                transpose_dest</* is_32bit */ true>(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile</* out_of_order */ true>(0, cb_output, t_idx);
                tile_regs_release();
                unpacker_fence();
            }
            // lse = a m + ln l, row layout, kept to row 0, then transposed into
            // column layout: the value per row in column 0, zeros elsewhere.
            cb_reserve_back(cb_lse, Bt);
            constexpr uint32_t kLseReg = 0, kMaskReg = 1, kLogReg = 2;
            for (uint32_t a = 0; a < Bt; ++a) {
                tile_regs_acquire();
                // The conditional reconfig must name what SrcA actually holds:
                // the O^T tiles before the first query tile, the bfloat16 row
                // mask after every one (measured: naming the wrong one left
                // SrcA in bfloat16 and unpacked m as such).
                reconfig_data_format_srca(a == 0u ? cb_out_seed : cb_ones_row, cb_max_seed);
                copy_init(cb_max_seed);
                copy_tile(cb_max_seed, a, kLseReg);
                binop_with_scalar_tile_init();
                mul_unary_tile(kLseReg, scaler_bits);
                reconfig_data_format_srca(cb_max_seed, cb_sum_seed);
                copy_init(cb_sum_seed);
                copy_tile(cb_sum_seed, a, kLogReg);
                log_tile_init</* fast_and_approx */ false>();
                log_tile</* fast_and_approx */ false>(kLogReg);
                add_binary_tile_init();
                add_binary_tile(kLseReg, kLogReg, kLseReg);
                reconfig_data_format_srca(cb_sum_seed, cb_ones_row);
                copy_init(cb_ones_row);
                copy_tile(cb_ones_row, 0, kMaskReg);
                mask_tile_init();
                mask_tile(kLseReg, kMaskReg);
                transpose_dest_init</* is_32bit */ true>(cb_ones_row);
                transpose_dest</* is_32bit */ true>(kLseReg);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_lse);
                pack_tile</* out_of_order */ true>(kLseReg, cb_lse, a);
                tile_regs_release();
                unpacker_fence();
            }
            cb_push_back(cb_output, Bt * qWt);
            cb_push_back(cb_lse, Bt);
        }

        // ---- hand the updated state to the reader, pop the slot, release it.
        cb_push_back(cb_max_out, Bt);
        cb_push_back(cb_sum_out, Bt);
        cb_push_back(cb_out_out, Bt * qWt);
        {
            DeviceZoneScopedN("T-POPS");
            cb_pop_front(cb_query, Bt * qWt);
            cb_pop_front(cb_max_seed, Bt);
            cb_pop_front(cb_sum_seed, Bt);
            cb_pop_front(cb_sum_plain, Bt);
            cb_pop_front(cb_out_seed, Bt * qWt);
            cb_pop_front(cb_scores, score_tiles);
            cb_pop_front(cb_scores_exact, score_tiles);
            cb_pop_front(cb_probs, score_tiles);
            cb_pop_front(cb_block_max, Bt);
            if (!fresh) {
                cb_pop_front(cb_rescale, Bt);
            }
        }
        cb_reserve_back(cb_slot_release, 1);
        cb_push_back(cb_slot_release, 1);
    }
    }  // slices
}
