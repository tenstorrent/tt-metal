// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused Gumbel-max sampling, compute half.
//
// Per vocab tile this kernel does, entirely inside DST, what ttnn_fixed::sample spells as five
// separate ttnn ops (rand -> log -> neg -> log -> neg -> mul -> add -> sub):
//
//     score = logits * (1 / temperature) + (-log(-log(U)))  [ - padding_mask ]
//
// and hands the score tile straight to the writer, which folds it into a running per-row argmax.
// Nothing [B, 1, tokens, V]-sized is ever written to DRAM.

#include <cstdint>

#include "api/compute/bcast.h"  // unary_bcast, for the [1, V] padding mask
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"  // log_tile, pack_tile, tile_regs_*
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"  // mul_unary_tile
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

constexpr auto cb_logits = tt::CBIndex::c_0;
constexpr auto cb_mask = tt::CBIndex::c_1;
constexpr auto cb_scores = tt::CBIndex::c_2;

#ifdef DO_LOGITS_MASK
constexpr bool do_logits_mask = true;
#else
constexpr bool do_logits_mask = false;
#endif

// Every caller builds the padding mask as [1, 1, 1, V] -- one row, reused for every token -- because
// which vocab columns are padding does not depend on the token position. In TILE layout that single
// logical row lives in row 0 of each tile with rows 1..31 zero-filled, so a plain tile-for-tile
// subtract would mask ONLY token row 0 and leave every other row unmasked (it then argmaxes onto the
// first padding column). ttnn::subtract used to hide this by broadcasting; here the broadcast has to
// be explicit, via unary_bcast<ROW> which splats row 0 down all 32 rows as the tile lands in DST.

// temperature == 0 is greedy decoding: no noise, no scaling, just argmax over the (masked) logits.
// It still runs through this kernel rather than a separate ttnn::argmax, so the greedy path gets the
// same fusion win -- the score tiles stream straight into the writer's running argmax and the
// [B, 1, tokens, V] untilized copy that ttnn::argmax would need never exists.
#ifdef DO_GUMBEL_NOISE
constexpr bool do_gumbel_noise = true;
#else
constexpr bool do_gumbel_noise = false;
#endif

constexpr uint32_t onetile = 1U;

// DST slots. fp32_dest_acc_en is on, so the noise keeps full FP32 precision through the two logs --
// a bf16 round trip near U ~ 1 would quantize -log(-log(U)) catastrophically (bf16 has 8 mantissa
// bits, and the interesting part of the upper tail lives in the last few ULPs below 1.0).
constexpr uint32_t score_reg = 0U;
constexpr uint32_t operand_reg = 1U;

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t seed = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t rand_from_bits = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t rand_scale_bits = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t inv_temperature_bits = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t rand_stream_id = get_arg_val<uint32_t>(rt_idx++);

    compute_kernel_hw_startup(cb_logits, cb_scores);
    init_sfpu(cb_logits, cb_scores);

    // One init for the whole core: the LFSR then advances monotonically across every rand_tile call
    // below. Combined with the (device, core) specific stream id this makes the noise reproducible
    // for a given seed and work split, and disjoint across cores and data-parallel devices.
    if constexpr (do_gumbel_noise) {
        rand_tile_init(seed, rand_stream_id);
    }

    for (uint32_t row = 0U; row < num_rows_per_core; ++row) {
        // block_size always divides Wt (see get_block_size), so no tail handling is needed.
        for (uint32_t col = 0U; col < Wt;) {
            cb_wait_front(cb_logits, block_size);
            if constexpr (do_logits_mask) {
                cb_wait_front(cb_mask, block_size);
            }

            for (uint32_t block_idx = 0U; block_idx < block_size; ++block_idx, ++col) {
                cb_reserve_back(cb_scores, onetile);
                tile_regs_acquire();

                if constexpr (do_gumbel_noise) {
                    // ---- Gumbel noise: g = -log(-log(U)), U ~ Uniform[from, from + scale] ----
                    //
                    // Every SFPU step below re-runs its *_init. That is not boilerplate: rand_tile
                    // records replay slots 0-15/0-16 and (on Wormhole) programs LREG12/LREG13, so
                    // any SFPU op that follows it must reprogram what it depends on. Dropping these
                    // inits would leave log/neg reading whatever rand left behind.
                    rand_tile(score_reg, rand_from_bits, rand_scale_bits);
                    log_tile_init();
                    log_tile(score_reg);
                    negative_tile_init();
                    negative_tile(score_reg);
                    log_tile_init();
                    log_tile(score_reg);
                    negative_tile_init();
                    negative_tile(score_reg);

                    // ---- logits / temperature, accumulated onto the noise ----
                    // The scaling is applied to the LOGITS, never to the noise: score = logits/T + g.
                    // Scaling the noise instead would invert the temperature's meaning entirely.
                    copy_tile_init(cb_logits);
                    copy_tile(cb_logits, block_idx, operand_reg);
                    binop_with_scalar_tile_init();
                    mul_unary_tile(operand_reg, inv_temperature_bits);
                    add_binary_tile_init();
                    add_binary_tile(score_reg, operand_reg, score_reg);
                } else {
                    // ---- greedy (temperature == 0): the logits ARE the scores ----
                    // No 1/temperature scaling: it would be a division by zero on the host, and a
                    // positive scale factor cannot change an argmax anyway.
                    copy_tile_init(cb_logits);
                    copy_tile(cb_logits, block_idx, score_reg);
                }

                // ---- optional additive padding mask, broadcast down the token rows ----
                if constexpr (do_logits_mask) {
                    unary_bcast_init<BroadcastType::ROW>(cb_mask);
                    unary_bcast<BroadcastType::ROW>(cb_mask, block_idx, operand_reg);
                    unary_bcast_uninit<BroadcastType::ROW>(cb_mask);
                    sub_binary_tile_init();
                    sub_binary_tile(score_reg, operand_reg, score_reg);
                }

                tile_regs_commit();

                tile_regs_wait();
                pack_reconfig_data_format(cb_scores);
                pack_tile(score_reg, cb_scores);
                tile_regs_release();

                cb_push_back(cb_scores, onetile);
            }

            cb_pop_front(cb_logits, block_size);
            if constexpr (do_logits_mask) {
                cb_pop_front(cb_mask, block_size);
            }
        }
    }
}
