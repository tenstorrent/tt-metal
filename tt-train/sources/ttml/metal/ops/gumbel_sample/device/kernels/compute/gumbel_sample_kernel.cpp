// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused Gumbel-max sampling, compute half.
//
// Per vocab tile this kernel does, entirely inside DST, what a composite sample would spell as a
// chain of separate ttnn ops (rand -> log -> neg -> log -> neg -> mul -> add -> sub):
//
//     score = logits * (1 / temperature) + (-log(-log(U)))  [ - padding_mask ]
//
// and hands the score tile straight to the writer, which folds it into a running per-row argmax.
// Nothing [B, 1, tokens, V]-sized is ever written to DRAM.
//
// The noise chain is two SFPU passes over DST: rand_tile draws the raw uniforms, then one op-local
// SFPI sweep (gumbel_sfpu.h) folds both logs, both negations, the temperature scale and the add
// into LREGs, storing each score datum exactly once.

#include <cstdint>

#include "api/compute/bcast.h"  // unary_bcast, for the [1, V] padding mask
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"  // pack_tile, tile_regs_*
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "gumbel_sfpu.h"  // gumbel_score_tile, the fused noise/scale/add pass

constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

constexpr auto cb_logits = tt::CBIndex::c_0;
constexpr auto cb_mask = tt::CBIndex::c_1;
constexpr auto cb_scores = tt::CBIndex::c_2;

constexpr bool do_logits_mask = get_compile_time_arg_val(2) != 0;

// Every caller builds the padding mask as [1, 1, 1, V] -- one row, reused for every token -- because
// which vocab columns are padding does not depend on the token position. In TILE layout that single
// logical row lives in row 0 of each tile with rows 1..31 zero-filled, so a plain tile-for-tile
// subtract would mask ONLY token row 0 and leave every other row unmasked (it then argmaxes onto the
// first padding column). This requires the mask to be broadcasted; here the broadcast has to
// be explicit, via unary_bcast<ROW> which splats row 0 down all 32 rows as the tile lands in DST.

// temperature == 0 is greedy decoding: no noise, no scaling, just argmax over the (masked) logits.
// It still runs through this kernel rather than a separate ttnn::argmax, so the greedy path gets the
// same fusion win -- the score tiles stream straight into the writer's running argmax and the
// [B, 1, tokens, V] untilized copy that ttnn::argmax would need never exists.
constexpr bool do_gumbel_noise = get_compile_time_arg_val(3) != 0;

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

    // The packer only ever emits cb_scores, so its format is loop-invariant.
    pack_reconfig_data_format(cb_scores);

    // One init for the whole core: the LFSR then advances monotonically across every rand_tile call
    // below. Combined with the (device, core) specific stream id this makes the noise reproducible
    // for a given seed and work split, and disjoint across cores and data-parallel devices.
    if constexpr (do_gumbel_noise) {
        rand_tile_init(seed, rand_stream_id);
    }

    // Work unit is one TILE, so a core's run is arbitrary in length and the last block may be
    // partial. Reader, compute and writer all derive `current` identically and stay in lockstep.
    for (uint32_t t = 0U; t < num_tiles; t += block_size) {
        const uint32_t remaining = num_tiles - t;
        const uint32_t current = (remaining < block_size) ? remaining : block_size;

        {
            cb_wait_front(cb_logits, current);
            if constexpr (do_logits_mask) {
                cb_wait_front(cb_mask, current);
            }

            for (uint32_t block_idx = 0U; block_idx < current; ++block_idx) {
                cb_reserve_back(cb_scores, onetile);
                tile_regs_acquire();

                if constexpr (do_gumbel_noise) {
                    // ---- pass 1: U ~ Uniform[from, from + scale] into the score slot ----
                    // rand_tile stays a standalone pass: the PRNG's per-tile draw order defines the
                    // reproducible noise stream, and rand owns the mutable LREG file (LREG0-7),
                    // programs const LREG12/13 on Wormhole, and a 16-instruction replay row;
                    // nothing may interleave with it (see ckernel_sfpu_rand.h).
                    rand_tile(score_reg, rand_from_bits, rand_scale_bits);

                    copy_tile_init(cb_logits);
                    copy_tile(cb_logits, block_idx, operand_reg);

                    // ---- pass 2: score = logits * (1/T) + (-log(-log(U))), one SFPI sweep ----
                    // The scaling is applied to the LOGITS, never to the noise: score = logits/T + g.
                    // Scaling the noise instead would invert the temperature's meaning entirely.
                    // The init reprograms what rand_tile left behind: counters, SFPU config,
                    // and the programmable const LREGs holding the log constants (Wormhole's
                    // rand_tile overwrites two of those slots on every call). The pass uses no
                    // replay slots, so rand's replayed row is untouched.
                    gumbel_score_tile_init();
                    // The offset template arg is unsigned, so a reversed pair would wrap into a
                    // silent out-of-bounds DST read rather than a negative offset.
                    static_assert(operand_reg > score_reg, "logits tile must sit above the noise tile in DST");
                    gumbel_score_tile<operand_reg - score_reg>(score_reg, inv_temperature_bits);
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
                pack_tile(score_reg, cb_scores);
                tile_regs_release();

                cb_push_back(cb_scores, onetile);
            }

            cb_pop_front(cb_logits, current);
            if constexpr (do_logits_mask) {
                cb_pop_front(cb_mask, current);
            }
        }
    }
}
