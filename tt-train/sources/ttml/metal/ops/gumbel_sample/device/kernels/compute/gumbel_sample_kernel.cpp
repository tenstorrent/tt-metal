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

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"  // tile_regs_*
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/reg_api.h"
#include "api/compute/sfpu_binary_bcast.h"  // sfpu_sub_bcast_row, the mask apply
#include "api/compute/tile_move_copy.h"
#include "gumbel_sfpu.h"  // gumbel_score_tile, the fused noise/scale/add pass
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"  // pack_and_push_block

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
// first padding column). The mask must therefore be broadcast down the token rows. That is done as
// copy_tile + sfpu_sub_bcast_row: the mask tile lands in the operand slot VERBATIM via the plain A2D
// copy (ELWADD-based under fp32 dest, safe on every arch -- and the only bridge from an L1 CB into
// the DST slots the SFPU can address), and one SFPU pass both broadcasts its row 0 down the 32 token
// rows and subtracts, entirely inside DST.
//
// Deliberately NOT unary_bcast<ROW> (broadcasting as the tile is unpacked, then subtracting), even
// though that looks like the natural fit: with fp32_dest_acc_en (always on here) the 16-bit ROW
// broadcast is a MOVB2D MOP, and WORMHOLE silicon addresses the dest as 16-bit rows unless SrcA's
// ALU format is TF32 -- the mask lands in the wrong dest rows and the subtract reads zeros, silently
// un-masking every padded column (Blackhole auto-promotes SrcA under fp32 accumulation and is
// unaffected; tt-llk's test_bcast.py skips this combination on WH as broken, and binary_ng falls
// back to an SFPU broadcast for the same reason). The SFPU apply measured no slower than
// unary_bcast on Blackhole and is the only one of the two that works on Wormhole.

// temperature == 0 is greedy decoding: no noise, no scaling, just argmax over the (masked) logits.
// It still runs through this kernel rather than a separate ttnn::argmax, so the greedy path gets the
// same fusion win -- the score tiles stream straight into the writer's running argmax and the
// [B, 1, tokens, V] untilized copy that ttnn::argmax would need never exists.
constexpr bool do_gumbel_noise = get_compile_time_arg_val(3) != 0;

// DST slots. fp32_dest_acc_en is on, so the noise keeps full FP32 precision through the two logs --
// a bf16 round trip near U ~ 1 would quantize -log(-log(U)) catastrophically (bf16 has 8 mantissa
// bits, and the interesting part of the upper tail lives in the last few ULPs below 1.0).
//
// fp32 also halves DST: a half-sync acquire window holds FOUR fp32 tiles (identical on Wormhole and
// Blackhole). What divides that window is whether each score needs an OPERAND tile beside it: the
// noise path pairs every score with a logits tile (the mask, when present, reuses that slot once
// the score is computed), and the greedy+mask path pairs every score with its mask tile
// -- two slots per tile, so a batch of two: scores in slots 0..1, operands in 2..3. Pure greedy
// (no noise, no mask) copies the logits straight into the score slots and pairs them with nothing,
// so all four slots hold scores: a batch of four.
//
// The batch exists to amortize per-op setup: rand_tile is the only op that clobbers the
// programmable const LREGs holding the log constants (see gumbel_sfpu.h), so drawing the whole
// batch's noise first lets ONE gumbel_score_tile_init cover every score pass in the batch, and the
// per-op inits and acquire/commit/pack handshakes drop to one per batch too.
//
// A scored batch of 4 was considered and rejected: it needs 8 fp32 tiles live at once, which only
// SyncFull dest mode provides -- and SyncFull serializes math against pack, giving up the overlap
// that half-sync's 4/4 split buys to shave already-sub-percent setup cost.
constexpr bool use_operand_slots = do_gumbel_noise || do_logits_mask;
constexpr uint32_t dst_batch = use_operand_slots ? 2U : 4U;
constexpr uint32_t score_base = 0U;
// One past the scores; meaningful only when use_operand_slots (both consumers sit behind
// `if constexpr` on the flags that make it true).
constexpr uint32_t operand_base = score_base + dst_batch;
static_assert(dst_batch * (use_operand_slots ? 2U : 1U) <= 4U, "DST batch overflows the 4-tile fp32 half-sync window");

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

            // The tail batch of a block may be a single tile. The slot layout does not shift with
            // it: operands stay at operand_base, so the score pass keeps ONE compile-time DST
            // offset in both cases (a slot simply goes unused on a short batch).
            for (uint32_t k = 0U; k < current; k += dst_batch) {
                const uint32_t batch = (current - k < dst_batch) ? (current - k) : dst_batch;

                tile_regs_acquire();

                if constexpr (do_gumbel_noise) {
                    // ---- pass 1: U ~ Uniform[from, from + scale], one draw per score slot ----
                    // rand_tile stays a standalone pass: rand owns the mutable LREG file (LREG0-7),
                    // programs const LREG12/13 on Wormhole, and a 16-instruction replay row;
                    // nothing may interleave with it (see ckernel_sfpu_rand.h). The batch's draws
                    // all run up front IN TILE ORDER, so the LFSR is consumed in exactly the
                    // sequence a per-tile loop would use -- the noise stream is bit-identical.
                    for (uint32_t i = 0U; i < batch; ++i) {
                        rand_tile(score_base + i, rand_from_bits, rand_scale_bits);
                    }

                    copy_tile_init(cb_logits);
                    for (uint32_t i = 0U; i < batch; ++i) {
                        copy_tile(cb_logits, k + i, operand_base + i);
                    }

                    // ---- pass 2: score = logits * (1/T) + (-log(-log(U))), one SFPI sweep ----
                    // The scaling is applied to the LOGITS, never to the noise: score = logits/T + g.
                    // Scaling the noise instead would invert the temperature's meaning entirely.
                    // ONE init covers the whole batch: it reprograms what rand_tile left behind
                    // (counters, SFPU config, and the programmable const LREGs holding the log
                    // constants -- Wormhole's rand_tile overwrites two of those slots on every
                    // call), the batch's rand draws are all behind us, and nothing else touches
                    // those LREGs before the last score pass. No replay slots are used, so rand's
                    // replayed row is untouched.
                    gumbel_score_tile_init();
                    // The offset template arg is unsigned, so a reversed layout would wrap into a
                    // silent out-of-bounds DST read rather than a negative offset.
                    static_assert(operand_base > score_base, "logits tiles must sit above the noise tiles in DST");
                    for (uint32_t i = 0U; i < batch; ++i) {
                        gumbel_score_tile<operand_base - score_base>(score_base + i, inv_temperature_bits);
                    }
                } else {
                    // ---- greedy (temperature == 0): the logits ARE the scores ----
                    // No 1/temperature scaling: it would be a division by zero on the host, and a
                    // positive scale factor cannot change an argmax anyway.
                    copy_tile_init(cb_logits);
                    for (uint32_t i = 0U; i < batch; ++i) {
                        copy_tile(cb_logits, k + i, score_base + i);
                    }
                }

                // ---- optional additive padding mask, broadcast down the token rows ----
                // The logits operand slots are dead once the scores exist; the mask reuses them:
                // the raw mask tile lands there with a plain copy (same A2D path the logits ride;
                // the mask CB shares their format, so no reconfig), then one SFPU pass broadcasts
                // its row 0 down all 32 rows AND subtracts, in place on the score tile. Rows 1..31
                // of the copied mask tile are zero-fill the pass never reads. See the mask note at
                // the top of this file for why this is NOT a unary_bcast. sfpu_sub_bcast_row_init
                // is re-run per batch because rand_tile and gumbel_score_tile_init recycle SFPU
                // state between batches; the row path records no replay rows, so rand's replayed
                // row (see the rand note above) stays untouched.
                if constexpr (do_logits_mask) {
                    copy_tile_init(cb_mask);
                    for (uint32_t i = 0U; i < batch; ++i) {
                        copy_tile(cb_mask, k + i, operand_base + i);
                    }
                    sfpu_sub_bcast_row_init();
                    for (uint32_t i = 0U; i < batch; ++i) {
                        sfpu_sub_bcast_row(score_base + i, operand_base + i);
                    }
                }

                tile_regs_commit();
                pack_and_push_block(cb_scores, batch);
            }

            cb_pop_front(cb_logits, current);
            if constexpr (do_logits_mask) {
                cb_pop_front(cb_mask, current);
            }
        }
    }
}
