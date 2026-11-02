// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused Gumbel-max sampling, compute half. Per vocab tile, entirely inside DST:
//
//     score = logits * (1 / temperature) + (-log(-log(U)))  [ - padding_mask ]
//
// rand_tile draws the uniforms, one SFPI sweep (gumbel_sfpu.h) fuses the rest, and the score tile
// goes straight to the writer's running argmax -- nothing [B, 1, tokens, V]-sized touches DRAM.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"  // tile_regs_*
#include "api/compute/compute_kernel_hw_startup.h"
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

// The mask is one [1, 1, 1, V] row applied to every token row, so it must be broadcast down the
// tile. That is done as copy_tile + sfpu_sub_bcast_row (broadcast and subtract in one SFPU pass),
// NOT as unary_bcast<ROW>: under fp32 dest the 16-bit ROW broadcast lands in the wrong dest rows
// on Wormhole silicon and silently un-masks every padded column.

// temperature == 0 is greedy decoding: the noise and scaling are compiled out and the kernel
// reduces to a fused argmax over the (masked) logits.
constexpr bool do_gumbel_noise = get_compile_time_arg_val(3) != 0;

// DST layout. fp32_dest_acc_en is always on (a bf16 round trip near U ~ 1 would quantize
// -log(-log(U)) catastrophically), so a half-sync acquire window holds four fp32 tiles. Paths that
// pair each score with an operand tile (noise: logits; greedy+mask: mask) batch two scores in
// slots 0..1 with operands in 2..3; pure greedy uses all four slots as scores. Batching lets one
// gumbel_score_tile_init cover the whole batch after its rand_tile draws (rand clobbers the const
// LREGs the score pass reads).
constexpr bool use_operand_slots = do_gumbel_noise || do_logits_mask;
constexpr uint32_t dst_batch = use_operand_slots ? 2U : 4U;
constexpr uint32_t score_base = 0U;
constexpr uint32_t operand_base = score_base + dst_batch;  // meaningful only when use_operand_slots
static_assert(dst_batch * (use_operand_slots ? 2U : 1U) <= 4U, "DST batch overflows the 4-tile fp32 half-sync window");

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t seed = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t rand_from_bits = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t rand_scale_bits = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t inv_temperature_bits = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t rand_stream_id = get_arg_val<uint32_t>(rt_idx++);

    compute_kernel_hw_startup(cb_logits, cb_scores);
    copy_init(cb_logits);

    // One init per core: the LFSR advances monotonically across all rand_tile calls, and the
    // (device, core) stream id keeps streams reproducible yet disjoint across cores and devices.
    if constexpr (do_gumbel_noise) {
        rand_tile_init(seed, rand_stream_id);
    }

    // Reader, compute and writer derive `current` identically and stay in lockstep.
    for (uint32_t t = 0U; t < num_tiles; t += block_size) {
        const uint32_t remaining = num_tiles - t;
        const uint32_t current = (remaining < block_size) ? remaining : block_size;

        {
            cb_wait_front(cb_logits, current);
            if constexpr (do_logits_mask) {
                cb_wait_front(cb_mask, current);
            }

            // The tail batch may be short; the slot layout does not shift (a slot goes unused).
            for (uint32_t k = 0U; k < current; k += dst_batch) {
                const uint32_t batch = (current - k < dst_batch) ? (current - k) : dst_batch;

                tile_regs_acquire();

                if constexpr (do_gumbel_noise) {
                    // Pass 1: uniform draws, all up front and in tile order -- rand_tile owns the
                    // mutable LREGs and a replay row, so nothing may interleave with it.
                    for (uint32_t i = 0U; i < batch; ++i) {
                        rand_tile(score_base + i, rand_from_bits, rand_scale_bits);
                    }

                    copy_init(cb_logits);
                    for (uint32_t i = 0U; i < batch; ++i) {
                        copy_tile(cb_logits, k + i, operand_base + i);
                    }

                    // Pass 2: score = logits * (1/T) + (-log(-log(U))). The scaling applies to the
                    // LOGITS, never the noise. The init must run after the batch's LAST rand_tile:
                    // rand reprograms the const LREGs holding the log constants.
                    gumbel_score_tile_init();
                    // The offset template arg is unsigned; a reversed layout would wrap out of bounds.
                    static_assert(operand_base > score_base, "logits tiles must sit above the noise tiles in DST");
                    for (uint32_t i = 0U; i < batch; ++i) {
                        gumbel_score_tile<operand_base - score_base>(score_base + i, inv_temperature_bits);
                    }
                } else {
                    // Greedy: the logits ARE the scores (no 1/T; it cannot change an argmax).
                    copy_init(cb_logits);
                    for (uint32_t i = 0U; i < batch; ++i) {
                        copy_tile(cb_logits, k + i, score_base + i);
                    }
                }

                // Optional additive mask, broadcast down the token rows (see the note at the top).
                // The operand slots are dead once the scores exist, so the mask reuses them. The
                // init is re-run per batch: rand/gumbel inits recycle SFPU state between batches.
                if constexpr (do_logits_mask) {
                    copy_init(cb_mask);
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
