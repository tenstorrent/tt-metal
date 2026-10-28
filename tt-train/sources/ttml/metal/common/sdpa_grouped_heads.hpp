// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <functional>

namespace ttml::metal::sdpa {

/**
 * Decide how many Q heads of one GQA group a single SDPA work item covers ("heads_per_item").
 *
 * With heads_per_item == 1 every Q row of every head streams its own causal K/V prefix (the
 * per-head row layout), so the heads_per_group Q heads sharing a K/V head re-read the same K/V
 * heads_per_group times. At long seq_len that traffic (~ heads_per_group * St^2 / 2 tile rows
 * per K/V head) makes the kernels DRAM-bound.
 *
 * With heads_per_item == d the work item is (batch, kv_head, sub-group of d heads, q_row): the
 * reader stages d Q rows (and, in backward, dO/O/lse rows) once and streams each K/V chunk once
 * for all of them, cutting K/V DRAM traffic by d. The price is d x coarser work granularity.
 *
 * Cost model (fitted on Blackhole p150, D=256, 6 Q heads / 1 K/V head):
 *   T(d) ~ ceil(items_d / cores) * (1 + d * kHeadCostPerStream)
 * i.e. a core's time is dominated by streaming the K/V prefix of each of its items, plus a
 * small per-head compute increment on top of each stream. Measured fw times (ms):
 *   S=8k: d=6 7.4 | d=3 8.5 | d=2 10.0 | d=1 17.0
 *   S=4k: d=6 1.9 | d=3 2.7 | d=2  2.6 | d=1  4.6   (d=6 runs on only 64 of 110 cores)
 *   S=2k: d=6 1.0 | d=3 0.6 | d=2  0.7 | d=1  1.2   (d=6 runs on only 32 cores -> loses)
 * We therefore minimise T(d) over the divisors of heads_per_group, but never drop below half
 * the grid (the 2k/d=6 case), and require the per-head L1 state to fit.
 *
 * Environment overrides (debugging / A-B measurements):
 *   TTML_SDPA_DISABLE_GROUPED_HEADS=1  -> always 1 (legacy per-head rows)
 *   TTML_SDPA_HEADS_PER_ITEM=<d>       -> force d (must divide heads_per_group and fit in L1)
 *
 * @param heads_per_group  Q heads per K/V head.
 * @param base_items       Balanced work items when the whole group is one item
 *                         (batch * kv_heads * pairs_per_seq).
 * @param num_cores        Cores available to the op.
 * @param fits_in_l1       Predicate: do the CBs for a given heads_per_item fit in L1.
 */
inline uint32_t pick_heads_per_item(
    uint32_t heads_per_group,
    uint64_t base_items,
    uint32_t num_cores,
    const std::function<bool(uint32_t)>& fits_in_l1) {
    if (heads_per_group <= 1U || num_cores == 0U) {
        return 1U;
    }
    if (const char* disabled = std::getenv("TTML_SDPA_DISABLE_GROUPED_HEADS");
        disabled != nullptr && disabled[0] == '1') {
        return 1U;
    }
    if (const char* forced = std::getenv("TTML_SDPA_HEADS_PER_ITEM"); forced != nullptr) {
        const auto d = static_cast<uint32_t>(std::strtoul(forced, nullptr, 10));
        if (d >= 1U && heads_per_group % d == 0U && (d == 1U || fits_in_l1(d))) {
            return d;
        }
    }

    // Per-head compute increment relative to one K/V stream, in tenths (0.1 per head).
    constexpr uint64_t kStreamCostTenths = 10U;
    constexpr uint64_t kHeadCostTenths = 1U;

    uint32_t best_d = 1U;
    uint64_t best_cost = UINT64_MAX;
    for (uint32_t d = 1U; d <= heads_per_group; ++d) {
        if (heads_per_group % d != 0U) {
            continue;
        }
        const uint64_t items = base_items * (heads_per_group / d);
        if (d > 1U && (items * 2U < num_cores || !fits_in_l1(d))) {
            continue;
        }
        const uint64_t items_per_core = (items + num_cores - 1U) / num_cores;
        const uint64_t cost = items_per_core * (kStreamCostTenths + d * kHeadCostTenths);
        if (cost < best_cost) {
            best_cost = cost;
            best_d = d;
        }
    }
    return best_d;
}

/**
 * Decide how many consecutive K/V tile-rows one dK/dV backward work item covers
 * ("kv_rows_per_item"). This is the mirror image of pick_heads_per_item: the KV kernel already
 * loops all Q heads of the group inside one K/V row, so its redundant traffic is the causal
 * Q / dO / lse / u suffix that is re-streamed for every K/V row. Holding R rows (and R dK/dV
 * accumulators) resident streams that suffix once per R rows, cutting traffic ~R x.
 *
 * Unlike the forward / dQ kernels, the (Q row, K row) step here is heavy (four tile-matmul
 * passes plus the softmax-backward chain), so the kernel flips from DRAM-bound to
 * compute-bound already at R=2 and the two costs do not add up - the slower of the two wins:
 *   T(R) ~ max( items_R * kStreamCost / cores,              // shared DRAM bandwidth
 *               ceil(items_R / cores) * R * kRowCost )       // busiest core's compute
 * Fitted on Blackhole p150 (D=256, 6 Q heads / 1 K/V head; dK/dV time in ms):
 *   S=8k: R=1 ~25 | R=2 ~13 | R=4 ~22 | R=8 ~44     (R>=4 leaves most of the grid idle)
 *   S=4k: R=1 ~ 7 | R=2 ~ 6.5
 *   S=2k: all R within noise
 * Candidates are powers of two that divide St with St/R even (so balanced pairs still exist).
 *
 * Environment overrides:
 *   TTML_SDPA_DISABLE_GROUPED_HEADS=1  -> always 1
 *   TTML_SDPA_KV_ROWS_PER_ITEM=<R>     -> force R (must divide St, St/R even, fit in L1)
 *
 * @param St          Sequence length in tiles.
 * @param base_items  Balanced work items with one K/V row per item (batch * kv_heads * St / 2).
 * @param num_cores   Cores available to the op.
 * @param fits_in_l1  Predicate: do the CBs for a given kv_rows_per_item fit in L1.
 */
inline uint32_t pick_kv_rows_per_item(
    uint32_t St, uint64_t base_items, uint32_t num_cores, const std::function<bool(uint32_t)>& fits_in_l1) {
    constexpr uint32_t kMaxRows = 8U;
    const auto is_valid = [&](uint32_t r) { return r >= 1U && St % r == 0U && ((St / r) % 2U == 0U); };

    if (num_cores == 0U || St < 4U) {
        return 1U;
    }
    if (const char* disabled = std::getenv("TTML_SDPA_DISABLE_GROUPED_HEADS");
        disabled != nullptr && disabled[0] == '1') {
        return 1U;
    }
    if (const char* forced = std::getenv("TTML_SDPA_KV_ROWS_PER_ITEM"); forced != nullptr) {
        const auto r = static_cast<uint32_t>(std::strtoul(forced, nullptr, 10));
        if (is_valid(r) && (r == 1U || fits_in_l1(r))) {
            return r;
        }
    }

    // Cost of streaming one item's Q-side suffix through the shared DRAM pipe vs. computing one
    // K/V row against it on a single core, in common (arbitrary) units; see the header comment.
    constexpr uint64_t kStreamCost = 10U;
    constexpr uint64_t kRowCost = 3U;

    uint32_t best_r = 1U;
    uint64_t best_cost = UINT64_MAX;
    for (uint32_t r = 1U; r <= kMaxRows; r *= 2U) {
        if (!is_valid(r)) {
            continue;
        }
        const uint64_t items = base_items / r;
        if (r > 1U && (items == 0U || !fits_in_l1(r))) {
            continue;
        }
        const uint64_t items_per_core = (items + num_cores - 1U) / num_cores;
        const uint64_t dram_cost = (items * kStreamCost + num_cores - 1U) / num_cores;
        const uint64_t compute_cost = items_per_core * r * kRowCost;
        const uint64_t cost = dram_cost > compute_cost ? dram_cost : compute_cost;
        if (cost < best_cost) {
            best_cost = cost;
            best_r = r;
        }
    }
    return best_r;
}

}  // namespace ttml::metal::sdpa
