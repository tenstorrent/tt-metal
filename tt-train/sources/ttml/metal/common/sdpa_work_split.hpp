// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "metal/common/sdpa_item_math.hpp"

// How the SDPA programs (forward and backward) share their work units out to cores. A unit is a work
// item (forward: one Q chunk of a head stack, sdpa_item_math.hpp) or a causal light/heavy pair (backward:
// one row pair). Each core gets runs of consecutive units. Only the contiguous split is shared: the ops
// decide differently when to pair. The forward pairs only when there is a pair per core
// (sdpa_fw_config.cpp, main's forward rule, which ring SDPA relies on) and then uses split_items_paired;
// the backward pairs whenever there are more rows than cores, on min(cores, pairs) cores (its program
// factories, main's backward rule).
namespace ttml::metal {

struct ItemRun {
    uint32_t first_item;
    uint32_t num_items;
};
using CoreItemRuns = std::vector<ItemRun>;

// One contiguous block of items per core; the first (total_items % num_cores) cores get one more.
inline std::vector<CoreItemRuns> split_items_contiguous(uint32_t total_items, uint32_t num_cores) {
    std::vector<CoreItemRuns> per_core(num_cores);
    const uint32_t items_per_core = total_items / num_cores;
    const uint32_t remainder = total_items % num_cores;
    uint32_t item = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        const uint32_t num_items = items_per_core + (core < remainder ? 1U : 0U);
        per_core[core].push_back({item, num_items});
        item += num_items;
    }
    return per_core;
}

inline void append_item(CoreItemRuns& runs, uint32_t item) {
    if (!runs.empty() && runs.back().first_item + runs.back().num_items == item) {
        ++runs.back().num_items;
    } else {
        runs.push_back({item, 1U});
    }
}

// Forward, causal: Q chunk i of a head stack attends ~(i+1) K/V steps, so pairing chunk i with chunk
// (q_num_chunks - 1 - i) gives every pair the same work. The pairs are handed out contiguously like
// items above; the items are numbered by sdpa_items::item_of. The order of items within a core does not
// change its total work, so they are sorted, which folds them into the fewest runs (a fully covered stack
// is one run).
inline std::vector<CoreItemRuns> split_items_paired(uint32_t num_stacks, uint32_t q_num_chunks, uint32_t num_cores) {
    const uint32_t pairs_per_stack = q_num_chunks / 2U;
    const auto pair_runs = split_items_contiguous(num_stacks * pairs_per_stack, num_cores);
    std::vector<CoreItemRuns> per_core(num_cores);
    std::vector<uint32_t> items;
    for (uint32_t core = 0; core < num_cores; ++core) {
        const auto [first_pair, num_pairs] = pair_runs[core].front();
        items.clear();
        for (uint32_t p = first_pair; p < first_pair + num_pairs; ++p) {
            const uint32_t stack = p / pairs_per_stack;
            const uint32_t light = p % pairs_per_stack;
            items.push_back(sdpa_items::item_of(stack, light, q_num_chunks));
            items.push_back(sdpa_items::item_of(stack, q_num_chunks - 1U - light, q_num_chunks));
        }
        std::sort(items.begin(), items.end());
        for (const uint32_t item : items) {
            append_item(per_core[core], item);
        }
    }
    return per_core;
}

}  // namespace ttml::metal
