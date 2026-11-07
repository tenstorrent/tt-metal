// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, pool step: unit order and per-core unit ranges, shared by the three kernels.
//
// A unit is (tensor, head, group): tensor 0/1/2 = Q/K/V, and group g covers the tile rows
// [TILE_ROWS_PER_GROUP * g, TILE_ROWS_PER_GROUP * (g + 1)) of that head's shard, clipped to SHARD_ROW_TILES.
// 64 tile rows = 2048 tokens = 32 blocks of 64 tokens, which pool to one 32-row output tile row.
// Groups g < FULL_GROUPS are full, group FULL_GROUPS (if any) holds the remaining rows, later groups are empty.
//
// Full units come first. If they fit the grid with a core to spare, each core gets at most one full unit and the
// partial units share the remaining cores; otherwise units are split evenly.

#pragma once

#include <cstdint>

namespace sol_pool {

constexpr uint32_t BLOCK_TOKENS = 64;         // tokens per Sol-Attn block
constexpr uint32_t TILE_ROWS_PER_GROUP = 64;  // 32 blocks x 2 tile rows each: one output tile row

struct Unit {
    uint32_t tensor;  // 0 = Q, 1 = K, 2 = V
    uint32_t head;
    uint32_t group;  // output tile row
};

// NUM_HEADS: heads on this device. SHARD_ROW_TILES: tile rows (32 tokens each) per shard.
// SHARD_BLOCK_TILES: output tile rows per head = blocks per shard padded to 32, divided by 32.
template <uint32_t NUM_HEADS, uint32_t SHARD_ROW_TILES, uint32_t SHARD_BLOCK_TILES>
struct Layout {
    static constexpr uint32_t FULL_GROUPS = (SHARD_ROW_TILES / TILE_ROWS_PER_GROUP < SHARD_BLOCK_TILES)
                                                ? SHARD_ROW_TILES / TILE_ROWS_PER_GROUP
                                                : SHARD_BLOCK_TILES;
    static constexpr uint32_t PARTIAL_GROUPS = SHARD_BLOCK_TILES - FULL_GROUPS;
    static constexpr uint32_t NUM_FULL_UNITS = 3 * NUM_HEADS * FULL_GROUPS;
    static constexpr uint32_t NUM_PARTIAL_UNITS = 3 * NUM_HEADS * PARTIAL_GROUPS;
    static constexpr uint32_t NUM_UNITS = NUM_FULL_UNITS + NUM_PARTIAL_UNITS;
    // Divisors that are never zero (the branch that would divide by zero is never taken then).
    static constexpr uint32_t FULL_GROUPS_OR_1 = FULL_GROUPS ? FULL_GROUPS : 1;
    static constexpr uint32_t PARTIAL_GROUPS_OR_1 = PARTIAL_GROUPS ? PARTIAL_GROUPS : 1;

    static Unit decode(uint32_t unit) {
        if (unit < NUM_FULL_UNITS) {
            return {
                unit / (NUM_HEADS * FULL_GROUPS_OR_1),
                (unit % (NUM_HEADS * FULL_GROUPS_OR_1)) / FULL_GROUPS_OR_1,
                unit % FULL_GROUPS_OR_1};
        }
        const uint32_t partial = unit - NUM_FULL_UNITS;
        return {
            partial / (NUM_HEADS * PARTIAL_GROUPS_OR_1),
            (partial % (NUM_HEADS * PARTIAL_GROUPS_OR_1)) / PARTIAL_GROUPS_OR_1,
            FULL_GROUPS + partial % PARTIAL_GROUPS_OR_1};
    }

    // Units [start, start + count) of core core_index of num_cores.
    static void range(uint32_t core_index, uint32_t num_cores, uint32_t& start, uint32_t& count) {
        const bool one_full_per_core =
            NUM_FULL_UNITS > 0 && NUM_FULL_UNITS + (NUM_PARTIAL_UNITS > 0 ? 1u : 0u) <= num_cores;
        if (one_full_per_core) {
            if (core_index < NUM_FULL_UNITS) {
                start = core_index;
                count = 1;
                return;
            }
            const uint32_t partial_cores = num_cores - NUM_FULL_UNITS;  // cores left for the partial units
            const uint32_t partial_core_index = core_index - NUM_FULL_UNITS;
            start = NUM_FULL_UNITS + partial_core_index * NUM_PARTIAL_UNITS / partial_cores;
            count = NUM_FULL_UNITS + (partial_core_index + 1) * NUM_PARTIAL_UNITS / partial_cores - start;
            return;
        }
        start = core_index * NUM_UNITS / num_cores;
        count = (core_index + 1) * NUM_UNITS / num_cores - start;
    }
};

}  // namespace sol_pool
