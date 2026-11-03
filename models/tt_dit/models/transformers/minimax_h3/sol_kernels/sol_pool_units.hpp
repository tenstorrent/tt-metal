// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, pool step: unit order and per-core unit ranges, shared by the three kernels.
//
// A unit is (tensor t, head h, output tile row g): the 64 tile rows [64g, 64g+64) of one Q/K/V shard, clipped to RT.
// Groups g < G_FULL are full, group G_FULL (if any) holds the remaining rows, later groups are empty.
//
// Full units come first. If they fit the grid with a core to spare, each core gets at most one full unit and the short
// units share the remaining cores; otherwise units are split evenly.

#pragma once

#include <cstdint>

namespace sol_pool {

constexpr uint32_t BLOCK = 64;

struct Unit {
    uint32_t t, h, g;
};

template <uint32_t H, uint32_t RT, uint32_t NBPT>
struct Layout {
    static constexpr uint32_t G_FULL = (RT / BLOCK < NBPT) ? RT / BLOCK : NBPT;
    static constexpr uint32_t G_SHORT = NBPT - G_FULL;
    static constexpr uint32_t NF = 3 * H * G_FULL;
    static constexpr uint32_t NS = 3 * H * G_SHORT;
    static constexpr uint32_t UNITS = NF + NS;
    // Guard the divisions by zero (that branch is never taken then).
    static constexpr uint32_t GF1 = G_FULL ? G_FULL : 1;
    static constexpr uint32_t GS1 = G_SHORT ? G_SHORT : 1;

    static Unit decode(uint32_t u) {
        if (u < NF) {
            return {u / (H * GF1), (u % (H * GF1)) / GF1, u % GF1};
        }
        const uint32_t s = u - NF;
        return {s / (H * GS1), (s % (H * GS1)) / GS1, G_FULL + s % GS1};
    }

    // Units [start, start + count) of core ci of ncores.
    static void range(uint32_t ci, uint32_t ncores, uint32_t& start, uint32_t& count) {
        const bool one_full_per_core = NF > 0 && NF + (NS > 0 ? 1u : 0u) <= ncores;
        if (one_full_per_core) {
            if (ci < NF) {
                start = ci;
                count = 1;
                return;
            }
            const uint32_t rest = ncores - NF;  // cores left for the short units
            const uint32_t cj = ci - NF;
            start = NF + cj * NS / rest;
            count = NF + (cj + 1) * NS / rest - start;
            return;
        }
        start = ci * UNITS / ncores;
        count = (ci + 1) * UNITS / ncores - start;
    }
};

}  // namespace sol_pool
