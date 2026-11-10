// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool unit order and per-core units, shared by the pool reader, compute and writer and the senders so they
// agree without per-core runtime args.
//
// A unit is (tensor t in {q, k, v}, head h, output tile row g): the 32 blocks [32g, 32g+32) of one head of one shard,
// i.e. the 64 tile rows [64g, 64g+64) clipped to RT. Groups g < G_FULL = RT/64 are full; group G_FULL, if any, holds
// the RT % 64 remaining rows; later groups are empty (zero output tiles).
//
// The K and V units come first, so the senders can push them while the Q units are still pooled: K/V full, K/V short,
// Q full, Q short. Units are dealt to the pool cores round-robin, unit u on core u % NCORES in round u / NCORES, so
// round 0 is K/V on every core.

#pragma once

#include <cstdint>

namespace block_pool {

constexpr uint32_t BLOCK = 64;
// Program semaphores: the pool-to-route barrier on every core, then one per round on the sender cores, counting the
// K/V units of that round written to DRAM.
constexpr uint32_t BARRIER_SEM = 1;
constexpr uint32_t ROUND_SEM = 2;

struct Unit {
    uint32_t t, h, g;
};

template <uint32_t H, uint32_t RT, uint32_t NBPT>
struct Layout {
    static constexpr uint32_t G_FULL = (RT / BLOCK < NBPT) ? RT / BLOCK : NBPT;
    static constexpr uint32_t G_SHORT = NBPT - G_FULL;
    static constexpr uint32_t KV_FULL = 2 * H * G_FULL;
    static constexpr uint32_t KV_UNITS = 2 * H * NBPT;
    static constexpr uint32_t Q_FULL = KV_UNITS + H * G_FULL;
    static constexpr uint32_t UNITS = 3 * H * NBPT;
    static constexpr uint32_t GF1 = G_FULL ? G_FULL : 1;
    static constexpr uint32_t GS1 = G_SHORT ? G_SHORT : 1;

    static Unit decode(uint32_t u) {
        if (u < KV_FULL) {
            return {1 + u / (H * GF1), (u % (H * GF1)) / GF1, u % GF1};
        }
        if (u < KV_UNITS) {
            const uint32_t s = u - KV_FULL;
            return {1 + s / (H * GS1), (s % (H * GS1)) / GS1, G_FULL + s % GS1};
        }
        if (u < Q_FULL) {
            const uint32_t s = u - KV_UNITS;
            return {0, s / GF1, s % GF1};
        }
        const uint32_t s = u - Q_FULL;
        return {0, s / GS1, G_FULL + s % GS1};
    }

    // Cores holding a K/V unit in round k.
    template <uint32_t NCORES>
    static constexpr uint32_t kv_cores(uint32_t k) {
        return KV_UNITS - k * NCORES < NCORES ? KV_UNITS - k * NCORES : NCORES;
    }
};

}  // namespace block_pool
