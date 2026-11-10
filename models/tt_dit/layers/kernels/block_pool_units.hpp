// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool unit order, shared by the pool reader, compute and writer and the senders so they agree without per-core
// runtime args.
//
// A unit is (tensor t in {q, k, v}, head h, output tile row g): the 32 blocks [32g, 32g+32) of one head of one shard,
// i.e. the 64 tile rows [64g, 64g+64) clipped to RT. Groups g < G_FULL = RT/64 are full; group G_FULL, if any, holds
// the RT % 64 remaining rows; later groups are empty (zero output tiles).
//
// The K and V units come first, so the senders can push them while the Q units are still pooled: K/V full, K/V short,
// Q full, Q short. The pool cores claim units in this order from a queue on the first sender core, so a core that gets
// more read bandwidth takes more units.

#pragma once

#include <cstdint>

namespace block_pool {

constexpr uint32_t BLOCK = 64;
// Program semaphores. BARRIER_SEM: the pool-to-route barrier on every core. FLAGS_READY_SEM (pool cores): one increment
// per sender once its page is zeroed. CLAIM_SEM (pool cores): the queue's reply to the reader's latest request,
// (request number << 16) | unit. CLAIMS_SEM (pool cores): the unit pages the reader has written to CB_UNITS.
// FLAGS_ADDR_SEM (pool cores): the address of the senders' page. QUEUE_GO_SEM (sender cores): set once the page is
// zeroed, for the queue kernel on the first sender core.
constexpr uint32_t BARRIER_SEM = 1;
constexpr uint32_t FLAGS_READY_SEM = 2;
constexpr uint32_t CLAIM_SEM = 3;
constexpr uint32_t CLAIMS_SEM = 4;
constexpr uint32_t FLAGS_ADDR_SEM = 5;
constexpr uint32_t QUEUE_GO_SEM = 6;
// Pool cores: one 16 B page per claimed unit, holding its index, then a page holding UNITS.
constexpr uint32_t CB_UNITS = 31;
constexpr uint32_t UNIT_PAGE_BYTES = 16;

// Word offsets in the senders' page (CB 3 on the sender cores): the KV_UNITS completion flags, the page's address in
// its own 16 B word (the multicast source), then, used on the first sender core by the queue, one request word per
// pool core and one 16 B reply word per pool core.
template <uint32_t KV_UNITS, uint32_t NCORES>
struct SenderPage {
    static constexpr uint32_t ADDR_WORD = (KV_UNITS + 3) / 4 * 4;
    static constexpr uint32_t REQ_WORD = ADDR_WORD + 4;
    static constexpr uint32_t REPLY_WORD = REQ_WORD + (NCORES + 3) / 4 * 4;
    static constexpr uint32_t BYTES = (REPLY_WORD + 4 * NCORES) * 4;
};

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
};

}  // namespace block_pool
