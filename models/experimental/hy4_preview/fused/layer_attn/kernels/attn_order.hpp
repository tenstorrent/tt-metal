// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The order in which a merge core of X4 takes its partners' partials (fused/layer_attn/PLAN.md section 19.14), free
// of kernel-only dependencies so that the host runs the same text (tests/host_order.cpp, tests/test_order_host.py).
//
// A stage (the column partners, then the row partners) delivers EXACTLY `count` partials to the merge kernel in
// every outcome. In arrival order (the kernels' own loop, the default) the partner taken next is the lowest one
// whose partial has landed: the merges of a row then depend on which payload the fabric delivered first, and every
// merge packs to BFloat16, so two replays of one traced step can differ. In FIXED order the partner taken next is
// the lowest one still missing, and the stage waits for it: one sequence of merges on every replay.
#pragma once

#include <cstdint>

namespace attn_order {

constexpr uint32_t NONE = 32;

// The partner to deliver now, or NONE. `remaining`: the partners (bits) not delivered yet; `seen`: those of them
// whose partial has landed.
constexpr uint32_t next_arrived(uint32_t seen) { return seen != 0 ? static_cast<uint32_t>(__builtin_ctz(seen)) : NONE; }
constexpr uint32_t next_fixed(uint32_t seen, uint32_t remaining) {
    const uint32_t first = remaining & (0u - remaining);
    return (seen & first) != 0 ? static_cast<uint32_t>(__builtin_ctz(first)) : NONE;
}

// One stage. sample(): the arrival bits as they are now; deliver(r): partner r's partial to the merge kernel;
// blank(): a partial's worth of pages as the buffer holds them (the merge kernel waits for `count` partials and must
// get them whatever happens); limit: polls without a delivery before the stage gives up. Returns the partners
// still missing (0: the stage is complete).
template <bool FIXED, typename Sample, typename Deliver, typename Blank>
inline uint32_t stage(uint32_t mask, uint32_t count, uint32_t limit, Sample sample, Deliver deliver, Blank blank) {
    uint32_t remaining = mask;
    uint32_t pushed = 0;
    uint32_t polls = 0;
    while (remaining != 0 && pushed < count) {
        const uint32_t seen = sample() & remaining;
        const uint32_t r = FIXED ? next_fixed(seen, remaining) : next_arrived(seen);
        if (r != NONE) {
            deliver(r);
            remaining &= ~(1u << r);
            ++pushed;
            polls = 0;
        } else if (++polls >= limit) {
            break;
        }
    }
    for (; pushed < count; ++pushed) {
        blank();
    }
    return remaining;
}

}  // namespace attn_order
