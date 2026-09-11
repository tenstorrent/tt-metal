// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The cyclic (core, timestep) -> (row, column) schedule of the causal
// backward pass without atomic DRAM additions.
//
// Reference: main.tex in the tt-flash-attn prototype, Algorithm 2
// ("FlashAttention-2 Causal Backward Pass without Atomic DRAM Additions") and
// its two relay refinements, Algorithms 3 and 4. The Python reference is
// tt_flash_attn/schedule.py; tests/ops/cyclic_schedule_test.cpp pins this
// header against fixtures generated from it.
//
// Conventions follow the paper, so they are 1-based where it is:
//   * T = 2C block rows and columns, and M = T + 1 is the modulus;
//   * cores are 1 <= c <= C, block indices are 1 <= j <= i <= T;
//   * timesteps are 0-based, 0 <= t <= T, so there are T + 1 of them.
// Zero is therefore free as a "no such core" and "no such column" sentinel.
//
// This header is compiled for both the host (program factory) and the device
// (kernels), so it stays free of everything a kernel cannot have: no
// <vector>, no std::, no allocation, no exceptions, no asserts. Every
// function is constexpr and integer-only. The device evaluates the schedule
// on the core rather than receiving it in runtime arguments, because it is
// this much arithmetic and at T = 128 the per-timestep descriptors would not
// fit the runtime-argument budget.
//
// Preconditions are documented, not checked. Out-of-domain arguments produce
// meaningless values rather than a diagnostic; the host-side test covers the
// whole valid domain for every supported C.

#pragma once

#include <cstdint>

namespace ttml::metal::ops::cyclic_sdpa_bw {

//: No such column. Columns are 1..T.
constexpr uint32_t kNoColumn = 0;
//: No such core. Cores are 1..C.
constexpr uint32_t kNoCore = 0;
//: No such timestep. Timesteps are 0..T, so this cannot be 0.
constexpr uint32_t kNoTimestep = 0xFFFFFFFFu;

//: A scheduled block pair: row block i against column block j.
struct BlockPair {
    uint32_t i;
    uint32_t j;
};

//: The two columns a core owns, in the order the paper names them.
struct OwnedColumns {
    uint32_t first;   // c
    uint32_t second;  // T - c + 1
};

//: A maximal run of consecutive timesteps at which a row is active.
struct Streak {
    uint32_t start;
    uint32_t end;  // inclusive
};

//: Who populates a receive slot, and by which route.
struct Producer {
    uint32_t core;  // the previous consumer, or the receiver itself
    bool internal;  // true: forwarded from the previous consumer (possibly itself)
                    // false: the receiver loads it from DRAM at a streak start
};

class CyclicSchedule {
public:
    explicit constexpr CyclicSchedule(uint32_t cores) : C_(cores) {}

    constexpr uint32_t C() const {
        return C_;
    }
    //: T = 2C block rows and columns.
    constexpr uint32_t T() const {
        return 2u * C_;
    }
    //: The modulus M = T + 1 = 2C + 1.
    constexpr uint32_t M() const {
        return 2u * C_ + 1u;
    }
    //: Number of timesteps, T + 1.
    constexpr uint32_t num_timesteps() const {
        return M();
    }

    // ------------------------------------------------------------ ownership
    //: Owner of column j: c(j) = min(j, T + 1 - j). Precondition: 1 <= j <= T.
    constexpr uint32_t owner(uint32_t j) const {
        const uint32_t mirror = M() - j;
        return j < mirror ? j : mirror;
    }

    //: The columns core c owns: c and T - c + 1. Precondition: 1 <= c <= C.
    constexpr OwnedColumns owned_columns(uint32_t c) const {
        return {c, M() - c};
    }

    // ------------------------------------------------------------- schedule
    // The block pair core c processes at timestep t:
    //
    //   m = (t + (C + 1) c) mod M
    //   (i, j) = (m, c)              if m >= c
    //   (i, j) = (m - c + T + 1, T - c + 1)  otherwise
    //
    // Precondition: 1 <= c <= C and 0 <= t <= T. Every pair it returns
    // satisfies t == i + C j (mod M).
    constexpr BlockPair pair(uint32_t c, uint32_t t) const {
        const uint32_t m = (t + (C_ + 1u) * c) % M();
        if (m >= c) {
            return {m, c};
        }
        // m + M - c rather than m - c + T + 1: same value, no unsigned wrap.
        return {m + M() - c, M() - c};
    }

    // ------------------------------------------------------------- activity
    // The column processed for row i at timestep t, or kNoColumn if the row
    // is inactive then. For an active row this is the nonzero residue
    // 2 (i - t) mod M that is also a causal column (1 <= j <= i).
    //
    // Precondition: 1 <= i <= T and 0 <= t <= T.
    constexpr uint32_t column_at(uint32_t i, uint32_t t) const {
        // i + M - t is positive because t <= T < M, so this never wraps.
        const uint32_t j = (2u * ((i + M() - t) % M())) % M();
        return (j >= 1u && j <= i) ? j : kNoColumn;
    }

    constexpr bool is_active(uint32_t i, uint32_t t) const {
        return column_at(i, t) != kNoColumn;
    }

    //: True if t is active for row i and t - 1 is not, t = 0 counting as a start.
    constexpr bool is_streak_start(uint32_t i, uint32_t t) const {
        if (!is_active(i, t)) {
            return false;
        }
        return t == 0u || !is_active(i, t - 1u);
    }

    // t_prev(i, t) = max{t' in A_i : t' < t}, or kNoTimestep if there is none.
    //
    // Walks down from t, so it costs up to t steps of cheap arithmetic. That
    // is affordable because the callers need it only at a streak start, of
    // which each row has at most three (main.tex, "Exact streak count").
    constexpr uint32_t prev_active(uint32_t i, uint32_t t) const {
        for (uint32_t s = t; s > 0u; --s) {
            if (is_active(i, s - 1u)) {
                return s - 1u;
            }
        }
        return kNoTimestep;
    }

    //: True if t begins a streak of row i other than its first.
    //
    // These are exactly the timesteps at which a consumer must reload the row
    // packet from DRAM *and* order that reload after the preceding streak's
    // spill. Algorithm 3 gets that ordering from the chip-wide barrier;
    // Algorithm 4 gets it from an endpoint counter.
    constexpr bool is_later_streak_start(uint32_t i, uint32_t t) const {
        return is_streak_start(i, t) && prev_active(i, t) != kNoTimestep;
    }

    //: The maximal streak of row i containing t. Precondition: is_active(i, t).
    constexpr Streak streak_at(uint32_t i, uint32_t t) const {
        uint32_t start = t;
        while (start > 0u && is_active(i, start - 1u)) {
            --start;
        }
        uint32_t end = t;
        while (end < T() && is_active(i, end + 1u)) {
            ++end;
        }
        return {start, end};
    }

    // True if row i is active at some timestep after t. At a streak end this
    // says whether the spill is an inter-streak one, which is what decides
    // whether the endpoint publishes its progress: a final spill needs
    // completion but no publication (main.tex, Algorithm 4).
    constexpr bool has_later_active(uint32_t i, uint32_t t) const {
        for (uint32_t s = t + 1u; s <= T(); ++s) {
            if (is_active(i, s)) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------- endpoint relay (Algorithm 4)
    // e(i, t): the core whose inter-streak spill the reload at t waits on.
    // By the endpoint spill property (main.tex, Lemma "Endpoint spill
    // property") this is always core 1 or core 2.
    //
    // Precondition: is_later_streak_start(i, t). Inside a streak the previous
    // active timestep is merely t - 1, whose column belongs to an arbitrary
    // core and certifies nothing.
    constexpr uint32_t spill_endpoint(uint32_t i, uint32_t t) const {
        return owner(column_at(i, prev_active(i, t)));
    }

    // The value the reload at t waits for: t_prev(i, t) + 1.
    //
    // Publication uses positive tags -- an endpoint that completes an
    // inter-streak spill at t* publishes t* + 1 -- so the threshold is
    // t_prev + 1 and not t. Waiting for t deadlocks whenever the row is
    // inactive at t - 1, which is every later streak start: with T = 16, row
    // 15's final streak starts at t = 16 while its preceding spill is at
    // t = 14 on core 2, whose last publication is 15.
    //
    // Precondition: is_later_streak_start(i, t).
    constexpr uint32_t endpoint_threshold(uint32_t i, uint32_t t) const {
        return prev_active(i, t) + 1u;
    }

    // ------------------------------------------------- two-slot relay protocol
    // Who populates core r's receive slot for destination timestep u.
    //
    // internal = true when the row was active at u - 1 and the packet is
    // forwarded from that consumer (possibly r itself, which is the core-1
    // self-transition); otherwise r loads it from DRAM.
    //
    // Precondition: 1 <= r <= C and 0 <= u <= T.
    constexpr Producer producer(uint32_t r, uint32_t u) const {
        const uint32_t i = pair(r, u).i;
        if (u >= 1u) {
            const uint32_t j_prev = column_at(i, u - 1u);
            if (j_prev != kNoColumn) {
                return {owner(j_prev), true};
            }
        }
        return {r, false};
    }

    // The core that consumes row i at t + 1, or kNoCore when the streak ends
    // at t and the packet is spilled to DRAM instead of forwarded.
    constexpr uint32_t next_consumer(uint32_t i, uint32_t t) const {
        if (t >= T()) {
            return kNoCore;
        }
        const uint32_t j_next = column_at(i, t + 1u);
        return j_next == kNoColumn ? kNoCore : owner(j_next);
    }

private:
    uint32_t C_;
};

}  // namespace ttml::metal::ops::cyclic_sdpa_bw
