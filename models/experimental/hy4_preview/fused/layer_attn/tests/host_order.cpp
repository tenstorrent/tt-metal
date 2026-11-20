// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Host run of kernels/attn_order.hpp (tests/test_order_host.py): a stage of X4's merge core against EVERY order of
// arrival of its partners, with and without partners that never land, in both orders of taking them.
//   host_order <partners: bits of the stage's mask, e.g. 0xFE> <shift of the mask (16 rows: 8 x the sub-pass)>
// Checked for every permutation of the partners as their order of arrival, every gap pattern of the arrivals
// (0, 1 or 3 polls between two) and every subset of partners that never lands:
//   - deliveries + blank pages == count, always (what the merge kernel's unpack side waits for);
//   - no partner is delivered before it has landed, none twice;
//   - FIXED: the partners delivered are, in ascending order, exactly those below the lowest partner that never
//     lands (all of them when every one lands), whatever the order of arrival; the stage reports the others missing;
//   - arrival order: every partner that lands is delivered, in the order of landing (lowest first among those
//     landed by the same poll);
//   - a stage with a partner missing gives up after `limit` polls without a delivery.
// Prints: cases run, the longest stage in polls.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../kernels/attn_order.hpp"

namespace {
constexpr uint32_t LIMIT = 64;  // (beyond the last arrival of every pattern: 7 partners, 3 polls apart)

template <bool FIXED>
bool run(
    const std::vector<uint32_t>& order,
    const std::vector<uint32_t>& gap,
    uint32_t never,
    uint32_t mask,
    uint64_t& longest) {
    const uint32_t count = static_cast<uint32_t>(__builtin_popcount(mask));
    std::vector<uint32_t> at(32, 0xFFFFFFFFu);  // the poll at which a partner's bit appears
    uint32_t t = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        t += gap[i];
        if (!((never >> order[i]) & 1u)) {
            at[order[i]] = t;
        }
    }
    uint32_t now = 0, blanks = 0;
    std::vector<uint32_t> delivered, when;
    const uint32_t missing = attn_order::stage<FIXED>(
        mask,
        count,
        LIMIT,
        [&]() {
            uint32_t word = 0;
            for (uint32_t b = 0; b < 32; ++b) {
                if (at[b] <= now) {
                    word |= 1u << b;
                }
            }
            ++now;
            return word | ~mask;  // (bits of other stages in the word must not matter)
        },
        [&](uint32_t r) {
            delivered.push_back(r);
            when.push_back(now - 1);
        },
        [&]() { ++blanks; });
    longest = std::max<uint64_t>(longest, now);
    if (delivered.size() + blanks != count) {
        return false;
    }
    uint32_t got = 0;
    for (size_t i = 0; i < delivered.size(); ++i) {
        const uint32_t r = delivered[i];
        if (!((mask >> r) & 1u) || ((got >> r) & 1u) || at[r] > when[i]) {
            return false;
        }
        got |= 1u << r;
    }
    if (missing != (mask & ~got)) {
        return false;
    }
    const uint32_t lands = mask & ~never;
    if (FIXED) {
        const uint32_t gone = mask & never;
        const uint32_t below = gone ? (mask & ((gone & (0u - gone)) - 1u)) : mask;
        if (got != below || !std::is_sorted(delivered.begin(), delivered.end())) {
            return false;
        }
    } else {
        if (got != lands) {
            return false;
        }
        for (size_t i = 1; i < delivered.size();
             ++i) {  // never a partner that landed later before one that landed earlier and was seen
            if (at[delivered[i]] < at[delivered[i - 1]] && at[delivered[i]] <= when[i - 1] &&
                delivered[i] < delivered[i - 1]) {
                return false;
            }
        }
    }
    if (missing != 0 && now < LIMIT) {
        return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }
    const uint32_t mask = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 0)) << std::strtoul(argv[2], nullptr, 0);
    std::vector<uint32_t> partners;
    for (uint32_t b = 0; b < 32; ++b) {
        if ((mask >> b) & 1u) {
            partners.push_back(b);
        }
    }
    const size_t n = partners.size();
    uint64_t cases = 0, longest = 0;
    std::vector<uint32_t> order = partners;
    const uint32_t gaps[3] = {0, 1, 3};
    do {
        for (uint32_t pattern = 0; pattern < 3; ++pattern) {
            std::vector<uint32_t> gap(n);
            for (size_t i = 0; i < n; ++i) {
                gap[i] = gaps[(pattern + i * (pattern + 1)) % 3];
            }
            for (uint32_t sub = 0; sub < (1u << n); ++sub) {
                uint32_t never = 0;
                for (size_t i = 0; i < n; ++i) {
                    if ((sub >> i) & 1u) {
                        never |= 1u << partners[i];
                    }
                }
                if (!run<true>(order, gap, never, mask, longest) || !run<false>(order, gap, never, mask, longest)) {
                    std::printf("FAILED mask %#x never %#x pattern %u\n", mask, never, pattern);
                    return 1;
                }
                cases += 2;
            }
        }
    } while (std::next_permutation(order.begin(), order.end()));
    std::printf("%llu %llu\n", static_cast<unsigned long long>(cases), static_cast<unsigned long long>(longest));
    return 0;
}
