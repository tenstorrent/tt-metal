// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Host check of the receive slots of the SM_ROWS form (tests/test_due_slots.py compiles and runs this). It includes
// the kernels' own header: the list of due (user, step) pairs of an S-block (sm::DueList) is what the io kernel
// publishes from (kernels/sm_attn_io.cpp: `due.publishable(due_k, arrived, gave_up)` on the next slot only), what the
// compute kernel consumes (kernels/sm_attn.cpp: `for k < due.n`), and what a sender numbers its partner's slot by
// (`DueList(counts, pb).slot(u, s)`). For every count row read from stdin (16 raw counts of the selection table) and
// every S-block:
//   - the io kernel's counts are the row clamped to the table's capacity (sm_dm.hpp clamp_count); the compute kernel's
//     are those through the two-16-bit-counts-per-word copy (sm_attn_io.cpp / sm_attn.cpp);
//   - the sequence the io kernel publishes, with the partials arriving in a random order, and again when it gives up
//     after a random part of them, must equal the sequence the compute kernel consumes;
//   - the compute kernel's per-user bookkeeping (first / last merge of a user) must close: a user's pairs are
//     contiguous and as many as sm::num_receives says;
//   - every sender's slot must be a slot of its partner's list, every slot must have exactly one sender;
//   - a list is never longer than the buffers hold (7 per 8 users) and is empty exactly when sm::any_receive is false.
// Output: one line "rows <n> blocks <n> pairs <n> longest <n> checksum <n>", or the first disagreement and exit 1.

#define SM_ROWS 1
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "../kernels/sm_common.hpp"

constexpr uint32_t USERS = 16, GROUPS = 8, CHUNK = 128, TOPK = 2048, RECEIVE_SLOTS = 7;
using Due = sm::DueList<USERS, GROUPS, CHUNK>;
using Seq = std::vector<std::pair<uint32_t, uint32_t>>;

static Seq publish(const Due& due, const Seq& arrivals, size_t give_up_after) {
    // the io kernel's loop on its receive slots: only the next slot of the list, when it is publishable
    Seq out;
    uint32_t arrived[sm::MERGE_STEPS] = {0, 0, 0};
    uint32_t k = 0;
    bool gave_up = false;
    for (size_t t = 0;; ++t) {
        while (k < due.n && due.publishable(k, arrived, gave_up)) {
            out.emplace_back(due.user[k], due.step[k]);
            ++k;
        }
        if (k == due.n) {
            break;
        }
        if (t >= give_up_after || t >= arrivals.size()) {
            gave_up = true;  // nothing more arrives: the kernel's bounded idle count ran out
        } else {
            arrived[arrivals[t].second] |= 1u << arrivals[t].first;
        }
    }
    return out;
}

int main() {
    std::mt19937 rng(12345);
    unsigned long rows = 0, pairs = 0, longest = 0, checksum = 0;
    uint32_t raw[USERS];
    for (;;) {
        bool ok = true;
        for (uint32_t u = 0; u < USERS; ++u) {
            unsigned long v;
            if (std::scanf("%lu", &v) != 1) {
                ok = false;
                break;
            }
            raw[u] = static_cast<uint32_t>(v);
        }
        if (!ok) {
            break;
        }
        ++rows;
        uint32_t io_counts[USERS], cp_counts[USERS];
        for (uint32_t u = 0; u < USERS; ++u) {
            io_counts[u] = raw[u] > TOPK ? TOPK : raw[u];
        }
        for (uint32_t w = 0; w < USERS / 2; ++w) {  // two counts per word, as the io kernel hands them over
            const uint32_t word = io_counts[2 * w] | (io_counts[2 * w + 1] << 16);
            cp_counts[2 * w] = word & 0xFFFF;
            cp_counts[2 * w + 1] = word >> 16;
        }
        std::vector<Due> lists;
        for (uint32_t b = 0; b < GROUPS; ++b) {
            lists.emplace_back(io_counts, b);
        }
        std::vector<std::vector<uint32_t>> senders(GROUPS);
        for (uint32_t b = 0; b < GROUPS; ++b) {
            senders[b].assign(lists[b].n, 0);
        }
        for (uint32_t b = 0; b < GROUPS; ++b) {
            const Due& io = lists[b];
            const Due cp(cp_counts, b);
            if (io.n > 2 * RECEIVE_SLOTS || (io.n != 0) != sm::any_receive<USERS, GROUPS, CHUNK>(io_counts, b)) {
                std::printf(
                    "row %lu block %u: %u due pairs, any_receive disagrees or the buffers are too small\n",
                    rows,
                    b,
                    io.n);
                return 1;
            }
            // what the compute kernel consumes, with its bookkeeping
            Seq consumed;
            uint32_t done = 0;
            for (uint32_t k = 0; k < cp.n; ++k) {
                const uint32_t u = cp.user[k];
                if (k == 0 || cp.user[k - 1] != u) {
                    done = 0;
                }
                const uint32_t total =
                    sm::num_receives(sm::rel<GROUPS>(b, u), sm::active_blocks<GROUPS, CHUNK>(cp_counts[u]));
                const bool last_of_user = k + 1 == cp.n || cp.user[k + 1] != u;
                if (done >= total || (done == total - 1) != last_of_user) {
                    std::printf(
                        "row %lu block %u: the compute kernel's first / last bookkeeping does not close at slot %u\n",
                        rows,
                        b,
                        k);
                    return 1;
                }
                consumed.emplace_back(u, cp.step[k]);
                ++done;
            }
            // what the io kernel publishes: every partial arrives, in a random order; then it gives up part-way
            Seq arrivals(consumed);
            std::shuffle(arrivals.begin(), arrivals.end(), rng);
            const size_t cut = arrivals.empty() ? 0 : rng() % (arrivals.size() + 1);
            if (publish(io, arrivals, arrivals.size()) != consumed || publish(io, arrivals, cut) != consumed) {
                std::printf(
                    "row %lu block %u: the io kernel publishes another sequence than the compute kernel consumes\n",
                    rows,
                    b);
                return 1;
            }
            // the senders of this S-block: user u's partial goes to its partner at the step of its relative index
            for (uint32_t u = 0; u < USERS; ++u) {
                const uint32_t r = sm::rel<GROUPS>(b, u);
                const uint32_t n_act = sm::active_blocks<GROUPS, CHUNK>(io_counts[u]);
                if (r == 0 || r >= n_act) {
                    continue;
                }
                const uint32_t s = sm::send_step(r);
                const uint32_t pb = (u + r - (1u << s)) % GROUPS;
                const uint32_t slot = Due(io_counts, pb).slot(u, s);
                if (slot >= lists[pb].n) {
                    std::printf(
                        "row %lu: block %u sends user %u step %u to block %u, which has no slot for it\n",
                        rows,
                        b,
                        u,
                        s,
                        pb);
                    return 1;
                }
                ++senders[pb][slot];
            }
            pairs += io.n;
            longest = io.n > longest ? io.n : longest;
            for (uint32_t k = 0; k < io.n; ++k) {
                checksum = checksum * 1000003UL + (b * 64 + io.user[k] * 4 + io.step[k] + 1);
            }
        }
        for (uint32_t b = 0; b < GROUPS; ++b) {
            for (uint32_t k = 0; k < lists[b].n; ++k) {
                if (senders[b][k] != 1) {
                    std::printf("row %lu block %u slot %u: %u senders\n", rows, b, k, senders[b][k]);
                    return 1;
                }
            }
        }
    }
    std::printf(
        "rows %lu blocks %lu pairs %lu longest %lu checksum %lu\n", rows, rows * GROUPS, pairs, longest, checksum);
    return 0;
}
