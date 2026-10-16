// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only check of the unicast all_gather schedule (ttnn/.../all_gather/device/kernels/gather_rules.hpp
// unicast_schedule), at seqno level, no bytes. Runs every device's reader and writer, both directions, as state
// machines with the kernels' semantics (unicast_reader.cpp / unicast_writer.cpp: CB depth 2, B chunks per CB entry,
// data_valid signalled at the factory's granularity for the leading num_granular iterations) and checks
//  (a) every (device, stripe, seqno) of the output is written exactly once, own stripe locally, others remotely;
//  (b) the run completes (no deadlock) and no relay reads an unwritten chunk;
//  (c) total_chunks == chunks the upstream writer actually delivers (the reader subtracts it from a reused
//      semaphore, so it must be exact).
// Line and ring, even and odd N, N = 2 (whose ring iteration 0 is also the split one).

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <sstream>
#include <string>
#include <vector>

#include "ttnn/operations/ccl/all_gather/device/kernels/gather_rules.hpp"

namespace {

enum Dir : uint32_t { Forward = 0, Backward = 1 };

UnicastSchedule schedule(uint32_t n, bool is_ring, uint32_t dev, Dir dir, uint32_t slice_count) {
    return unicast_schedule(n, is_ring, dev, dir == Forward, slice_count);
}

// The device a direction sends to (the factory's get_physical_neighbor_from_physical_coord on a 1D axis).
uint32_t neighbor(uint32_t n, bool is_ring, uint32_t dev, uint32_t dir) {
    if (is_ring) {
        return dir == Forward ? (dev + 1) % n : (dev + n - 1) % n;
    }
    return dir == Forward ? dev + 1 : dev - 1;
}

// Seqno range iteration `iter` of a direction sends (and, on iteration 0 with do_local_write, writes locally).
struct IterRange {
    uint32_t stripe, skip, take;
};
IterRange iter_range(const UnicastSchedule& s, uint32_t dev, uint32_t n, uint32_t slice_count, uint32_t iter) {
    const bool last = iter + 1 == s.num_iters;
    return {(dev + iter * s.stripe_step) % n, last ? s.final_skip : 0u, last ? s.final_take : slice_count};
}

// The reader's data_valid base for iteration iter >= 1 (unicast_reader.cpp).
uint32_t base_seqno(uint32_t iter, uint32_t slice_count, uint32_t skip) {
    return iter == 0 ? 0 : (iter - 1) * slice_count + skip;
}

struct Entry {
    uint32_t stripe, first, count;
};

std::vector<std::string> run(uint32_t n, bool ring, uint32_t c, uint32_t b) {
    std::vector<std::string> viol;
    const std::string topo = ring ? "ring" : "line";
    const uint32_t depth = 2;
    const uint32_t g = std::max(1u, ((c + b - 1) / b + 1) / 2);  // data_valid_granularity at 2 signals per stripe
    const uint32_t A = 2 * n;                                    // agent key = dev*2 + dir
    std::vector<UnicastSchedule> S(A);
    for (uint32_t d = 0; d < n; ++d) {
        for (uint32_t dr = 0; dr < 2; ++dr) {
            S[d * 2 + dr] = schedule(n, ring, d, Dir(dr), c);
        }
    }
    // writes[(dev*n + stripe)*c + seq] -> list of (local?) flags
    std::vector<std::vector<bool>> writes(size_t(n) * n * c);
    auto cell = [&](uint32_t d, uint32_t st, uint32_t q) -> std::vector<bool>& {
        return writes[(size_t(d) * n + st) * c + q];
    };
    std::vector<uint64_t> dv(A, 0), delivered(A, 0);
    std::vector<std::deque<Entry>> cb(A);

    struct RState {
        uint32_t it = 0, r = 0;
        bool done = false;
    };
    struct WState {
        uint32_t it = 0, sent = 0, pend_c = 0, pend_p = 0;
        bool done = false;
    };
    std::vector<RState> R(A);
    std::vector<WState> W(A);

    auto reader_step = [&](uint32_t k) -> bool {
        auto& st = R[k];
        const auto& s = S[k];
        const uint32_t d = k / 2;
        if (st.done) {
            return false;
        }
        if (st.it < s.num_iters) {
            const auto ir = iter_range(s, d, n, c, st.it);
            if (st.r >= ir.take) {
                st.it++;
                st.r = 0;
                return true;
            }
            const uint32_t bt = std::min(b, ir.take - st.r);
            if (st.it > 0) {
                if (dv[k] < base_seqno(st.it, c, ir.skip) + st.r + bt) {
                    return false;
                }
                for (uint32_t q = ir.skip + st.r; q < ir.skip + st.r + bt; ++q) {
                    if (cell(d, ir.stripe, q).empty()) {
                        std::ostringstream o;
                        o << "(b) RACE N=" << n << " " << topo << " C=" << c << " dev=" << d << " dir=" << k % 2
                          << " stripe=" << ir.stripe << " seq=" << q;
                        viol.push_back(o.str());
                    }
                }
            }
            if (cb[k].size() >= depth) {
                return false;
            }
            cb[k].push_back({ir.stripe, ir.skip + st.r, bt});
            st.r += bt;
            return true;
        }
        if (dv[k] < s.total_chunks) {
            return false;
        }
        dv[k] -= s.total_chunks;
        st.done = true;
        return true;
    };
    auto writer_step = [&](uint32_t k) -> bool {
        auto& st = W[k];
        const auto& s = S[k];
        const uint32_t d = k / 2, dr = k % 2;
        if (st.done) {
            return false;
        }
        if (st.it >= s.num_iters) {
            st.done = true;
            return true;
        }
        const auto ir = iter_range(s, d, n, c, st.it);
        const uint32_t nb = neighbor(n, ring, d, dr);
        const uint32_t nk = nb * 2 + dr;
        if (st.sent >= ir.take) {
            if (st.pend_c) {
                dv[nk] += st.pend_c;
                delivered[nk] += st.pend_c;
            }
            st.pend_c = st.pend_p = st.sent = 0;
            st.it++;
            return true;
        }
        if (cb[k].empty()) {
            return false;
        }
        const Entry e = cb[k].front();
        cb[k].pop_front();
        const bool local_copy = st.it == 0 && s.do_local_write;
        for (uint32_t q = e.first; q < e.first + e.count; ++q) {
            cell(nb, e.stripe, q).push_back(false);
            if (local_copy) {
                cell(d, e.stripe, q).push_back(true);
            }
        }
        st.pend_c += e.count;
        st.pend_p += 1;
        if (st.it < s.num_granular && st.pend_p == g) {
            dv[nk] += st.pend_c;
            delivered[nk] += st.pend_c;
            st.pend_c = st.pend_p = 0;
        }
        st.sent += e.count;
        return true;
    };

    for (;;) {
        bool progress = false, all_done = true;
        for (uint32_t k = 0; k < A; ++k) {
            progress |= reader_step(k);
            progress |= writer_step(k);
            all_done &= R[k].done && W[k].done;
        }
        if (all_done) {
            break;
        }
        if (!progress) {
            for (uint32_t k = 0; k < A; ++k) {
                if (!R[k].done || !W[k].done) {
                    std::ostringstream o;
                    o << "(b) DEADLOCK N=" << n << " " << topo << " C=" << c << " dev=" << k / 2 << " dir=" << k % 2
                      << " reader_done=" << R[k].done << " writer_done=" << W[k].done << " dv=" << dv[k]
                      << " total_chunks=" << S[k].total_chunks;
                    viol.push_back(o.str());
                }
            }
            break;
        }
    }

    for (uint32_t d = 0; d < n; ++d) {
        for (uint32_t st = 0; st < n; ++st) {
            const bool own = st == d;
            const char* kinds[3] = {"UNWRITTEN", "MULTI", "WRONG_SRC"};
            for (int kind = 0; kind < 3; ++kind) {
                // collapse to ranges
                std::ostringstream rs;
                int64_t start = -1, prev = -1;
                auto flush = [&]() {
                    if (start >= 0) {
                        rs << "[" << start << "," << prev + 1 << ")";
                    }
                };
                for (uint32_t q = 0; q < c; ++q) {
                    const auto& w = cell(d, st, q);
                    const bool hit = kind == 0 ? w.empty() : kind == 1 ? w.size() > 1 : (w.size() == 1 && w[0] != own);
                    if (!hit) {
                        continue;
                    }
                    if (start >= 0 && q == prev + 1) {
                        prev = q;
                        continue;
                    }
                    flush();
                    start = prev = q;
                }
                flush();
                if (!rs.str().empty()) {
                    std::ostringstream o;
                    o << "(a) " << kinds[kind] << " N=" << n << " " << topo << " C=" << c << " dev=" << d
                      << " stripe=" << st << (own ? " (own)" : " (remote)") << " seqnos " << rs.str();
                    viol.push_back(o.str());
                }
            }
        }
    }
    for (uint32_t k = 0; k < A; ++k) {
        if (delivered[k] != S[k].total_chunks) {
            std::ostringstream o;
            o << "(c) N=" << n << " " << topo << " C=" << c << " dev=" << k / 2 << " dir=" << k % 2
              << " total_chunks=" << S[k].total_chunks << " delivered=" << delivered[k];
            viol.push_back(o.str());
        }
    }
    return viol;
}

constexpr uint32_t kN[] = {2, 3, 4, 5, 6, 8};
constexpr uint32_t kC[] = {1, 7, 56, 57, 1000};
constexpr uint32_t kB[] = {1, 3};

}  // namespace

// Every N, topology, slice count and CB entry size is clean: complete, written once, deadlock-free, exact totals.
TEST(AllGatherUnicastSchedule, CompleteAndDeadlockFree) {
    for (uint32_t n : kN) {
        for (bool ring : {true, false}) {
            for (uint32_t c : kC) {
                for (uint32_t b : kB) {
                    const auto v = run(n, ring, c, b);
                    for (const auto& s : v) {
                        ADD_FAILURE() << "B=" << b << " " << s;
                    }
                }
            }
        }
    }
}

// Pure-function spot checks of the derivation itself.
TEST(AllGatherUnicastSchedule, DerivationSpotChecks) {
    // 2-device ring: one iteration, which is also the split one, so both directions write their half locally.
    constexpr auto f = unicast_schedule(2, true, 0, true, 57);
    constexpr auto bk = unicast_schedule(2, true, 0, false, 57);
    static_assert(f.num_iters == 1 && bk.num_iters == 1);
    static_assert(f.final_skip == 0 && f.final_take == 28 && f.do_local_write);
    static_assert(bk.final_skip == 28 && bk.final_take == 29 && bk.do_local_write);
    static_assert(f.total_chunks == 28 && bk.total_chunks == 29);
    // 4-device ring: iteration 0 is not the split one, so forward's local write covers the whole stripe.
    static_assert(unicast_schedule(4, true, 1, true, 57).num_iters == 2);
    static_assert(unicast_schedule(4, true, 1, true, 57).do_local_write);
    static_assert(!unicast_schedule(4, true, 1, false, 57).do_local_write);
    static_assert(unicast_schedule(4, true, 1, true, 57).num_granular == 1);
    // Odd ring: no split.
    static_assert(unicast_schedule(5, true, 2, false, 57).final_skip == 0);
    static_assert(unicast_schedule(5, true, 2, false, 57).final_take == 57);
    // Line: only the last device's backward writes locally; its forward is a sink direction.
    static_assert(unicast_schedule(4, false, 3, false, 7).do_local_write);
    static_assert(unicast_schedule(4, false, 3, true, 7).num_iters == 0);
    static_assert(unicast_schedule(4, false, 0, false, 7).num_iters == 0);
    static_assert(unicast_schedule(4, false, 1, true, 7).total_chunks == 7);    // receives device 0's stripe
    static_assert(unicast_schedule(4, false, 1, false, 7).total_chunks == 14);  // devices 2 and 3
    // A line's downstream relays all but its last stripe; the end device's upstream signals only once.
    static_assert(unicast_schedule(4, false, 2, true, 7).num_granular == 0);
    static_assert(unicast_schedule(4, false, 1, true, 7).num_granular == 2);
    static_assert(relay_iters(4, false, 0, true) == 1 && relay_iters(4, false, 0, false) == 0);
    SUCCEED();
}
