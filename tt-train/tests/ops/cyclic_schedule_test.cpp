// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The device-side cyclic schedule must agree with the simulator that
// specifies it.
//
// metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp and parity_snake.hpp
// reimplement tt_flash_attn/schedule.py and tt_flash_attn/device/topology.py
// so a kernel can evaluate the schedule on the core. A disagreement between
// the two does not show up as a wrong gradient: a wrong endpoint threshold or
// a wrong producer hangs the relay instead. So the reimplementation is pinned
// against fixtures generated from the Python reference
// (cyclic_schedule_golden.hpp, from tools/export_golden.py).
//
// Two levels of check, because they fail differently:
//
//   * full tables for C = 4, 8 and 16, which name the (core, timestep) that
//     disagrees;
//   * a digest for every configuration including C = 64, compared section by
//     section so the first differing section names the part of the schedule
//     that is wrong.
//
// Everything here is host arithmetic. No device is opened.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cyclic_schedule_golden.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"

namespace {

using namespace ttml::metal::ops::cyclic_sdpa_bw;
namespace golden = ttml::metal::ops::cyclic_sdpa_bw::golden;

// FNV-1a 64 over a flat uint32 stream, four bytes per value, little-endian.
// Must stay identical to Digest in tools/export_golden.py; the stream order it
// documents is the fixture format.
class Digest {
public:
    void push(uint32_t v) {
        for (int shift = 0; shift < 32; shift += 8) {
            h_ ^= static_cast<uint64_t>((v >> shift) & 0xFFu);
            h_ *= kPrime;
        }
    }

    template <typename... Rest>
    void push(uint32_t first, Rest... rest) {
        push(first);
        push(static_cast<uint32_t>(rest)...);
    }

    uint64_t value() const {
        return h_;
    }

private:
    static constexpr uint64_t kOffset = 0xCBF29CE484222325ULL;
    static constexpr uint64_t kPrime = 0x100000001B3ULL;
    uint64_t h_ = kOffset;
};

// The active timesteps of row i, ascending. A_i in the paper.
std::vector<uint32_t> active_timesteps(const CyclicSchedule& s, uint32_t i) {
    std::vector<uint32_t> out;
    for (uint32_t t = 0; t <= s.T(); ++t) {
        if (s.is_active(i, t)) {
            out.push_back(t);
        }
    }
    return out;
}

// The maximal streaks of row i, in ascending order of start.
std::vector<Streak> streaks(const CyclicSchedule& s, uint32_t i) {
    std::vector<Streak> out;
    for (uint32_t t = 0; t <= s.T(); ++t) {
        if (s.is_streak_start(i, t)) {
            out.push_back(s.streak_at(i, t));
        }
    }
    return out;
}

struct Move {
    uint32_t i;
    uint32_t src;
    uint32_t dst;
};

// Row-packet forwards from t to t + 1, rows ascending. src == dst is a
// self-transition.
std::vector<Move> moves(const CyclicSchedule& s, uint32_t t) {
    std::vector<Move> out;
    for (uint32_t i = 1; i <= s.T(); ++i) {
        const uint32_t j0 = s.column_at(i, t);
        const uint32_t j1 = s.column_at(i, t + 1);
        if (j0 != kNoColumn && j1 != kNoColumn) {
            out.push_back({i, s.owner(j0), s.owner(j1)});
        }
    }
    return out;
}

// The digest of one configuration, recorded at every section boundary. The
// order here IS the fixture format; see the generated header.
std::vector<std::pair<std::string, uint64_t>> section_digests(
    uint32_t C, uint32_t grid_w) {
    const CyclicSchedule s(C);
    const uint32_t T = s.T();
    Digest d;
    std::vector<std::pair<std::string, uint64_t>> out;
    const auto mark = [&](const char* name) { out.emplace_back(name, d.value()); };

    // 1. shape
    d.push(C, T);
    mark("shape");

    // 2. column ownership
    for (uint32_t c = 1; c <= C; ++c) {
        const auto owned = s.owned_columns(c);
        d.push(owned.first, owned.second);
    }
    for (uint32_t j = 1; j <= T; ++j) {
        d.push(s.owner(j));
    }
    mark("ownership");

    // 3. the schedule itself
    for (uint32_t c = 1; c <= C; ++c) {
        for (uint32_t t = 0; t <= T; ++t) {
            const auto p = s.pair(c, t);
            d.push(p.i, p.j);
        }
    }
    mark("pairs");

    // 4. row activity
    for (uint32_t i = 1; i <= T; ++i) {
        const auto a = active_timesteps(s, i);
        d.push(static_cast<uint32_t>(a.size()));
        for (uint32_t t : a) {
            d.push(t);
        }
    }
    for (uint32_t i = 1; i <= T; ++i) {
        const auto st = streaks(s, i);
        d.push(static_cast<uint32_t>(st.size()));
        for (const auto& iv : st) {
            d.push(iv.start, iv.end);
        }
    }
    mark("activity");

    // 5. later streak starts: the reload gate of Algorithm 4
    for (uint32_t i = 1; i <= T; ++i) {
        for (uint32_t t = 0; t <= T; ++t) {
            if (s.is_later_streak_start(i, t)) {
                d.push(1u, s.prev_active(i, t), s.spill_endpoint(i, t), s.endpoint_threshold(i, t));
            } else {
                d.push(0u);
            }
        }
    }
    mark("later_streak_starts");

    // 6. the two-slot protocol's static producer map
    for (uint32_t r = 1; r <= C; ++r) {
        for (uint32_t u = 0; u <= T; ++u) {
            const auto p = s.producer(r, u);
            d.push(p.core, p.internal ? 1u : 0u);
        }
    }
    mark("producer");

    for (uint32_t i = 1; i <= T; ++i) {
        for (uint32_t t = 0; t <= T; ++t) {
            d.push(s.next_consumer(i, t));
        }
    }
    mark("next_consumer");

    // 7. placement
    for (uint32_t k = 0; k < C; ++k) {
        d.push(snake_core_at(C, k));
    }
    for (uint32_t c = 1; c <= C; ++c) {
        const auto xy = placement_of(C, grid_w, c);
        d.push(xy.x, xy.y);
    }
    mark("placement");

    // 8. forwards per timestep
    for (uint32_t t = 0; t < T; ++t) {
        const auto mv = moves(s, t);
        d.push(static_cast<uint32_t>(mv.size()));
        for (const auto& m : mv) {
            d.push(m.i, m.src, m.dst);
        }
    }
    mark("moves");

    return out;
}

}  // namespace

// ------------------------------------------------------------------ digests
TEST(CyclicScheduleTest, DigestMatchesTheSimulatorSectionBySection) {
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const auto& cfg = golden::kConfigs[k];
        const auto mine = section_digests(cfg.C, cfg.grid_w);
        ASSERT_EQ(mine.size(), cfg.num_sections) << "C=" << cfg.C;
        for (uint32_t n = 0; n < cfg.num_sections; ++n) {
            // The first section that differs is the part of the schedule that
            // is wrong; later sections differ only because the hash carries.
            ASSERT_EQ(mine[n].first, std::string(cfg.sections[n].name)) << "C=" << cfg.C;
            ASSERT_EQ(mine[n].second, cfg.sections[n].digest)
                << "C=" << cfg.C << ": section '" << mine[n].first
                << "' disagrees with the Python reference";
        }
        EXPECT_EQ(mine.back().second, cfg.digest) << "C=" << cfg.C;
    }
}

// -------------------------------------------------------------- full tables
TEST(CyclicScheduleTest, PairsMatchTheFullTables) {
    const auto check = [](uint32_t C, const uint32_t* table) {
        const CyclicSchedule s(C);
        for (uint32_t c = 1; c <= C; ++c) {
            for (uint32_t t = 0; t <= s.T(); ++t) {
                const auto p = s.pair(c, t);
                const uint32_t base = 2u * ((c - 1u) * (s.T() + 1u) + t);
                EXPECT_EQ(p.i, table[base]) << "C=" << C << " c=" << c << " t=" << t << " (row)";
                EXPECT_EQ(p.j, table[base + 1u]) << "C=" << C << " c=" << c << " t=" << t
                                                 << " (column)";
            }
        }
    };
    check(4, golden::kPairs_C4);
    check(8, golden::kPairs_C8);
    check(16, golden::kPairs_C16);
}

TEST(CyclicScheduleTest, SnakeAndPlacementMatchTheFullTables) {
    const auto check = [](uint32_t C, uint32_t grid_w, const uint32_t* snake,
                          const uint32_t* placement) {
        for (uint32_t k = 0; k < C; ++k) {
            EXPECT_EQ(snake_core_at(C, k), snake[k]) << "C=" << C << " k=" << k;
        }
        for (uint32_t c = 1; c <= C; ++c) {
            const auto xy = placement_of(C, grid_w, c);
            EXPECT_EQ(xy.x, placement[2u * (c - 1u)]) << "C=" << C << " c=" << c << " (x)";
            EXPECT_EQ(xy.y, placement[2u * (c - 1u) + 1u]) << "C=" << C << " c=" << c << " (y)";
        }
    };
    check(4, 2, golden::kSnake_C4, golden::kPlacement_C4);
    check(8, 4, golden::kSnake_C8, golden::kPlacement_C8);
    check(16, 4, golden::kSnake_C16, golden::kPlacement_C16);
}

TEST(CyclicScheduleTest, LaterStreakStartsMatchTheFullTables) {
    const auto check = [](uint32_t C, const uint32_t* table, uint32_t count) {
        const CyclicSchedule s(C);
        uint32_t seen = 0;
        for (uint32_t i = 1; i <= s.T(); ++i) {
            for (uint32_t t = 0; t <= s.T(); ++t) {
                if (!s.is_later_streak_start(i, t)) {
                    continue;
                }
                ASSERT_LT(seen, count) << "C=" << C << ": more later streak starts than golden";
                const uint32_t* row = table + 5u * seen;
                EXPECT_EQ(i, row[0]);
                EXPECT_EQ(t, row[1]);
                EXPECT_EQ(s.prev_active(i, t), row[2]) << "C=" << C << " i=" << i << " t=" << t;
                EXPECT_EQ(s.spill_endpoint(i, t), row[3]) << "C=" << C << " i=" << i << " t=" << t;
                EXPECT_EQ(s.endpoint_threshold(i, t), row[4])
                    << "C=" << C << " i=" << i << " t=" << t;
                ++seen;
            }
        }
        EXPECT_EQ(seen, count) << "C=" << C;
    };
    check(4, golden::kLaterStreakStarts_C4, golden::kNumLaterStreakStarts_C4);
    check(8, golden::kLaterStreakStarts_C8, golden::kNumLaterStreakStarts_C8);
    check(16, golden::kLaterStreakStarts_C16, golden::kNumLaterStreakStarts_C16);
}

// ----------------------------------------------------------- the properties
// These do not need fixtures: they are the theorem's claims, checked directly
// on this implementation at every configuration the port will use.
TEST(CyclicScheduleTest, EveryCausalPairIsVisitedExactlyOnce) {
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const uint32_t C = golden::kConfigs[k].C;
        const CyclicSchedule s(C);
        std::vector<uint8_t> seen((s.T() + 1u) * (s.T() + 1u), 0);
        for (uint32_t c = 1; c <= C; ++c) {
            for (uint32_t t = 0; t <= s.T(); ++t) {
                const auto p = s.pair(c, t);
                ASSERT_LE(p.j, p.i) << "C=" << C << ": non-causal pair";
                ASSERT_LE(p.i, s.T());
                ASSERT_GE(p.j, 1u);
                auto& cell = seen[p.i * (s.T() + 1u) + p.j];
                EXPECT_EQ(cell, 0u) << "C=" << C << ": pair (" << p.i << "," << p.j
                                    << ") visited twice";
                cell = 1u;
            }
        }
        uint32_t count = 0;
        for (uint32_t i = 1; i <= s.T(); ++i) {
            for (uint32_t j = 1; j <= i; ++j) {
                count += seen[i * (s.T() + 1u) + j];
            }
        }
        EXPECT_EQ(count, s.T() * (s.T() + 1u) / 2u) << "C=" << C;
    }
}

TEST(CyclicScheduleTest, RowsAreDistinctWithinATimestep) {
    // The property one barrier per timestep rests on: no two cores update the
    // same dQ_i at the same timestep.
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const uint32_t C = golden::kConfigs[k].C;
        const CyclicSchedule s(C);
        for (uint32_t t = 0; t <= s.T(); ++t) {
            std::vector<uint8_t> seen(s.T() + 1u, 0);
            for (uint32_t c = 1; c <= C; ++c) {
                auto& cell = seen[s.pair(c, t).i];
                EXPECT_EQ(cell, 0u) << "C=" << C << " t=" << t << ": row reused";
                cell = 1u;
            }
        }
    }
}

TEST(CyclicScheduleTest, StreakTotalIsFiveCMinusThree) {
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const auto& cfg = golden::kConfigs[k];
        const CyclicSchedule s(cfg.C);
        uint32_t total = 0;
        for (uint32_t i = 1; i <= s.T(); ++i) {
            total += static_cast<uint32_t>(streaks(s, i).size());
        }
        EXPECT_EQ(total, 5u * cfg.C - 3u) << "C=" << cfg.C;
        EXPECT_EQ(total, cfg.streak_total);
    }
}

TEST(CyclicScheduleTest, InterStreakSpillsEndOnCoreOneOrTwo) {
    // The endpoint spill property: the lemma two counters rest on. If a large
    // C produced an inter-streak spill away from cores 1 and 2, Algorithm 4
    // would be waiting on a counter nobody publishes.
    for (uint32_t C = 2; C <= 140; ++C) {
        const CyclicSchedule s(C);
        for (uint32_t i = 1; i <= s.T(); ++i) {
            for (uint32_t t = 0; t <= s.T(); ++t) {
                if (!s.is_later_streak_start(i, t)) {
                    continue;
                }
                const uint32_t e = s.spill_endpoint(i, t);
                EXPECT_TRUE(e == 1u || e == 2u) << "C=" << C << " i=" << i << " t=" << t
                                                << ": endpoint " << e;
                EXPECT_EQ(s.endpoint_threshold(i, t), s.prev_active(i, t) + 1u);
            }
        }
    }
}

TEST(CyclicScheduleTest, ConsecutiveConsumersAreSnakeAdjacent) {
    // Folded stride -2 locality, the claim that makes the relay
    // nearest-neighbour.
    for (uint32_t C = 2; C <= 70; ++C) {
        const CyclicSchedule s(C);
        for (uint32_t t = 0; t < s.T(); ++t) {
            for (const auto& m : moves(s, t)) {
                EXPECT_TRUE(snake_adjacent(C, m.src, m.dst))
                    << "C=" << C << " t=" << t << " row " << m.i << ": " << m.src << " -> "
                    << m.dst;
            }
        }
    }
}

TEST(CyclicScheduleTest, EachSnakeEdgeCarriesAtMostOnePacketPerTimestep) {
    // The collision-free lemma, as a logical-edge claim.
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const uint32_t C = golden::kConfigs[k].C;
        const CyclicSchedule s(C);
        for (uint32_t t = 0; t < s.T(); ++t) {
            std::vector<uint8_t> used(C, 0);
            for (const auto& m : moves(s, t)) {
                if (m.src == m.dst) {
                    continue;  // self-transition: no edge
                }
                const uint32_t ka = snake_index_of(C, m.src);
                const uint32_t kb = snake_index_of(C, m.dst);
                const uint32_t edge = ka < kb ? ka : kb;  // edge between k and k+1
                EXPECT_EQ(used[edge], 0u) << "C=" << C << " t=" << t << ": snake edge " << edge
                                          << " carries two packets";
                used[edge] = 1u;
            }
        }
    }
}

TEST(CyclicScheduleTest, PlacementIsNearestNeighborForEveryLadderShape) {
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const auto& cfg = golden::kConfigs[k];
        EXPECT_EQ(cfg.grid_w * cfg.grid_h, cfg.C);
        EXPECT_TRUE(placement_is_nearest_neighbor(cfg.C, cfg.grid_w, cfg.grid_h))
            << "C=" << cfg.C << " on " << cfg.grid_w << "x" << cfg.grid_h;
    }
}

TEST(CyclicScheduleTest, HasLaterActiveAgreesWithTheStreaks) {
    // It decides whether a spill is inter-streak, and so whether an endpoint
    // publishes. Saying "no" too early strands a later reload forever; saying
    // "yes" at a final spill publishes a value nobody waits for.
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const uint32_t C = golden::kConfigs[k].C;
        const CyclicSchedule s(C);
        for (uint32_t i = 1; i <= s.T(); ++i) {
            const auto st = streaks(s, i);
            for (uint32_t t = 0; t <= s.T(); ++t) {
                bool expected = false;
                for (const auto& iv : st) {
                    if (iv.start > t) {
                        expected = true;
                    }
                }
                // At a streak end, "a later streak exists" and "the row is
                // active later" are the same statement.
                if (s.is_active(i, t) && s.streak_at(i, t).end == t) {
                    EXPECT_EQ(s.has_later_active(i, t), expected)
                        << "C=" << C << " i=" << i << " t=" << t;
                }
                bool any_active_later = false;
                for (uint32_t u = t + 1u; u <= s.T(); ++u) {
                    any_active_later = any_active_later || s.is_active(i, u);
                }
                EXPECT_EQ(s.has_later_active(i, t), any_active_later)
                    << "C=" << C << " i=" << i << " t=" << t;
            }
        }
    }
}

TEST(CyclicScheduleTest, ThePapersWorkedEndpointExample) {
    // main.tex's remark: with T = 16, row 15's final streak starts at t = 16
    // on core 2, its preceding spill is at t = 14, so the threshold is 15.
    // Waiting for 16 deadlocks. This is the case a port is most likely to get
    // wrong, and getting it wrong hangs rather than answers incorrectly.
    const CyclicSchedule s(8);
    ASSERT_EQ(s.T(), 16u);
    EXPECT_TRUE(s.is_later_streak_start(15, 16));
    EXPECT_EQ(s.prev_active(15, 16), 14u);
    EXPECT_EQ(s.spill_endpoint(15, 16), 2u);
    EXPECT_EQ(s.endpoint_threshold(15, 16), 15u);
    EXPECT_EQ(s.pair(2, 16).i, 15u);
    EXPECT_EQ(s.pair(2, 16).j, 15u);
}

TEST(CyclicScheduleTest, TheOnlySelfTransitionIsCoreOne) {
    // main.tex, communication contract item 1: the sole self-transition is
    // core 1, column 1 -> T.
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const uint32_t C = golden::kConfigs[k].C;
        const CyclicSchedule s(C);
        uint32_t self_transitions = 0;
        for (uint32_t t = 0; t < s.T(); ++t) {
            for (const auto& m : moves(s, t)) {
                if (m.src != m.dst) {
                    continue;
                }
                EXPECT_EQ(m.src, 1u) << "C=" << C << " t=" << t << " row " << m.i;
                EXPECT_EQ(s.column_at(m.i, t), 1u);
                EXPECT_EQ(s.column_at(m.i, t + 1u), s.T());
                ++self_transitions;
            }
        }
        EXPECT_EQ(self_transitions, 1u) << "C=" << C;
    }
}

TEST(CyclicScheduleTest, ProducerAgreesWithTheForwardItDescribes) {
    // producer(r, u) and next_consumer(i, t) are two views of the same edge:
    // if r loads from a producer p at u, then p forwards to r at u - 1.
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const uint32_t C = golden::kConfigs[k].C;
        const CyclicSchedule s(C);
        for (uint32_t r = 1; r <= C; ++r) {
            for (uint32_t u = 0; u <= s.T(); ++u) {
                const uint32_t i = s.pair(r, u).i;
                const auto p = s.producer(r, u);
                if (p.internal) {
                    ASSERT_GE(u, 1u);
                    EXPECT_EQ(s.next_consumer(i, u - 1u), r)
                        << "C=" << C << " r=" << r << " u=" << u;
                    EXPECT_EQ(s.pair(p.core, u - 1u).i, i);
                } else {
                    EXPECT_EQ(p.core, r);
                    EXPECT_TRUE(u == 0u || !s.is_active(i, u - 1u));
                    EXPECT_TRUE(s.is_streak_start(i, u));
                }
            }
        }
    }
}

TEST(CyclicScheduleTest, EveryCoreChangesColumnExactlyTwice) {
    // Theorem claim 4: every core changes its resident column exactly twice,
    // exactly one core changes at each timestep 1..T, and every change is at a
    // diagonal block.
    for (uint32_t k = 0; k < golden::kNumConfigs; ++k) {
        const uint32_t C = golden::kConfigs[k].C;
        const CyclicSchedule s(C);
        std::vector<uint32_t> changes_at(s.T() + 1u, 0);
        for (uint32_t c = 1; c <= C; ++c) {
            uint32_t changes = 0;
            for (uint32_t t = 1; t <= s.T(); ++t) {
                if (s.pair(c, t).j != s.pair(c, t - 1u).j) {
                    ++changes;
                    ++changes_at[t];
                    const auto p = s.pair(c, t);
                    EXPECT_EQ(p.i, p.j) << "C=" << C << " c=" << c << " t=" << t
                                        << ": column change off the diagonal";
                }
            }
            EXPECT_EQ(changes, 2u) << "C=" << C << " c=" << c;
        }
        for (uint32_t t = 1; t <= s.T(); ++t) {
            EXPECT_EQ(changes_at[t], 1u) << "C=" << C << " t=" << t;
        }
    }
}
