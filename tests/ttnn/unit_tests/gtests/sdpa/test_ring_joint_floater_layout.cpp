// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the expanded-K floater layout (ring_joint_floater_layout.hpp): the two tuned shapes' floater
// groups exactly, their homes and visits against the layout's documented rules, and the invariants the kernels'
// protocol and the program's K delivery rely on.

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_floater_layout.hpp"

namespace {

using namespace ttnn::operations::transformer::sdpa::ring_joint;

constexpr uint32_t kGridX = 11;
constexpr uint32_t kGridY = 10;
constexpr uint32_t kGatherWorkers = 4;
constexpr uint32_t kRingSize = 8;

// The floater groups, exactly: the dense layout's idle cores (core y * 11 + x) and the gather column's cores below its
// workers (110 + y - 4), multicast flag per group.
struct Golden {
    uint32_t heads, bands;
    std::array<std::array<uint32_t, 4>, 5> groups;
    std::array<bool, 5> multicast;
};

const std::array<Golden, 2> kGoldens = {{
    // Kimi K3: 24 heads, 2 bands of 5 rows; idle (4..10, 4) and (4..10, 9).
    {24,
     2,
     {{{48, 49, 50, 51}, {52, 53, 54, 110}, {103, 104, 105, 106}, {107, 108, 109, 115}, {111, 112, 113, 114}}},
     {true, true, true, true, true}},
    // Kimi K2.7: 16 heads, 3 bands of 3 rows; idle (10, 2), (10, 5), (10, 8) and row 9.
    {16,
     3,
     {{{99, 100, 101, 102}, {103, 104, 105, 106}, {107, 108, 109, 115}, {110, 111, 112, 113}, {32, 65, 98, 114}}},
     {true, true, true, true, false}},
}};

FloaterSchedule schedule_of(const Golden& golden) {
    const std::optional<FloaterSchedule> schedule = floater_layout_schedule(golden.heads, golden.bands);
    EXPECT_TRUE(schedule.has_value());
    return schedule.value_or(FloaterSchedule{});
}

FloaterLayout layout_of(const Golden& golden) {
    return build_floater_layout(
        schedule_of(golden), kGridX, kGridY, golden.heads, golden.bands, kGatherWorkers, golden.heads * 2, kRingSize);
}

TEST(RingJointFloaterLayout, MatchesTheTunedTables) {
    for (const Golden& golden : kGoldens) {
        SCOPED_TRACE(::testing::Message() << "heads=" << golden.heads << " bands=" << golden.bands);
        const FloaterLayout layout = layout_of(golden);
        const FloaterSchedule schedule = schedule_of(golden);
        const uint32_t total_q_chunks = golden.heads * 2;
        const uint32_t rows_per_band = (total_q_chunks + kGridX - 1) / kGridX;
        ASSERT_EQ(layout.num_cores_all, kGridX * kGridY + kGridY - kGatherWorkers);
        ASSERT_EQ(layout.groups.size(), golden.groups.size());
        for (uint32_t f = 0; f < golden.groups.size(); ++f) {
            EXPECT_EQ(layout.groups[f], golden.groups[f]) << "group " << f;
            EXPECT_EQ(layout.group_multicast[f], golden.multicast[f]) << "group " << f;
        }
        // Homes: where the program without floaters puts them.
        for (uint32_t core = 0; core < kGridX * kGridY; ++core) {
            const uint32_t band = core / kGridX / rows_per_band;
            const uint32_t unit = core / kGridX % rows_per_band * kGridX + core % kGridX;
            const bool home = band < golden.bands && unit < total_q_chunks;
            EXPECT_EQ(layout.home_unit[core], home ? unit : total_q_chunks) << "core " << core;
            if (home) {
                EXPECT_EQ(layout.home_band[core], band) << "core " << core;
            }
        }
        // Visits round-robin over the groups: piece p of block g is visit n = p * blocks + g, run by group n % groups
        // as its visit n / groups (ordinal n / groups + 1); a first piece merges at merge_after of that visit index.
        for (uint32_t g = 0; g < kFloaterHomeBlocks; ++g) {
            for (uint32_t k = 0; k < kFloaterGroupCores; ++k) {
                const uint32_t home = layout.block_core(g, k);
                EXPECT_EQ(layout.home_band[home], g / (golden.heads / 2));
                EXPECT_EQ(layout.home_unit[home], kFloaterGroupCores * (g % (golden.heads / 2)) + k);
                EXPECT_EQ(layout.home_merge_iter[home], schedule.merge_after[g / kFloaterGroups]) << "block " << g;
                for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
                    const uint32_t n = piece * kFloaterHomeBlocks + g;
                    const HomePiece& p = layout.home_pieces[home][piece];
                    EXPECT_EQ(p.floater, golden.groups[n % kFloaterGroups][k]) << "block " << g << " piece " << piece;
                    EXPECT_EQ(p.ordinal, n / kFloaterGroups + 1) << "block " << g << " piece " << piece;
                    const FloaterVisit& visit = layout.visits[n % kFloaterGroups][n / kFloaterGroups];
                    EXPECT_EQ(visit.block, g);
                    EXPECT_EQ(visit.piece, piece);
                    EXPECT_EQ(visit.iter, schedule.iters[piece]);
                }
            }
        }
    }
}

TEST(RingJointFloaterLayout, HoldsTheProtocolsInvariants) {
    for (const Golden& golden : kGoldens) {
        SCOPED_TRACE(::testing::Message() << "heads=" << golden.heads << " bands=" << golden.bands);
        const FloaterLayout layout = layout_of(golden);
        const uint32_t total_q_chunks = golden.heads * 2;
        std::set<uint32_t> floater_cores;
        for (uint32_t f = 0; f < layout.groups.size(); ++f) {
            for (uint32_t k = 0; k < kFloaterGroupCores; ++k) {
                const uint32_t core = layout.groups[f][k];
                EXPECT_TRUE(floater_cores.insert(core).second) << "core " << core << " in two groups";
                EXPECT_EQ(layout.group_of[core], f);
                EXPECT_EQ(layout.k_of[core], k);
                EXPECT_EQ(layout.home_unit[core], total_q_chunks) << "a floater owns no unit";
            }
            // Visits: first pieces, then second pieces; within the semaphore's 31 ordinals and the per-iteration cap.
            const auto& visits = layout.visits[f];
            EXPECT_LT(visits.size(), 32u);
            std::array<uint32_t, kDonatedPieces> per_piece{};
            for (uint32_t v = 0; v < visits.size(); ++v) {
                EXPECT_TRUE(v == 0 || visits[v - 1].piece <= visits[v].piece);
                ++per_piece[visits[v].piece];
            }
            for (uint32_t count : per_piece) {
                EXPECT_LE(count, kFloaterMaxVisitsPerIter);
            }
        }
        EXPECT_EQ(floater_cores.size(), kFloaterCores);
        // Every (home block, piece) is visited once; homes and floaters are disjoint; no home sits in the gather
        // column.
        std::set<std::pair<uint32_t, uint32_t>> visited;
        for (const auto& visits : layout.visits) {
            for (const FloaterVisit& visit : visits) {
                EXPECT_TRUE(visited.insert({visit.block, visit.piece}).second);
            }
        }
        EXPECT_EQ(visited.size(), kFloaterHomeBlocks * kDonatedPieces);
        std::set<std::pair<uint32_t, uint32_t>> ordinals;
        uint32_t homes = 0;
        for (uint32_t core = 0; core < layout.num_cores_all; ++core) {
            if (layout.home_unit[core] == total_q_chunks) {
                continue;
            }
            ++homes;
            EXPECT_FALSE(layout.is_floater(core));
            EXPECT_LT(layout.logical_xy(core).first, kGridX) << "home core " << core;
            EXPECT_EQ(layout.unit_home[layout.home_band[core] * total_q_chunks + layout.home_unit[core]], core);
            for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
                const HomePiece& p = layout.home_pieces[core][piece];
                EXPECT_TRUE(layout.is_floater(p.floater));
                // Each (floater, ordinal) is one visit, of one home's piece.
                EXPECT_TRUE(ordinals.insert({p.floater, p.ordinal}).second);
            }
        }
        EXPECT_EQ(homes, kFloaterHomeBlocks * kFloaterGroupCores);
    }
}

// What the program's K delivery relies on: every row of homes is one band (its homes cut the same chunks) and starts
// at column 0 with no floater among its homes (the row multicast stops at its last home); a multicast group fills its
// rectangle; groups without multicast come last.
TEST(RingJointFloaterLayout, KeepsTheKDeliveryShapes) {
    for (const Golden& golden : kGoldens) {
        SCOPED_TRACE(::testing::Message() << "heads=" << golden.heads << " bands=" << golden.bands);
        const FloaterLayout layout = layout_of(golden);
        const uint32_t total_q_chunks = golden.heads * 2;
        for (uint32_t y = 0; y < kGridY; ++y) {
            bool past_homes = false;
            for (uint32_t x = 0; x < kGridX; ++x) {
                const uint32_t core = y * kGridX + x;
                const bool home = layout.home_unit[core] < total_q_chunks;
                EXPECT_FALSE(home && past_homes) << "row " << y << " column " << x;
                past_homes = past_homes || !home;
                if (home) {
                    EXPECT_EQ(layout.home_band[core], layout.home_band[y * kGridX]) << "row " << y;
                } else {
                    EXPECT_TRUE(layout.is_floater(core)) << "idle core " << core;
                }
            }
        }
        bool seen_unicast = false;
        for (uint32_t f = 0; f < layout.groups.size(); ++f) {
            EXPECT_FALSE(layout.group_multicast[f] && seen_unicast) << "group " << f << " after one without multicast";
            seen_unicast = seen_unicast || !layout.group_multicast[f];
            if (!layout.group_multicast[f]) {
                continue;
            }
            uint32_t x0 = kGridX + 1, y0 = kGridY, x1 = 0, y1 = 0;
            for (const uint32_t core : layout.groups[f]) {
                const auto [x, y] = layout.logical_xy(core);
                x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
            }
            EXPECT_EQ((x1 - x0 + 1) * (y1 - y0 + 1), kFloaterGroupCores) << "group " << f;
        }
    }
}

// Kimi K3 / K2.7 MLA on the Galaxy 8x4: expanded K at 5k chunks (640 Q rows per device) and 2k (256), every fidelity.
// The floaters were measured slower or no faster than none at 512k cached tokens on exactly the shapes marked false.
TEST(RingJointFloaterLayout, ComputeToTransferMatchesTheGalaxyMeasurements) {
    // ring_links 4: the torus (two links each way); 2: an unwrapped line, whose end devices receive from one side.
    struct Measured {
        uint32_t heads, q_rows, fidelity_multiplier, ring_links;
        bool floaters_faster;
    };
    const std::array<Measured, 22> measured = {{
        {24, 640, 1, 4, false}, {24, 640, 2, 4, true},  {24, 640, 3, 4, true},  {24, 640, 4, 4, true},
        {24, 256, 1, 4, false}, {24, 256, 2, 4, false}, {24, 256, 3, 4, false}, {24, 256, 4, 4, true},
        {16, 640, 1, 4, false}, {16, 640, 2, 4, true},  {16, 640, 3, 4, true},  {16, 640, 4, 4, true},
        {16, 256, 1, 4, false}, {16, 256, 2, 4, false}, {16, 256, 3, 4, false}, {16, 256, 4, 4, true},
        {24, 640, 2, 2, false}, {24, 640, 4, 2, true},  {24, 256, 4, 2, false}, {16, 640, 2, 2, false},
        {16, 640, 4, 2, true},  {16, 256, 4, 2, false},
    }};
    constexpr double kBfp8BytesPerElement = 1088.0 / 1024.0;
    for (const Measured& m : measured) {
        const double ratio = floater_compute_to_transfer({
            .heads = m.heads,
            .q_rows = m.q_rows,
            .qk_dim = 192,
            .v_dim = 128,
            .expand_in = 512,
            .expand_out = 128,
            .fidelity_multiplier = m.fidelity_multiplier,
            .bytes_per_token = (576.0 + m.heads * 128.0) * kBfp8BytesPerElement,
            .cores = kGridX * kGridY,
            .ring_links = m.ring_links,
        });
        EXPECT_EQ(ratio >= kFloaterMinComputeToTransfer, m.floaters_faster)
            << "heads=" << m.heads << " q_rows=" << m.q_rows << " fidelity x" << m.fidelity_multiplier
            << " ring_links=" << m.ring_links << " ratio=" << ratio;
    }
}

TEST(RingJointFloaterLayout, TakesAScheduleFromValues) {
    EXPECT_FALSE(floater_schedule_from_values({}).has_value());
    const FloaterSchedule k3 = floater_layout_schedule(24, 2).value();
    std::vector<uint32_t> values = {k3.iters[0], k3.iters[1], k3.fracs[0], k3.fracs[1], k3.donate_tail ? 1u : 0u};
    values.insert(values.end(), std::begin(k3.merge_after), std::end(k3.merge_after));
    const std::optional<FloaterSchedule> schedule = floater_schedule_from_values(values);
    ASSERT_TRUE(schedule.has_value());
    for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
        EXPECT_EQ(schedule->iters[piece], k3.iters[piece]);
        EXPECT_EQ(schedule->fracs[piece], k3.fracs[piece]);
    }
    EXPECT_EQ(schedule->donate_tail, k3.donate_tail);
    for (uint32_t v = 0; v < kFloaterFirstPieceVisits; ++v) {
        EXPECT_EQ(schedule->merge_after[v], k3.merge_after[v]);
    }
    EXPECT_TRUE(floater_schedule_from_values({1, 3, 1843, 1536, 1, 3, 4, 5, 6, 6})->donate_tail);
}

// What output-spec inference checks before the op's validation, without a TT_FATAL.
TEST(RingJointFloaterLayout, TellsWellFormedScheduleValues) {
    EXPECT_TRUE(floater_schedule_values_well_formed({}));
    EXPECT_TRUE(floater_schedule_values_well_formed({2, 4, 1075, 2765, 0, 3, 3, 3, 3, 3}));
    EXPECT_FALSE(floater_schedule_values_well_formed({2, 4, 1075, 2765, 0, 3, 3, 3, 3}));
    EXPECT_FALSE(floater_schedule_values_well_formed({2, 4, 1075, 2765, 2, 3, 3, 3, 3, 3}));
}

// floater_schedule_values_well_formed is what output-spec inference checks instead of calling the parser.
TEST(RingJointFloaterLayout, WellFormedAgreesWithTheParser) {
    const std::vector<std::vector<uint32_t>> samples = {
        {},
        {2, 4, 1075, 2765, 0, 3, 3, 3, 3, 3},
        {2, 4, 1075, 2765, 1, 3, 3, 3, 3, 3},
        {2, 4, 1075, 2765, 0, 3, 3, 3, 3},
        {2, 4, 1075, 2765, 0, 3, 3, 3, 3, 3, 3},
        {2, 4, 1075, 2765, 2, 3, 3, 3, 3, 3},
        {1}};
    for (const auto& values : samples) {
        bool parsed = true;
        try {
            floater_schedule_from_values(values);
        } catch (const std::exception&) {
            parsed = false;
        }
        EXPECT_EQ(floater_schedule_values_well_formed(values), parsed) << values.size() << " values";
    }
}

TEST(RingJointFloaterLayout, RejectsMalformedScheduleValues) {
    EXPECT_ANY_THROW(floater_schedule_from_values({2, 4, 1075, 2765, 0, 3, 3, 3, 3}));
    EXPECT_ANY_THROW(floater_schedule_from_values({2, 4, 1075, 2765, 0, 3, 3, 3, 3, 3, 3}));
    EXPECT_ANY_THROW(floater_schedule_from_values({2, 4, 1075, 2765, 2, 3, 3, 3, 3, 3}));
}

// 25 heads x 2 bands also gives 24 head-pair blocks by integer division, leaving head 24 without a home.
TEST(RingJointFloaterLayout, RefusesAShapeItCannotCoverWhole) {
    const FloaterSchedule k3 = floater_layout_schedule(24, 2).value();
    EXPECT_ANY_THROW(build_floater_layout(k3, kGridX, kGridY, 25, 2, kGatherWorkers, 50, kRingSize));
    EXPECT_ANY_THROW(build_floater_layout(k3, kGridX, kGridY, 24, 3, kGatherWorkers, 48, kRingSize));
}

// 12 heads x 4 bands also fills the 24 home blocks, but the exact layout tables do not cover it.
TEST(RingJointFloaterLayout, RunsOnlyOnTheCheckedLayouts) {
    for (uint32_t heads : {8u, 12u, 16u, 24u, 25u, 48u}) {
        for (uint32_t bands = 1; bands <= kKSplitMaxCount; ++bands) {
            const bool checked = (heads == 24 && bands == 2) || (heads == 16 && bands == 3);
            EXPECT_EQ(floater_layout_supported(heads, bands), checked) << "heads=" << heads << " bands=" << bands;
        }
    }
}

TEST(RingJointFloaterLayout, NoFloaterLayoutKeepsTheDefaults) {
    const FloaterLayout layout = no_floater_layout(kGridX, kGridX * kGridY, 48);
    EXPECT_EQ(layout.num_cores_all, kGridX * kGridY);
    EXPECT_TRUE(layout.groups.empty());
    for (uint32_t core = 0; core < layout.num_cores_all; ++core) {
        EXPECT_FALSE(layout.is_floater(core));
        EXPECT_EQ(layout.home_unit[core], 48u);
        EXPECT_EQ(layout.home_band[core], 0u);
    }
}

}  // namespace
