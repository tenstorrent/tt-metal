// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the expanded-K floater layout (ring_joint_floater_layout.hpp): exact tables for the two tuned
// shapes, so a change to the layout is a deliberate one, and the invariants the kernels' protocol relies on.

#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_floater_layout.hpp"

namespace {

using namespace ttnn::operations::transformer::sdpa::ring_joint;

constexpr uint32_t kGridX = 11;
constexpr uint32_t kGridY = 10;
constexpr uint32_t kGatherWorkers = 4;
constexpr uint32_t kRingSize = 8;

// Kimi K3: 24 heads, 2 bands.
constexpr std::array<std::array<uint32_t, 4>, 5> kK3Groups = {
    {{96, 97, 107, 108}, {10, 21, 32, 43}, {54, 65, 76, 87}, {110, 111, 112, 113}, {98, 114, 109, 115}}};
constexpr std::array<uint32_t, 96> kK3Homes = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 12, 13, 14, 15, 16, 17, 18,  19,  20,  22,  23,  24,  25,
    26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45,  46,  47,  48,  49,  50,  51,
    52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71,  72,  73,  74,  75,  77,  78,
    79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 99, 100, 101, 102, 103, 104, 105, 106};
// per home: band, unit, merge iteration, then per piece: floater core, ordinal
constexpr std::array<std::array<uint32_t, 7>, 96> kK3HomeTable = {
    {{0, 0, 2, 96, 1, 98, 5},    {0, 1, 2, 97, 1, 114, 5},   {0, 4, 2, 10, 1, 96, 6},    {0, 5, 2, 21, 1, 97, 6},
     {0, 8, 2, 54, 1, 10, 6},    {0, 9, 2, 65, 1, 21, 6},    {0, 12, 2, 110, 1, 54, 6},  {0, 13, 2, 111, 1, 65, 6},
     {0, 16, 2, 98, 1, 110, 6},  {0, 17, 2, 114, 1, 111, 6}, {0, 2, 2, 107, 1, 109, 5},  {0, 3, 2, 108, 1, 115, 5},
     {0, 6, 2, 32, 1, 107, 6},   {0, 7, 2, 43, 1, 108, 6},   {0, 10, 2, 76, 1, 32, 6},   {0, 11, 2, 87, 1, 43, 6},
     {0, 14, 2, 112, 1, 76, 6},  {0, 15, 2, 113, 1, 87, 6},  {0, 18, 2, 109, 1, 112, 6}, {0, 19, 2, 115, 1, 113, 6},
     {0, 20, 2, 96, 2, 98, 6},   {0, 21, 2, 97, 2, 114, 6},  {0, 24, 2, 10, 2, 96, 7},   {0, 25, 2, 21, 2, 97, 7},
     {0, 28, 2, 54, 2, 10, 7},   {0, 29, 2, 65, 2, 21, 7},   {0, 32, 2, 110, 2, 54, 7},  {0, 33, 2, 111, 2, 65, 7},
     {0, 36, 2, 98, 2, 110, 7},  {0, 37, 2, 114, 2, 111, 7}, {0, 22, 2, 107, 2, 109, 6}, {0, 23, 2, 108, 2, 115, 6},
     {0, 26, 2, 32, 2, 107, 7},  {0, 27, 2, 43, 2, 108, 7},  {0, 30, 2, 76, 2, 32, 7},   {0, 31, 2, 87, 2, 43, 7},
     {0, 34, 2, 112, 2, 76, 7},  {0, 35, 2, 113, 2, 87, 7},  {0, 38, 2, 109, 2, 112, 7}, {0, 39, 2, 115, 2, 113, 7},
     {0, 40, 2, 96, 3, 98, 7},   {0, 41, 2, 97, 3, 114, 7},  {0, 44, 2, 10, 3, 96, 8},   {0, 45, 2, 21, 3, 97, 8},
     {1, 0, 2, 54, 3, 10, 8},    {1, 1, 2, 65, 3, 21, 8},    {1, 4, 2, 110, 3, 54, 8},   {1, 5, 2, 111, 3, 65, 8},
     {1, 8, 2, 98, 3, 110, 8},   {1, 9, 2, 114, 3, 111, 8},  {0, 42, 2, 107, 3, 109, 7}, {0, 43, 2, 108, 3, 115, 7},
     {0, 46, 2, 32, 3, 107, 8},  {0, 47, 2, 43, 3, 108, 8},  {1, 2, 2, 76, 3, 32, 8},    {1, 3, 2, 87, 3, 43, 8},
     {1, 6, 2, 112, 3, 76, 8},   {1, 7, 2, 113, 3, 87, 8},   {1, 10, 2, 109, 3, 112, 8}, {1, 11, 2, 115, 3, 113, 8},
     {1, 12, 2, 96, 4, 98, 8},   {1, 13, 2, 97, 4, 114, 8},  {1, 16, 2, 10, 4, 96, 9},   {1, 17, 2, 21, 4, 97, 9},
     {1, 20, 2, 54, 4, 10, 9},   {1, 21, 2, 65, 4, 21, 9},   {1, 24, 2, 110, 4, 54, 9},  {1, 25, 2, 111, 4, 65, 9},
     {1, 28, 2, 98, 4, 110, 9},  {1, 29, 2, 114, 4, 111, 9}, {1, 14, 2, 107, 4, 109, 8}, {1, 15, 2, 108, 4, 115, 8},
     {1, 18, 2, 32, 4, 107, 9},  {1, 19, 2, 43, 4, 108, 9},  {1, 22, 2, 76, 4, 32, 9},   {1, 23, 2, 87, 4, 43, 9},
     {1, 26, 2, 112, 4, 76, 9},  {1, 27, 2, 113, 4, 87, 9},  {1, 30, 2, 109, 4, 112, 9}, {1, 31, 2, 115, 4, 113, 9},
     {1, 32, 2, 96, 5, 98, 9},   {1, 33, 2, 97, 5, 114, 9},  {1, 36, 2, 10, 5, 96, 10},  {1, 37, 2, 21, 5, 97, 10},
     {1, 40, 2, 54, 5, 10, 10},  {1, 41, 2, 65, 5, 21, 10},  {1, 44, 2, 110, 5, 54, 10}, {1, 45, 2, 111, 5, 65, 10},
     {1, 34, 2, 107, 5, 109, 9}, {1, 35, 2, 108, 5, 115, 9}, {1, 38, 2, 32, 5, 107, 10}, {1, 39, 2, 43, 5, 108, 10},
     {1, 42, 2, 76, 5, 32, 10},  {1, 43, 2, 87, 5, 43, 10},  {1, 46, 2, 112, 5, 76, 10}, {1, 47, 2, 113, 5, 87, 10}}};
// per group: visited blocks in run order (piece 0 then piece 1)
constexpr std::array<std::array<uint32_t, 10>, 5> kK3VisitBlocks = {
    {{0, 5, 10, 15, 20, 1, 6, 11, 16, 21},
     {1, 6, 11, 16, 21, 2, 7, 12, 17, 22},
     {2, 7, 12, 17, 22, 3, 8, 13, 18, 23},
     {3, 8, 13, 18, 23, 4, 9, 14, 19},
     {4, 9, 14, 19, 0, 5, 10, 15, 20}}};
constexpr std::array<uint32_t, 5> kK3VisitCounts = {10, 10, 10, 9, 9};
constexpr std::array<uint32_t, 5> kK3FirstPieceVisits = {5, 5, 5, 5, 4};
// Kimi K2.7: 16 heads, 3 bands.
constexpr std::array<std::array<uint32_t, 4>, 5> kK27Groups = {
    {{96, 97, 107, 108}, {10, 21, 32, 43}, {54, 65, 76, 87}, {110, 111, 112, 113}, {98, 114, 109, 115}}};
constexpr std::array<uint32_t, 96> kK27Homes = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 12, 13, 14, 15, 16, 17, 18,  19,  20,  22,  23,  24,  25,
    26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45,  46,  47,  48,  49,  50,  51,
    52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71,  72,  73,  74,  75,  77,  78,
    79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 99, 100, 101, 102, 103, 104, 105, 106};
// per home: band, unit, merge iteration, then per piece: floater core, ordinal
constexpr std::array<std::array<uint32_t, 7>, 96> kK27HomeTable = {
    {{0, 0, 2, 96, 1, 98, 5},    {0, 1, 2, 97, 1, 114, 5},   {0, 4, 2, 10, 1, 96, 6},    {0, 5, 2, 21, 1, 97, 6},
     {0, 8, 2, 54, 1, 10, 6},    {0, 9, 2, 65, 1, 21, 6},    {0, 12, 2, 110, 1, 54, 6},  {0, 13, 2, 111, 1, 65, 6},
     {0, 16, 2, 98, 1, 110, 6},  {0, 17, 2, 114, 1, 111, 6}, {0, 2, 2, 107, 1, 109, 5},  {0, 3, 2, 108, 1, 115, 5},
     {0, 6, 2, 32, 1, 107, 6},   {0, 7, 2, 43, 1, 108, 6},   {0, 10, 2, 76, 1, 32, 6},   {0, 11, 2, 87, 1, 43, 6},
     {0, 14, 2, 112, 1, 76, 6},  {0, 15, 2, 113, 1, 87, 6},  {0, 18, 2, 109, 1, 112, 6}, {0, 19, 2, 115, 1, 113, 6},
     {0, 20, 3, 96, 2, 98, 6},   {0, 21, 3, 97, 2, 114, 6},  {0, 24, 3, 10, 2, 96, 7},   {0, 25, 3, 21, 2, 97, 7},
     {0, 28, 3, 54, 2, 10, 7},   {0, 29, 3, 65, 2, 21, 7},   {1, 0, 3, 110, 2, 54, 7},   {1, 1, 3, 111, 2, 65, 7},
     {1, 4, 3, 98, 2, 110, 7},   {1, 5, 3, 114, 2, 111, 7},  {0, 22, 3, 107, 2, 109, 6}, {0, 23, 3, 108, 2, 115, 6},
     {0, 26, 3, 32, 2, 107, 7},  {0, 27, 3, 43, 2, 108, 7},  {0, 30, 3, 76, 2, 32, 7},   {0, 31, 3, 87, 2, 43, 7},
     {1, 2, 3, 112, 2, 76, 7},   {1, 3, 3, 113, 2, 87, 7},   {1, 6, 3, 109, 2, 112, 7},  {1, 7, 3, 115, 2, 113, 7},
     {1, 8, 4, 96, 3, 98, 7},    {1, 9, 4, 97, 3, 114, 7},   {1, 12, 4, 10, 3, 96, 8},   {1, 13, 4, 21, 3, 97, 8},
     {1, 16, 4, 54, 3, 10, 8},   {1, 17, 4, 65, 3, 21, 8},   {1, 20, 4, 110, 3, 54, 8},  {1, 21, 4, 111, 3, 65, 8},
     {1, 24, 4, 98, 3, 110, 8},  {1, 25, 4, 114, 3, 111, 8}, {1, 10, 4, 107, 3, 109, 7}, {1, 11, 4, 108, 3, 115, 7},
     {1, 14, 4, 32, 3, 107, 8},  {1, 15, 4, 43, 3, 108, 8},  {1, 18, 4, 76, 3, 32, 8},   {1, 19, 4, 87, 3, 43, 8},
     {1, 22, 4, 112, 3, 76, 8},  {1, 23, 4, 113, 3, 87, 8},  {1, 26, 4, 109, 3, 112, 8}, {1, 27, 4, 115, 3, 113, 8},
     {1, 28, 5, 96, 4, 98, 8},   {1, 29, 5, 97, 4, 114, 8},  {2, 0, 5, 10, 4, 96, 9},    {2, 1, 5, 21, 4, 97, 9},
     {2, 4, 5, 54, 4, 10, 9},    {2, 5, 5, 65, 4, 21, 9},    {2, 8, 5, 110, 4, 54, 9},   {2, 9, 5, 111, 4, 65, 9},
     {2, 12, 5, 98, 4, 110, 9},  {2, 13, 5, 114, 4, 111, 9}, {1, 30, 5, 107, 4, 109, 8}, {1, 31, 5, 108, 4, 115, 8},
     {2, 2, 5, 32, 4, 107, 9},   {2, 3, 5, 43, 4, 108, 9},   {2, 6, 5, 76, 4, 32, 9},    {2, 7, 5, 87, 4, 43, 9},
     {2, 10, 5, 112, 4, 76, 9},  {2, 11, 5, 113, 4, 87, 9},  {2, 14, 5, 109, 4, 112, 9}, {2, 15, 5, 115, 4, 113, 9},
     {2, 16, 5, 96, 5, 98, 9},   {2, 17, 5, 97, 5, 114, 9},  {2, 20, 5, 10, 5, 96, 10},  {2, 21, 5, 21, 5, 97, 10},
     {2, 24, 5, 54, 5, 10, 10},  {2, 25, 5, 65, 5, 21, 10},  {2, 28, 5, 110, 5, 54, 10}, {2, 29, 5, 111, 5, 65, 10},
     {2, 18, 5, 107, 5, 109, 9}, {2, 19, 5, 108, 5, 115, 9}, {2, 22, 5, 32, 5, 107, 10}, {2, 23, 5, 43, 5, 108, 10},
     {2, 26, 5, 76, 5, 32, 10},  {2, 27, 5, 87, 5, 43, 10},  {2, 30, 5, 112, 5, 76, 10}, {2, 31, 5, 113, 5, 87, 10}}};
// per group: visited blocks in run order (piece 0 then piece 1)
constexpr std::array<std::array<uint32_t, 10>, 5> kK27VisitBlocks = {
    {{0, 5, 10, 15, 20, 1, 6, 11, 16, 21},
     {1, 6, 11, 16, 21, 2, 7, 12, 17, 22},
     {2, 7, 12, 17, 22, 3, 8, 13, 18, 23},
     {3, 8, 13, 18, 23, 4, 9, 14, 19},
     {4, 9, 14, 19, 0, 5, 10, 15, 20}}};
constexpr std::array<uint32_t, 5> kK27VisitCounts = {10, 10, 10, 9, 9};
constexpr std::array<uint32_t, 5> kK27FirstPieceVisits = {5, 5, 5, 5, 4};

struct Golden {
    uint32_t heads, bands;
    const std::array<std::array<uint32_t, 4>, 5>& groups;
    const std::array<uint32_t, 96>& homes;
    const std::array<std::array<uint32_t, 7>, 96>& home_table;
    const std::array<std::array<uint32_t, 10>, 5>& visit_blocks;  // the first visit_counts[f] of each row
    const std::array<uint32_t, 5>& visit_counts;
    const std::array<uint32_t, 5>& first_piece_visits;
};

const std::array<Golden, 2> kGoldens = {{
    {24, 2, kK3Groups, kK3Homes, kK3HomeTable, kK3VisitBlocks, kK3VisitCounts, kK3FirstPieceVisits},
    {16, 3, kK27Groups, kK27Homes, kK27HomeTable, kK27VisitBlocks, kK27VisitCounts, kK27FirstPieceVisits},
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
        ASSERT_EQ(layout.num_cores_all, kGridX * kGridY + kGridY - kGatherWorkers);
        ASSERT_EQ(layout.groups.size(), golden.groups.size());
        for (uint32_t f = 0; f < golden.groups.size(); ++f) {
            EXPECT_EQ(layout.groups[f], golden.groups[f]) << "group " << f;
            ASSERT_EQ(layout.visits[f].size(), golden.visit_counts[f]) << "group " << f;
            for (uint32_t v = 0; v < layout.visits[f].size(); ++v) {
                const FloaterVisit& visit = layout.visits[f][v];
                const uint32_t piece = v < golden.first_piece_visits[f] ? 0 : 1;
                EXPECT_EQ(visit.block, golden.visit_blocks[f][v]) << "group " << f << " visit " << v;
                EXPECT_EQ(visit.piece, piece) << "group " << f << " visit " << v;
                EXPECT_EQ(visit.iter, schedule.iters[piece]) << "group " << f << " visit " << v;
            }
        }
        for (uint32_t i = 0; i < golden.homes.size(); ++i) {
            const uint32_t home = golden.homes[i];
            const auto& row = golden.home_table[i];
            SCOPED_TRACE(::testing::Message() << "home core " << home);
            EXPECT_EQ(layout.home_band[home], row[0]);
            EXPECT_EQ(layout.home_unit[home], row[1]);
            EXPECT_EQ(layout.home_merge_iter[home], row[2]);
            for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
                const HomePiece& p = layout.home_pieces[home][piece];
                EXPECT_EQ(p.floater, row[3 + 2 * piece]) << "piece " << piece;
                EXPECT_EQ(p.ordinal, row[4 + 2 * piece]) << "piece " << piece;
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
        // Every (home block, piece) is visited once; homes and floaters are disjoint; no home sits in the last column
        // or the gather column.
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
            EXPECT_LT(layout.logical_xy(core).first + 1, kGridX) << "home core " << core;
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
