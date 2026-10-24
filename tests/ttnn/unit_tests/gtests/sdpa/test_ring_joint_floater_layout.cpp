// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the expanded-K floater layout (ring_joint_floater_layout.hpp): exact tables for the two tuned
// shapes, so a change to the layout is a deliberate one, and the invariants the kernels' protocol relies on.

#include <array>
#include <cstdint>
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
// per home: band, unit, merge iteration, then per piece: slot, floater core, ordinal
constexpr std::array<std::array<uint32_t, 9>, 96> kK3HomeTable = {
    {{0, 0, 3, 0, 96, 1, 32, 98, 5},     {0, 1, 3, 2, 97, 1, 34, 114, 5},    {0, 4, 3, 8, 10, 1, 1, 96, 6},
     {0, 5, 3, 10, 21, 1, 3, 97, 6},     {0, 8, 3, 16, 54, 1, 9, 10, 6},     {0, 9, 3, 18, 65, 1, 11, 21, 6},
     {0, 12, 3, 24, 110, 1, 17, 54, 6},  {0, 13, 3, 26, 111, 1, 19, 65, 6},  {0, 16, 3, 32, 98, 1, 25, 110, 6},
     {0, 17, 3, 34, 114, 1, 27, 111, 6}, {0, 2, 3, 4, 107, 1, 36, 109, 5},   {0, 3, 3, 6, 108, 1, 38, 115, 5},
     {0, 6, 3, 12, 32, 1, 5, 107, 6},    {0, 7, 3, 14, 43, 1, 7, 108, 6},    {0, 10, 3, 20, 76, 1, 13, 32, 6},
     {0, 11, 3, 22, 87, 1, 15, 43, 6},   {0, 14, 3, 28, 112, 1, 21, 76, 6},  {0, 15, 3, 30, 113, 1, 23, 87, 6},
     {0, 18, 3, 36, 109, 1, 29, 112, 6}, {0, 19, 3, 38, 115, 1, 31, 113, 6}, {0, 20, 3, 1, 96, 2, 33, 98, 6},
     {0, 21, 3, 3, 97, 2, 35, 114, 6},   {0, 24, 3, 9, 10, 2, 0, 96, 7},     {0, 25, 3, 11, 21, 2, 2, 97, 7},
     {0, 28, 3, 17, 54, 2, 8, 10, 7},    {0, 29, 3, 19, 65, 2, 10, 21, 7},   {0, 32, 3, 25, 110, 2, 16, 54, 7},
     {0, 33, 3, 27, 111, 2, 18, 65, 7},  {0, 36, 3, 33, 98, 2, 24, 110, 7},  {0, 37, 3, 35, 114, 2, 26, 111, 7},
     {0, 22, 3, 5, 107, 2, 37, 109, 6},  {0, 23, 3, 7, 108, 2, 39, 115, 6},  {0, 26, 3, 13, 32, 2, 4, 107, 7},
     {0, 27, 3, 15, 43, 2, 6, 108, 7},   {0, 30, 3, 21, 76, 2, 12, 32, 7},   {0, 31, 3, 23, 87, 2, 14, 43, 7},
     {0, 34, 3, 29, 112, 2, 20, 76, 7},  {0, 35, 3, 31, 113, 2, 22, 87, 7},  {0, 38, 3, 37, 109, 2, 28, 112, 7},
     {0, 39, 3, 39, 115, 2, 30, 113, 7}, {0, 40, 3, 0, 96, 3, 32, 98, 7},    {0, 41, 3, 2, 97, 3, 34, 114, 7},
     {0, 44, 3, 8, 10, 3, 1, 96, 8},     {0, 45, 3, 10, 21, 3, 3, 97, 8},    {1, 0, 3, 16, 54, 3, 9, 10, 8},
     {1, 1, 3, 18, 65, 3, 11, 21, 8},    {1, 4, 3, 24, 110, 3, 17, 54, 8},   {1, 5, 3, 26, 111, 3, 19, 65, 8},
     {1, 8, 3, 32, 98, 3, 25, 110, 8},   {1, 9, 3, 34, 114, 3, 27, 111, 8},  {0, 42, 3, 4, 107, 3, 36, 109, 7},
     {0, 43, 3, 6, 108, 3, 38, 115, 7},  {0, 46, 3, 12, 32, 3, 5, 107, 8},   {0, 47, 3, 14, 43, 3, 7, 108, 8},
     {1, 2, 3, 20, 76, 3, 13, 32, 8},    {1, 3, 3, 22, 87, 3, 15, 43, 8},    {1, 6, 3, 28, 112, 3, 21, 76, 8},
     {1, 7, 3, 30, 113, 3, 23, 87, 8},   {1, 10, 3, 36, 109, 3, 29, 112, 8}, {1, 11, 3, 38, 115, 3, 31, 113, 8},
     {1, 12, 3, 1, 96, 4, 33, 98, 8},    {1, 13, 3, 3, 97, 4, 35, 114, 8},   {1, 16, 3, 9, 10, 4, 0, 96, 9},
     {1, 17, 3, 11, 21, 4, 2, 97, 9},    {1, 20, 3, 17, 54, 4, 8, 10, 9},    {1, 21, 3, 19, 65, 4, 10, 21, 9},
     {1, 24, 3, 25, 110, 4, 16, 54, 9},  {1, 25, 3, 27, 111, 4, 18, 65, 9},  {1, 28, 3, 33, 98, 4, 24, 110, 9},
     {1, 29, 3, 35, 114, 4, 26, 111, 9}, {1, 14, 3, 5, 107, 4, 37, 109, 8},  {1, 15, 3, 7, 108, 4, 39, 115, 8},
     {1, 18, 3, 13, 32, 4, 4, 107, 9},   {1, 19, 3, 15, 43, 4, 6, 108, 9},   {1, 22, 3, 21, 76, 4, 12, 32, 9},
     {1, 23, 3, 23, 87, 4, 14, 43, 9},   {1, 26, 3, 29, 112, 4, 20, 76, 9},  {1, 27, 3, 31, 113, 4, 22, 87, 9},
     {1, 30, 3, 37, 109, 4, 28, 112, 9}, {1, 31, 3, 39, 115, 4, 30, 113, 9}, {1, 32, 3, 0, 96, 5, 32, 98, 9},
     {1, 33, 3, 2, 97, 5, 34, 114, 9},   {1, 36, 3, 8, 10, 5, 1, 96, 10},    {1, 37, 3, 10, 21, 5, 3, 97, 10},
     {1, 40, 3, 16, 54, 5, 9, 10, 10},   {1, 41, 3, 18, 65, 5, 11, 21, 10},  {1, 44, 3, 24, 110, 5, 17, 54, 10},
     {1, 45, 3, 26, 111, 5, 19, 65, 10}, {1, 34, 3, 4, 107, 5, 36, 109, 9},  {1, 35, 3, 6, 108, 5, 38, 115, 9},
     {1, 38, 3, 12, 32, 5, 5, 107, 10},  {1, 39, 3, 14, 43, 5, 7, 108, 10},  {1, 42, 3, 20, 76, 5, 13, 32, 10},
     {1, 43, 3, 22, 87, 5, 15, 43, 10},  {1, 46, 3, 28, 112, 5, 21, 76, 10}, {1, 47, 3, 30, 113, 5, 23, 87, 10}}};
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
// per home: band, unit, merge iteration, then per piece: slot, floater core, ordinal
constexpr std::array<std::array<uint32_t, 9>, 96> kK27HomeTable = {
    {{0, 0, 3, 0, 96, 1, 32, 98, 5},     {0, 1, 3, 2, 97, 1, 34, 114, 5},    {0, 4, 3, 8, 10, 1, 1, 96, 6},
     {0, 5, 3, 10, 21, 1, 3, 97, 6},     {0, 8, 3, 16, 54, 1, 9, 10, 6},     {0, 9, 3, 18, 65, 1, 11, 21, 6},
     {0, 12, 3, 24, 110, 1, 17, 54, 6},  {0, 13, 3, 26, 111, 1, 19, 65, 6},  {0, 16, 3, 32, 98, 1, 25, 110, 6},
     {0, 17, 3, 34, 114, 1, 27, 111, 6}, {0, 2, 3, 4, 107, 1, 36, 109, 5},   {0, 3, 3, 6, 108, 1, 38, 115, 5},
     {0, 6, 3, 12, 32, 1, 5, 107, 6},    {0, 7, 3, 14, 43, 1, 7, 108, 6},    {0, 10, 3, 20, 76, 1, 13, 32, 6},
     {0, 11, 3, 22, 87, 1, 15, 43, 6},   {0, 14, 3, 28, 112, 1, 21, 76, 6},  {0, 15, 3, 30, 113, 1, 23, 87, 6},
     {0, 18, 3, 36, 109, 1, 29, 112, 6}, {0, 19, 3, 38, 115, 1, 31, 113, 6}, {0, 20, 4, 1, 96, 2, 33, 98, 6},
     {0, 21, 4, 3, 97, 2, 35, 114, 6},   {0, 24, 4, 9, 10, 2, 0, 96, 7},     {0, 25, 4, 11, 21, 2, 2, 97, 7},
     {0, 28, 4, 17, 54, 2, 8, 10, 7},    {0, 29, 4, 19, 65, 2, 10, 21, 7},   {1, 0, 4, 25, 110, 2, 16, 54, 7},
     {1, 1, 4, 27, 111, 2, 18, 65, 7},   {1, 4, 4, 33, 98, 2, 24, 110, 7},   {1, 5, 4, 35, 114, 2, 26, 111, 7},
     {0, 22, 4, 5, 107, 2, 37, 109, 6},  {0, 23, 4, 7, 108, 2, 39, 115, 6},  {0, 26, 4, 13, 32, 2, 4, 107, 7},
     {0, 27, 4, 15, 43, 2, 6, 108, 7},   {0, 30, 4, 21, 76, 2, 12, 32, 7},   {0, 31, 4, 23, 87, 2, 14, 43, 7},
     {1, 2, 4, 29, 112, 2, 20, 76, 7},   {1, 3, 4, 31, 113, 2, 22, 87, 7},   {1, 6, 4, 37, 109, 2, 28, 112, 7},
     {1, 7, 4, 39, 115, 2, 30, 113, 7},  {1, 8, 5, 0, 96, 3, 32, 98, 7},     {1, 9, 5, 2, 97, 3, 34, 114, 7},
     {1, 12, 5, 8, 10, 3, 1, 96, 8},     {1, 13, 5, 10, 21, 3, 3, 97, 8},    {1, 16, 5, 16, 54, 3, 9, 10, 8},
     {1, 17, 5, 18, 65, 3, 11, 21, 8},   {1, 20, 5, 24, 110, 3, 17, 54, 8},  {1, 21, 5, 26, 111, 3, 19, 65, 8},
     {1, 24, 5, 32, 98, 3, 25, 110, 8},  {1, 25, 5, 34, 114, 3, 27, 111, 8}, {1, 10, 5, 4, 107, 3, 36, 109, 7},
     {1, 11, 5, 6, 108, 3, 38, 115, 7},  {1, 14, 5, 12, 32, 3, 5, 107, 8},   {1, 15, 5, 14, 43, 3, 7, 108, 8},
     {1, 18, 5, 20, 76, 3, 13, 32, 8},   {1, 19, 5, 22, 87, 3, 15, 43, 8},   {1, 22, 5, 28, 112, 3, 21, 76, 8},
     {1, 23, 5, 30, 113, 3, 23, 87, 8},  {1, 26, 5, 36, 109, 3, 29, 112, 8}, {1, 27, 5, 38, 115, 3, 31, 113, 8},
     {1, 28, 6, 1, 96, 4, 33, 98, 8},    {1, 29, 6, 3, 97, 4, 35, 114, 8},   {2, 0, 6, 9, 10, 4, 0, 96, 9},
     {2, 1, 6, 11, 21, 4, 2, 97, 9},     {2, 4, 6, 17, 54, 4, 8, 10, 9},     {2, 5, 6, 19, 65, 4, 10, 21, 9},
     {2, 8, 6, 25, 110, 4, 16, 54, 9},   {2, 9, 6, 27, 111, 4, 18, 65, 9},   {2, 12, 6, 33, 98, 4, 24, 110, 9},
     {2, 13, 6, 35, 114, 4, 26, 111, 9}, {1, 30, 6, 5, 107, 4, 37, 109, 8},  {1, 31, 6, 7, 108, 4, 39, 115, 8},
     {2, 2, 6, 13, 32, 4, 4, 107, 9},    {2, 3, 6, 15, 43, 4, 6, 108, 9},    {2, 6, 6, 21, 76, 4, 12, 32, 9},
     {2, 7, 6, 23, 87, 4, 14, 43, 9},    {2, 10, 6, 29, 112, 4, 20, 76, 9},  {2, 11, 6, 31, 113, 4, 22, 87, 9},
     {2, 14, 6, 37, 109, 4, 28, 112, 9}, {2, 15, 6, 39, 115, 4, 30, 113, 9}, {2, 16, 6, 0, 96, 5, 32, 98, 9},
     {2, 17, 6, 2, 97, 5, 34, 114, 9},   {2, 20, 6, 8, 10, 5, 1, 96, 10},    {2, 21, 6, 10, 21, 5, 3, 97, 10},
     {2, 24, 6, 16, 54, 5, 9, 10, 10},   {2, 25, 6, 18, 65, 5, 11, 21, 10},  {2, 28, 6, 24, 110, 5, 17, 54, 10},
     {2, 29, 6, 26, 111, 5, 19, 65, 10}, {2, 18, 6, 4, 107, 5, 36, 109, 9},  {2, 19, 6, 6, 108, 5, 38, 115, 9},
     {2, 22, 6, 12, 32, 5, 5, 107, 10},  {2, 23, 6, 14, 43, 5, 7, 108, 10},  {2, 26, 6, 20, 76, 5, 13, 32, 10},
     {2, 27, 6, 22, 87, 5, 15, 43, 10},  {2, 30, 6, 28, 112, 5, 21, 76, 10}, {2, 31, 6, 30, 113, 5, 23, 87, 10}}};
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
    const std::array<std::array<uint32_t, 9>, 96>& home_table;
    const std::array<std::array<uint32_t, 10>, 5>& visit_blocks;  // the first visit_counts[f] of each row
    const std::array<uint32_t, 5>& visit_counts;
    const std::array<uint32_t, 5>& first_piece_visits;
};

const std::array<Golden, 2> kGoldens = {{
    {24, 2, kK3Groups, kK3Homes, kK3HomeTable, kK3VisitBlocks, kK3VisitCounts, kK3FirstPieceVisits},
    {16, 3, kK27Groups, kK27Homes, kK27HomeTable, kK27VisitBlocks, kK27VisitCounts, kK27FirstPieceVisits},
}};

FloaterLayout layout_of(const Golden& golden) {
    FloaterSchedule schedule{};
    EXPECT_TRUE(floater_schedule(golden.heads, golden.bands, schedule));
    return build_floater_layout(
        schedule, kGridX, kGridY, golden.heads, golden.bands, kGatherWorkers, golden.heads * 2, kRingSize);
}

TEST(RingJointFloaterLayout, MatchesTheTunedTables) {
    for (const Golden& golden : kGoldens) {
        SCOPED_TRACE(::testing::Message() << "heads=" << golden.heads << " bands=" << golden.bands);
        const FloaterLayout layout = layout_of(golden);
        FloaterSchedule schedule{};
        floater_schedule(golden.heads, golden.bands, schedule);
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
                EXPECT_EQ(p.slot, row[3 + 3 * piece]) << "piece " << piece;
                EXPECT_EQ(p.floater, row[4 + 3 * piece]) << "piece " << piece;
                EXPECT_EQ(p.ordinal, row[5 + 3 * piece]) << "piece " << piece;
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
        std::set<std::pair<uint32_t, uint32_t>> slots;
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
                // A floater core's slot is reused only by its visits kFloaterRingSlots apart.
                EXPECT_EQ(
                    p.slot / kFloaterRingSlots,
                    layout.group_of[p.floater] * kFloaterGroupCores + layout.k_of[p.floater]);
                EXPECT_TRUE(slots.insert({p.floater, p.ordinal}).second);
            }
        }
        EXPECT_EQ(homes, kFloaterHomeBlocks * kFloaterGroupCores);
    }
}

TEST(RingJointFloaterLayout, OnlyTheTunedShapesHaveASchedule) {
    FloaterSchedule schedule{};
    for (uint32_t heads : {8u, 16u, 24u, 32u, 64u}) {
        for (uint32_t bands = 1; bands <= kKSplitMaxCount; ++bands) {
            const bool tuned = (heads == 24 && bands == 2) || (heads == 16 && bands == 3);
            EXPECT_EQ(floater_schedule(heads, bands, schedule), tuned) << "heads=" << heads << " bands=" << bands;
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
