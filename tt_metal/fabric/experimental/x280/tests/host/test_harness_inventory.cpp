// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host tests for directional NoC hop counts and tile-pair selection.

#include "harness/inventory/inventory.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using x280::Noc0Coord;
using x280::harness::inventory::hops;
using x280::harness::inventory::kNocGridX;
using x280::harness::inventory::kNocGridY;
using x280::harness::inventory::NocId;
using x280::harness::inventory::parse_tile_pair_spec;

// L2CPU coordinates from x280_state.cpp::noc0_from_hw_l2cpu.
constexpr Noc0Coord kL2cpu0{8, 3};
constexpr Noc0Coord kL2cpu1{8, 9};
constexpr Noc0Coord kL2cpu2{8, 5};

// DRAM and Tensix coordinates from rt/x280_noc_map.h.
constexpr Noc0Coord kDram{9, 3};
constexpr Noc0Coord kTensixA{1, 11};
constexpr Noc0Coord kTensixB{16, 2};

}  // namespace

TEST(Hops, CountsDirectionalRingDistance) {
    struct Case {
        Noc0Coord from;
        Noc0Coord to;
        uint32_t noc0;
        uint32_t noc1;
    };
    constexpr std::array<Case, 8> cases = {{
        {kL2cpu0, kL2cpu0, 0u, 0u},
        {kL2cpu0, kDram, 1u, kNocGridX - 1u},
        {kL2cpu0, kL2cpu2, 2u, kNocGridY - 2u},
        {kL2cpu2, kL2cpu0, kNocGridY - 2u, 2u},
        {kL2cpu0, kL2cpu1, kNocGridY / 2u, kNocGridY / 2u},
        {kTensixB, {1u, 2u}, 2u, kNocGridX - 2u},
        {kTensixA, {1u, 0u}, 1u, kNocGridY - 1u},
        {{0u, 0u}, {kNocGridX - 1u, kNocGridY - 1u}, kNocGridX + kNocGridY - 2u, 2u},
    }};
    for (const auto& test : cases) {
        EXPECT_EQ(hops(test.from, test.to, NocId::Noc0), test.noc0);
        EXPECT_EQ(hops(test.from, test.to, NocId::Noc1), test.noc1);
    }
}

// The two NOCs are the same grid walked the other way, so one NOC's forward
// leg is the other's return leg.
TEST(Hops, ForwardOnOneNocIsTheReverseLegOnTheOther) {
    for (uint32_t from_x = 0; from_x < kNocGridX; from_x++) {
        for (uint32_t from_y = 0; from_y < kNocGridY; from_y++) {
            for (uint32_t to_x = 0; to_x < kNocGridX; to_x++) {
                for (uint32_t to_y = 0; to_y < kNocGridY; to_y++) {
                    const Noc0Coord from{from_x, from_y};
                    const Noc0Coord to{to_x, to_y};
                    EXPECT_EQ(hops(from, to, NocId::Noc0), hops(to, from, NocId::Noc1))
                        << "(" << from_x << "," << from_y << ") -> (" << to_x << "," << to_y << ")";
                }
            }
        }
    }
}

TEST(Hops, RejectsAnEndpointOutsideTheGridNamingIt) {
    try {
        hops(Noc0Coord{kNocGridX, 3}, kL2cpu0, NocId::Noc0);
        FAIL() << "an off-grid column must not produce a count";
    } catch (const std::out_of_range& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("from"), std::string::npos) << what;
        EXPECT_NE(what.find("(17,3)"), std::string::npos) << what;
    }

    try {
        hops(kL2cpu0, Noc0Coord{8, kNocGridY}, NocId::Noc1);
        FAIL() << "an off-grid row must not produce a count";
    } catch (const std::out_of_range& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("to"), std::string::npos) << what;
        EXPECT_NE(what.find("(8,12)"), std::string::npos) << what;
    }
    EXPECT_THROW(hops(Noc0Coord{UINT32_MAX, 3}, kL2cpu0, NocId::Noc0), std::out_of_range);
    EXPECT_THROW(hops(kL2cpu0, Noc0Coord{8, UINT32_MAX}, NocId::Noc0), std::out_of_range);
}

TEST(TilePairSpec, ParsesOrderedIndicesAndWhitespace) {
    for (const auto& test : {
             std::pair{"0,2", std::pair{0u, 2u}},
             std::pair{"2,0", std::pair{2u, 0u}},
             std::pair{"  1 , 3  ", std::pair{1u, 3u}},
         }) {
        const auto parsed = parse_tile_pair_spec(test.first);
        EXPECT_EQ(parsed.first, test.second.first);
        EXPECT_EQ(parsed.second, test.second.second);
    }
}

TEST(TilePairSpec, RejectsMalformedOrOutOfRangeIndices) {
    for (const char* spec : {
             "0",
             "0,1,2",
             "0,",
             ",1",
             ",",
             "0x2,1",
             "2abc,1",
             "-1,1",
             "0,4294967296",
         }) {
        EXPECT_THROW(parse_tile_pair_spec(spec), std::invalid_argument) << spec;
    }
}

TEST(TilePairSpec, RejectsANonNumericIndexNamingIt) {
    try {
        parse_tile_pair_spec("0,two");
        FAIL() << "a non-numeric index must not select a tile";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("second"), std::string::npos) << what;
        EXPECT_NE(what.find("two"), std::string::npos) << what;
    }
}

TEST(TilePairSpec, RejectsTheSameTileTwiceNamingIt) {
    try {
        parse_tile_pair_spec("3,3");
        FAIL() << "one tile cannot be both participants";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("twice"), std::string::npos) << what;
        EXPECT_NE(what.find('3'), std::string::npos) << what;
    }
}
