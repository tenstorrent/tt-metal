// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-only tests for the shared NoC coordinate representation: the one range
// check, checked packing, NoC 1 addressability, and the guarantee that only
// the resolver constructs a TranslatedCoord.

#include "x280/device/translated_coord.hpp"

#include "x280_noc_coord.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

namespace {

// A logical or NOC0 pair cannot become a TranslatedCoord, so it cannot reach a
// packer that takes one.
static_assert(!std::is_constructible_v<x280::TranslatedCoord, uint32_t, uint32_t>);
static_assert(!std::is_default_constructible_v<x280::TranslatedCoord>);
static_assert(!std::is_constructible_v<x280::TranslatedCoord, x280_noc_coord_t>);

TEST(X280NocCoord, RangeCheckIsSixBitsPerAxis) {
    EXPECT_TRUE(x280_noc_coord_valid({X280_NOC_COORD_MAX, X280_NOC_COORD_MAX}));
    EXPECT_FALSE(x280_noc_coord_valid({X280_NOC_COORD_MAX + 1u, 0u}));
    EXPECT_FALSE(x280_noc_coord_valid({0u, X280_NOC_COORD_MAX + 1u}));
}

TEST(X280NocCoord, RectangleRequiresStartBeforeEnd) {
    EXPECT_TRUE(x280_noc_rect_valid({{1u, 2u}, {1u, 2u}}));
    EXPECT_FALSE(x280_noc_rect_valid({{2u, 2u}, {1u, 3u}}));
    EXPECT_FALSE(x280_noc_rect_valid({{1u, 3u}, {2u, 2u}}));
    EXPECT_FALSE(x280_noc_rect_valid({{1u, 2u}, {64u, 2u}}));
}

TEST(X280NocCoord, PackingRefusesInsteadOfMasking) {
    uint32_t enc = 0xDEADu;
    EXPECT_FALSE(x280_noc_xy_pack({64u, 3u}, &enc));
    EXPECT_FALSE(x280_noc_xy_pack({3u, 64u}, &enc));
    EXPECT_EQ(enc, 0xDEADu);
    ASSERT_TRUE(x280_noc_xy_pack({8u, 3u}, &enc));
    EXPECT_EQ(enc, (3u << 6) | 8u);

    x280_noc_coord_t c{99u, 99u};
    EXPECT_FALSE(x280_noc_xy_unpack(1u << 12, &c));
    EXPECT_EQ(c.x, 99u);
    ASSERT_TRUE(x280_noc_xy_unpack(enc, &c));
    EXPECT_EQ(c.x, 8u);
    EXPECT_EQ(c.y, 3u);
}

TEST(X280NocCoord, Noc1RefusesUntranslatedRows) {
    for (const x280_noc_coord_t c :
         {x280_noc_coord_t{3u, 1u}, x280_noc_coord_t{0u, 0u}, x280_noc_coord_t{9u, 0u}, x280_noc_coord_t{63u, 1u}}) {
        EXPECT_FALSE(x280_noc_translated_addressable(c, 1u)) << c.x << "," << c.y;
        EXPECT_TRUE(x280_noc_translated_addressable(c, 0u)) << c.x << "," << c.y;
    }
    for (const x280_noc_coord_t c : {x280_noc_coord_t{8u, 0u}, x280_noc_coord_t{8u, 1u}, x280_noc_coord_t{3u, 2u}}) {
        EXPECT_TRUE(x280_noc_translated_addressable(c, 1u)) << c.x << "," << c.y;
    }
    EXPECT_FALSE(x280_noc_translated_addressable({64u, 26u}, 0u));
}

}  // namespace
