// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <set>
#include <vector>

#include "core_coord_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>

namespace basic_tests::CoreRangeSet {

TEST_F(CoreCoordFixture, TestCoreRangeSetMergeNoSolution) {
    EXPECT_EQ(::CoreRangeSet(sc1).merge(std::set{sc3}), ::CoreRangeSet(std::set{sc1, sc3}));
    EXPECT_EQ(::CoreRangeSet(cr1).merge(std::set{cr2}), ::CoreRangeSet(std::set{cr1, cr2}));
    EXPECT_EQ(::CoreRangeSet(cr1).merge(std::set{cr1, cr2}), ::CoreRangeSet(std::set{cr1, cr2}));
    EXPECT_EQ(::CoreRangeSet(cr1).merge(std::set{cr2}).merge(std::set{cr3}), ::CoreRangeSet(std::set{cr1, cr2, cr3}));
}

TEST_F(CoreCoordFixture, TestCoreRangeSetMergeCoreCoord) {
    ::CoreRangeSet empty_crs;
    EXPECT_EQ(empty_crs.merge(std::set{this->sc1}).ranges().size(), 1);
    EXPECT_EQ(::CoreRangeSet(cr1).merge(std::set{sc3, sc4}), ::CoreRangeSet(std::set{cr16}));
    EXPECT_EQ(::CoreRangeSet(cr1).merge(std::set{sc3}).merge(std::set{sc4}), ::CoreRangeSet(std::set{cr16}));
    ::CoreRange rect({0, 0}, {4, 2});
    std::set<::CoreRange> rect_pts;
    for (unsigned y = rect.start_coord.y; y <= rect.end_coord.y; y++) {
        for (unsigned x = rect.start_coord.x; x <= rect.end_coord.x; x++) {
            rect_pts.insert(::CoreRange({x, y}, {x, y}));
        }
    }
    EXPECT_EQ(empty_crs.merge(rect_pts), ::CoreRangeSet(std::set{rect}));

    // upside-down "T"
    rect_pts.insert({::CoreRange({2, 0}, {3, 5})});
    EXPECT_EQ(empty_crs.merge(rect_pts), ::CoreRangeSet(std::set{rect, ::CoreRange({2, 3}, {3, 5})}));

    // "H", sub-optimal currently, should be reduced down to 3 CRs instead of 5
    EXPECT_EQ(
        empty_crs.merge(
            std::vector{::CoreRange{{0, 0}, {1, 5}}, ::CoreRange{{3, 0}, {4, 5}}, ::CoreRange{{0, 2}, {4, 3}}}),
        ::CoreRangeSet(std::set{
            ::CoreRange{{0, 0}, {1, 1}},
            ::CoreRange{{0, 2}, {4, 3}},
            ::CoreRange{{0, 4}, {1, 5}},
            ::CoreRange{{3, 0}, {4, 1}},
            ::CoreRange{{3, 4}, {4, 5}}}));
}

TEST_F(CoreCoordFixture, TestCoreRangeSetMergeCoreRange) {
    EXPECT_EQ(::CoreRangeSet(cr1).merge(std::set{cr1}), ::CoreRangeSet(std::set{cr1}));
    EXPECT_EQ(::CoreRangeSet(cr7).merge(std::set{cr6}).merge(std::set{cr4}), ::CoreRangeSet(std::set{cr8}));
    EXPECT_EQ(
        ::CoreRangeSet(cr8).merge(std::set{cr7}).merge(std::set{cr6}).merge(std::set{cr4}),
        ::CoreRangeSet(std::set{cr8}));
    EXPECT_EQ(::CoreRangeSet(std::vector{cr1, cr2, cr3}).merge(std::set{cr4}), ::CoreRangeSet(std::set{cr4}));
    EXPECT_EQ(
        ::CoreRangeSet(std::vector{cr1, cr2}).merge(std::set{cr4}).merge(std::set{cr6}), ::CoreRangeSet(std::set{cr6}));
}

}  // namespace basic_tests::CoreRangeSet
