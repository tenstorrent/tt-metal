// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

#include <functional>
#include <set>

#include "gtest/gtest.h"

namespace basic_tests::core_coord_hash {
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

TEST(CoreRangeHashTest, CPU_DistinctRangesDoNotCollide) {
    const CoreRange a({3, 0}, {3, 0});
    const CoreRange b({1, 1}, {1, 1});
    EXPECT_NE(std::hash<CoreRange>{}(a), std::hash<CoreRange>{}(b));
    EXPECT_NE(ttsl::hash::canonical_key(a), ttsl::hash::canonical_key(b));
}

TEST(CoreRangeHashTest, CPU_OrderAndExtentMatter) {
    const CoreRange a({0, 0}, {1, 2});
    const CoreRange b({0, 0}, {2, 1});
    EXPECT_NE(std::hash<CoreRange>{}(a), std::hash<CoreRange>{}(b));
}

TEST(CoreRangeSetHashTest, CPU_DistinctGridsDoNotCollide) {
    const CoreRangeSet a(std::set<CoreRange>{CoreRange({0, 0}, {1, 1})});
    const CoreRangeSet b(std::set<CoreRange>{CoreRange({3, 3}, {5, 4})});
    EXPECT_NE(std::hash<CoreRangeSet>{}(a), std::hash<CoreRangeSet>{}(b));
    EXPECT_NE(ttsl::hash::canonical_key(a), ttsl::hash::canonical_key(b));
}

TEST(CoreRangeSetHashTest, CPU_LengthMatters) {
    const CoreRangeSet a(std::set<CoreRange>{CoreRange({0, 0}, {0, 0})});
    const CoreRangeSet b(std::set<CoreRange>{CoreRange({0, 0}, {0, 0}), CoreRange({2, 2}, {2, 2})});
    EXPECT_NE(std::hash<CoreRangeSet>{}(a), std::hash<CoreRangeSet>{}(b));
}

}  // namespace basic_tests::core_coord_hash
