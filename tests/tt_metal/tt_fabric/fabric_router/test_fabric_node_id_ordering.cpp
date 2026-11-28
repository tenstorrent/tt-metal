// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Offline unit tests for FabricNodeId's relational operators: the (mesh_id, chip_id)
// lexicographic order the fabric keys on. No device, no descriptors -- just the ordering
// and its four relational operators agreeing with each other.
//
// Regression for #62279: operator<= was `!(rhs > lhs)`, and since operator> is `rhs < lhs`
// that expands to `!(lhs < rhs)`, which is operator>=. The two operators were
// behaviourally identical, so `FabricNodeId(MeshId{0}, 0) <= FabricNodeId(MeshId{0}, 1)`
// was false.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_fabric {
namespace {

FabricNodeId node(uint32_t mesh, uint32_t chip) { return FabricNodeId(MeshId{mesh}, chip); }

// The reproduction from #62279, stated as the issue states it.
TEST(FabricNodeIdOrderingTest, LessOrEqualIsTrueForAnAscendingPairInOneMesh) {
    EXPECT_TRUE(node(0, 0) <= node(0, 1));
    EXPECT_FALSE(node(0, 1) <= node(0, 0));
}

TEST(FabricNodeIdOrderingTest, LessOrEqualIsTrueForAnAscendingPairAcrossMeshes) {
    EXPECT_TRUE(node(0, 7) <= node(1, 0));
    EXPECT_FALSE(node(1, 0) <= node(0, 7));
}

TEST(FabricNodeIdOrderingTest, BothInclusiveOperatorsHoldOnEqualIds) {
    EXPECT_TRUE(node(2, 3) <= node(2, 3));
    EXPECT_TRUE(node(2, 3) >= node(2, 3));
}

// operator<= and operator>= must NOT be the same function. This is the assertion that
// fails on the pre-fix source and is the whole point of the regression.
TEST(FabricNodeIdOrderingTest, LessOrEqualIsNotTheSameAsGreaterOrEqual) {
    EXPECT_NE(node(0, 0) <= node(0, 1), node(0, 0) >= node(0, 1));
}

// Exhaustive over a small mesh space: the four relational operators must agree with the
// lexicographic order they are all defined from, on every ordered pair.
TEST(FabricNodeIdOrderingTest, AllFourOperatorsAgreeWithLexicographicOrderExhaustively) {
    std::vector<FabricNodeId> ids;
    for (uint32_t mesh = 0; mesh < 3; ++mesh) {
        for (uint32_t chip = 0; chip < 4; ++chip) {
            ids.push_back(node(mesh, chip));
        }
    }
    for (const auto& a : ids) {
        for (const auto& b : ids) {
            const bool lt = (a.mesh_id < b.mesh_id) ||
                            (a.mesh_id == b.mesh_id && a.chip_id < b.chip_id);
            const bool eq = (a.mesh_id == b.mesh_id) && (a.chip_id == b.chip_id);
            EXPECT_EQ(a < b, lt);
            EXPECT_EQ(a > b, !lt && !eq);
            EXPECT_EQ(a <= b, lt || eq);
            EXPECT_EQ(a >= b, !lt);
        }
    }
}

}  // namespace
}  // namespace tt::tt_fabric
