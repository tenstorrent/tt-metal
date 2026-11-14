// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_fabric {
namespace {

TEST(FabricNodeIdTest, OrderingWithinMesh) {
    const FabricNodeId a(MeshId{0}, 0);
    const FabricNodeId b(MeshId{0}, 1);

    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(b > a);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(a >= b);
}

TEST(FabricNodeIdTest, OrderingAcrossMeshes) {
    const FabricNodeId a(MeshId{0}, 7);
    const FabricNodeId b(MeshId{1}, 0);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(b <= a);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(a >= b);
}

TEST(FabricNodeIdTest, EqualIdsCompareInclusive) {
    const FabricNodeId a(MeshId{2}, 3);
    const FabricNodeId b(MeshId{2}, 3);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
}

}  // namespace
}  // namespace tt::tt_fabric
