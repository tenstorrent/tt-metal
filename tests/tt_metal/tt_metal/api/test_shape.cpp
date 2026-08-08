// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/shape.hpp>
#include <tt_stl/reflection.hpp>

#include "gtest/gtest.h"

namespace tt::tt_metal {

// Regression test: shapes with different ranks but same padded representation
// must produce different hashes. ShapeBase::init() pads value_ to min-4D with
// leading 1s, so [128, 128] and [1, 128, 128] both store {1, 1, 128, 128}.
// The rank must be included in the hash to distinguish them.
TEST(TensorShapeTests, DifferentRankShapesProduceDifferentHashes) {
    tt::tt_metal::Shape shape_2d({128, 128});
    tt::tt_metal::Shape shape_3d({1, 128, 128});

    EXPECT_EQ(shape_2d.rank(), 2);
    EXPECT_EQ(shape_3d.rank(), 3);

    // Shapes are not equal despite same volume
    EXPECT_NE(shape_2d, shape_3d);

    // Hashes must differ
    auto hash_2d = ttsl::hash::detail::hash_object(shape_2d);
    auto hash_3d = ttsl::hash::detail::hash_object(shape_3d);
    EXPECT_NE(hash_2d, hash_3d);
}

TEST(TensorShapeTests, SameRankShapesWithDifferentDimsProduceDifferentHashes) {
    tt::tt_metal::Shape shape_a({32, 64});
    tt::tt_metal::Shape shape_b({64, 32});

    auto hash_a = ttsl::hash::detail::hash_object(shape_a);
    auto hash_b = ttsl::hash::detail::hash_object(shape_b);
    EXPECT_NE(hash_a, hash_b);
}

TEST(TensorShapeTests, IdenticalShapesProduceSameHash) {
    tt::tt_metal::Shape shape_a({1, 128, 128});
    tt::tt_metal::Shape shape_b({1, 128, 128});

    auto hash_a = ttsl::hash::detail::hash_object(shape_a);
    auto hash_b = ttsl::hash::detail::hash_object(shape_b);
    EXPECT_EQ(hash_a, hash_b);
}

TEST(TensorShapeTests, GetNormalizedIndexInRange) {
    tt::tt_metal::Shape shape({32, 64, 128});

    EXPECT_EQ(shape.get_normalized_index(2), 2u);
    EXPECT_EQ(shape.get_normalized_index(-3), 0u);
}

TEST(TensorShapeTests, GetNormalizedIndexOutOfRangeThrows) {
    tt::tt_metal::Shape shape({32, 64});

    EXPECT_ANY_THROW(shape.get_normalized_index(2));
    // Regression: normalized_index was unsigned, so this wrapped past the `>= 0` guard.
    EXPECT_ANY_THROW(shape.get_normalized_index(-3));
}

}  // namespace tt::tt_metal
