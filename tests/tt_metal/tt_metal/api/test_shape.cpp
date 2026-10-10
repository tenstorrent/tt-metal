// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include <tt-metalium/shape.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

// Regression test: shapes with different ranks but same padded representation
// must produce different hashes. ShapeBase::init() pads value_ to min-4D with
// leading 1s, so [128, 128] and [1, 128, 128] both store {1, 1, 128, 128}.
// The rank must be included in the hash to distinguish them.
TEST(TensorShapeTests, CPU_DifferentRankShapesProduceDifferentHashes) {
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

TEST(TensorShapeTests, CPU_SameRankShapesWithDifferentDimsProduceDifferentHashes) {
    tt::tt_metal::Shape shape_a({32, 64});
    tt::tt_metal::Shape shape_b({64, 32});

    auto hash_a = ttsl::hash::detail::hash_object(shape_a);
    auto hash_b = ttsl::hash::detail::hash_object(shape_b);
    EXPECT_NE(hash_a, hash_b);
}

TEST(TensorShapeTests, CPU_IdenticalShapesProduceSameHash) {
    tt::tt_metal::Shape shape_a({1, 128, 128});
    tt::tt_metal::Shape shape_b({1, 128, 128});

    auto hash_a = ttsl::hash::detail::hash_object(shape_a);
    auto hash_b = ttsl::hash::detail::hash_object(shape_b);
    EXPECT_EQ(hash_a, hash_b);
}

TEST(TensorShapeTests, GetNormalizedIndexInRange) {
    tt::tt_metal::Shape shape({32, 64});

    // Normalization is against the logical rank of 2, not the padded 4D storage.
    EXPECT_EQ(shape.get_normalized_index(1), 1u);
    EXPECT_EQ(shape.get_normalized_index(-2), 0u);
}

TEST(TensorShapeTests, GetNormalizedIndexOutOfRangeReportsCallerIndex) {
    tt::tt_metal::Shape shape({32, 64});

    EXPECT_THAT(
        ([&shape]() { shape.get_normalized_index(2); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("2 not in [-2, 2)")));
    EXPECT_THAT(
        ([&shape]() { shape.get_normalized_index(-3); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("-3 not in [-2, 2)")));
}

}  // namespace
}  // namespace tt::tt_metal
