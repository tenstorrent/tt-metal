// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "impl/metal2_host_api/llk_metadata.hpp"

namespace tt::tt_metal {
namespace {

// Resource metadata preserves an eight-wide face and its one-row, two-column grid.
TEST(BindingTokenFaceMetadata, EightWideFaces) {
    const LLKMetadata metadata{.format = DataFormat::Float16_b, .tile = Tile({16, 16}, {16, 8})};
    EXPECT_THAT(
        serialize_llk_metadata(metadata),
        ::testing::AllOf(
            ::testing::HasSubstr(".face_r_dim = 16u"),
            ::testing::HasSubstr(".face_c_dim = 8u"),
            ::testing::HasSubstr(".num_faces_r_dim = 1u"),
            ::testing::HasSubstr(".num_faces_c_dim = 2u")));
}

// Resource metadata preserves an eight-wide face in a two-row, two-column grid.
TEST(BindingTokenFaceMetadata, SquareEightWideFaces) {
    const LLKMetadata metadata{.format = DataFormat::Float16_b, .tile = Tile({16, 16}, {8, 8})};
    EXPECT_THAT(
        serialize_llk_metadata(metadata),
        ::testing::AllOf(
            ::testing::HasSubstr(".face_r_dim = 8u"),
            ::testing::HasSubstr(".face_c_dim = 8u"),
            ::testing::HasSubstr(".num_faces_r_dim = 2u"),
            ::testing::HasSubstr(".num_faces_c_dim = 2u")));
}

// Resource metadata preserves the existing geometry for sixteen-wide faces.
TEST(BindingTokenFaceMetadata, SixteenWideFaces) {
    const LLKMetadata metadata{.format = DataFormat::Float16_b, .tile = Tile({16, 32}, {8, 16})};
    EXPECT_THAT(
        serialize_llk_metadata(metadata),
        ::testing::AllOf(
            ::testing::HasSubstr(".face_r_dim = 8u"),
            ::testing::HasSubstr(".face_c_dim = 16u"),
            ::testing::HasSubstr(".num_faces_r_dim = 2u"),
            ::testing::HasSubstr(".num_faces_c_dim = 2u")));
}

}  // namespace
}  // namespace tt::tt_metal
