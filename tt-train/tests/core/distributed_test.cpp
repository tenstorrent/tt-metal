// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ttnn/tensor/xtensor/xtensor_all_includes.hpp>

#include "core/distributed_mapping.hpp"

namespace {

using MetalMeshShape = ::tt::tt_metal::distributed::MeshShape;
using ::testing::SizeIs;

template <typename T>
class MeshOpsTest : public ::testing::Test {
protected:
    // Common setup could go here if needed
};

using TestTypes = ::testing::Types<uint32_t, float>;
TYPED_TEST_SUITE(MeshOpsTest, TestTypes);

TYPED_TEST(MeshOpsTest, ConcatMesh2dToTensorRecomposition) {
    MetalMeshShape mesh_shape{2, 2};

    // Create shards that would come from a 4x4 tensor:
    // Expected final tensor:
    // [[0,1,2,3],
    //  [4,5,6,7],
    //  [8,9,10,11],
    //  [12,13,14,15]]
    //
    // Shards (2x2 each):
    xt::xarray<TypeParam> top_left = {{TypeParam(0), TypeParam(1)}, {TypeParam(4), TypeParam(5)}};
    xt::xarray<TypeParam> top_right = {{TypeParam(2), TypeParam(3)}, {TypeParam(6), TypeParam(7)}};
    xt::xarray<TypeParam> bot_left = {{TypeParam(8), TypeParam(9)}, {TypeParam(12), TypeParam(13)}};
    xt::xarray<TypeParam> bot_right = {{TypeParam(10), TypeParam(11)}, {TypeParam(14), TypeParam(15)}};

    std::vector<xt::xarray<TypeParam>> shards = {top_left, top_right, bot_left, bot_right};

    ttml::core::ConcatMesh2dToTensor<TypeParam> composer(mesh_shape, MetalMeshShape{0, 1});
    auto composed = composer.compose(shards);

    xt::xarray<TypeParam> expected = {
        {TypeParam(0), TypeParam(1), TypeParam(2), TypeParam(3)},
        {TypeParam(4), TypeParam(5), TypeParam(6), TypeParam(7)},
        {TypeParam(8), TypeParam(9), TypeParam(10), TypeParam(11)},
        {TypeParam(12), TypeParam(13), TypeParam(14), TypeParam(15)}};

    EXPECT_TRUE(xt::allclose(composed[0], expected));
}

TYPED_TEST(MeshOpsTest, ConcatMeshToXTensorOneDimConcatenation) {
    MetalMeshShape mesh_shape{1, 3};

    // Create a few shards: [0,1], [2,3], [4,5]
    xt::xarray<TypeParam> s1 = {TypeParam(0), TypeParam(1)};
    xt::xarray<TypeParam> s2 = {TypeParam(2), TypeParam(3)};
    xt::xarray<TypeParam> s3 = {TypeParam(4), TypeParam(5)};

    std::vector<xt::xarray<TypeParam>> shards = {s1, s2, s3};
    ttml::core::ConcatMeshToXTensor<TypeParam> composer(mesh_shape, 0);
    auto composed = composer.compose(shards);

    xt::xarray<TypeParam> expected = {
        TypeParam(0), TypeParam(1), TypeParam(2), TypeParam(3), TypeParam(4), TypeParam(5)};
    EXPECT_TRUE(xt::allclose(composed[0], expected));
}

TYPED_TEST(MeshOpsTest, VectorMeshToXTensorVectorReturn) {
    MetalMeshShape mesh_shape{2, 2};
    ttml::core::VectorMeshToXTensor<TypeParam> vectorComposer(mesh_shape);

    std::vector<xt::xarray<TypeParam>> shards = {
        xt::xarray<TypeParam>({TypeParam(0), TypeParam(1)}), xt::xarray<TypeParam>({TypeParam(2), TypeParam(3)})};

    auto result = vectorComposer.compose(shards);
    ASSERT_EQ(result.size(), shards.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        EXPECT_TRUE(xt::allclose(result[i], shards[i]));
    }
}

TYPED_TEST(MeshOpsTest, ConcatenateSameParametersAsCompose) {
    MetalMeshShape mesh_shape{1, 3};

    // Create a few shards: [0,1], [2,3], [4,5]
    xt::xarray<TypeParam> s1 = {TypeParam(0), TypeParam(1)};
    xt::xarray<TypeParam> s2 = {TypeParam(2), TypeParam(3)};
    xt::xarray<TypeParam> s3 = {TypeParam(4), TypeParam(5)};

    std::vector<xt::xarray<TypeParam>> shards = {s1, s2, s3};
    ttml::core::ConcatMeshToXTensor<TypeParam> composer(mesh_shape, 0);
    auto composed = ttml::core::concat(shards);

    xt::xarray<TypeParam> expected = {
        TypeParam(0), TypeParam(1), TypeParam(2), TypeParam(3), TypeParam(4), TypeParam(5)};
    EXPECT_TRUE(xt::allclose(composed, expected));
}
}  // namespace
