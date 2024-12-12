// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core/xtensor_all_includes.hpp>

#include "core/distributed_mapping.hpp"

namespace {

using ::testing::SizeIs;

template <typename T>
class MeshOpsTest : public ::testing::Test {
protected:
    // Common setup could go here if needed
};

using TestTypes = ::testing::Types<uint32_t, float>;
TYPED_TEST_SUITE(MeshOpsTest, TestTypes);

TYPED_TEST(MeshOpsTest, ChunkBasicNonDivisible3) {
    // Create a 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    // Using TypeParam ensures we test both uint32_t and float.
    xt::xarray<TypeParam> tensor = xt::arange<TypeParam>(10);

    // Chunk into 3 parts along dimension 0
    auto chunks = ttml::core::chunk(tensor, 3, 0);

    ASSERT_THAT(chunks, SizeIs(3));
    EXPECT_EQ(chunks[0].shape()[0], 4u);  // first chunk size 4
    EXPECT_EQ(chunks[1].shape()[0], 4u);  // next chunk size 4
    EXPECT_EQ(chunks[2].shape()[0], 2u);  // last chunk size 2
}

TYPED_TEST(MeshOpsTest, ChunkBasicLessChunksThanProvided) {
    // Create a 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12]
    xt::xarray<TypeParam> tensor = xt::arange<TypeParam>(13);

    // Chunk into 6 parts along dimension 0
    auto chunks = ttml::core::chunk(tensor, 6, 0);

    ASSERT_THAT(chunks, SizeIs(5));
    EXPECT_EQ(chunks[0].shape()[0], 3u);  // first chunk size 3
    EXPECT_EQ(chunks[1].shape()[0], 3u);  // next chunk size 3
    EXPECT_EQ(chunks[2].shape()[0], 3u);  // next chunk size 3
    EXPECT_EQ(chunks[3].shape()[0], 3u);  // next chunk size 3
    EXPECT_EQ(chunks[4].shape()[0], 1u);  // last chunk size 1
}

TYPED_TEST(MeshOpsTest, ShardXTensorToMeshBasicShard) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {1, 4};

    // A simple 1D tensor to shard across 4 devices
    auto tensor = xt::arange<TypeParam>(8);  // [0,...,7]

    ttml::core::ShardXTensorToMesh<TypeParam> sharder(mesh_shape, 0);
    auto shards = sharder.map(tensor);

    // With 4 shards, each shard should have size 2
    ASSERT_THAT(shards, SizeIs(4));
    for (auto& s : shards) {
        EXPECT_EQ(s.size(), 2u);
    }
}

TYPED_TEST(MeshOpsTest, ShardTensor2dMeshTwoDimSharding) {
    // Mesh shape: 2x2, total 4 devices
    tt::tt_metal::distributed::MeshShape mesh_shape = {2, 2};

    // Create a 2D tensor shape: (4,4)
    auto tensor = xt::arange<TypeParam>(16).reshape({4, 4});

    // Shard along row_dim=0 and col_dim=1
    ttml::core::ShardTensor2dMesh<TypeParam> sharder(mesh_shape, {0, 1});
    auto shards = sharder.map(tensor);

    ASSERT_THAT(shards, SizeIs(4));
    // Check shapes of shards
    for (auto& shard : shards) {
        EXPECT_EQ(shard.shape()[0], 2u);
        EXPECT_EQ(shard.shape()[1], 2u);
    }
}

TYPED_TEST(MeshOpsTest, ReplicateXTensorToMeshReplication) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {2, 2};
    int num_devices = mesh_shape.num_rows * mesh_shape.num_cols;  // 4

    auto tensor = xt::arange<TypeParam>(4);  // [0,1,2,3]

    ttml::core::ReplicateXTensorToMesh<TypeParam> replicator(mesh_shape);
    auto replicas = replicator.map(tensor);

    ASSERT_THAT(replicas, SizeIs(num_devices));
    for (const auto& t : replicas) {
        EXPECT_TRUE(xt::allclose(t, tensor));
    }
}

TYPED_TEST(MeshOpsTest, ConcatMesh2dToTensorRecomposition) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {2, 2};

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

    ttml::core::ConcatMesh2dToTensor<TypeParam> composer(mesh_shape, {0, 1});
    auto composed = composer.compose(shards);

    xt::xarray<TypeParam> expected = {
        {TypeParam(0), TypeParam(1), TypeParam(2), TypeParam(3)},
        {TypeParam(4), TypeParam(5), TypeParam(6), TypeParam(7)},
        {TypeParam(8), TypeParam(9), TypeParam(10), TypeParam(11)},
        {TypeParam(12), TypeParam(13), TypeParam(14), TypeParam(15)}};

    EXPECT_TRUE(xt::allclose(composed[0], expected));
}

TYPED_TEST(MeshOpsTest, ConcatMeshToXTensorOneDimConcatenation) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {1, 3};

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
    tt::tt_metal::distributed::MeshShape mesh_shape = {2, 2};
    ttml::core::VectorMeshToXTensor<TypeParam> vectorComposer(mesh_shape);

    std::vector<xt::xarray<TypeParam>> shards = {
        xt::xarray<TypeParam>({TypeParam(0), TypeParam(1)}), xt::xarray<TypeParam>({TypeParam(2), TypeParam(3)})};

    auto result = vectorComposer.compose(shards);
    ASSERT_EQ(result.size(), shards.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        EXPECT_TRUE(xt::allclose(result[i], shards[i]));
    }
}

TEST(ConcatenateTest, DefaultAxis) {
    xt::xarray<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    xt::xarray<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    std::vector<xt::xarray<double>> input = {a, b};

    xt::xarray<double> result = ttml::core::concatenate(input);  // axis=0 by default
    xt::xarray<double> expected = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};

    xt::allclose(result, expected);
}

TEST(ConcatenateTest, AxisOne) {
    xt::xarray<int> x = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> y = {{7, 8}, {9, 10}};
    std::vector<xt::xarray<int>> input = {x, y};

    xt::xarray<int> result = ttml::core::concatenate(input, 1);
    xt::xarray<int> expected = {{1, 2, 3, 7, 8}, {4, 5, 6, 9, 10}};

    xt::allclose(result, expected);
}

TEST(ConcatenateTest, MultipleArraysAxis0) {
    xt::xarray<float> a = {1.0f, 2.0f};
    xt::xarray<float> b = {3.0f, 4.0f};
    xt::xarray<float> c = {5.0f, 6.0f};
    std::vector<xt::xarray<float>> input = {a, b, c};

    xt::xarray<float> result = ttml::core::concatenate(input, 0);
    xt::xarray<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    xt::allclose(result, expected);
}

TEST(ConcatenateTest, EmptyArray) {
    xt::xarray<int> a = {{1, 2}, {3, 4}};
    xt::xarray<int> b;  // Empty
    std::vector<xt::xarray<int>> input = {a, b};

    EXPECT_ANY_THROW({ xt::xarray<int> result = ttml::core::concatenate(input, 0); });
}

TEST(ConcatenateTest, HigherDimensions) {
    xt::xarray<int> arr1 = xt::arange<int>(1, 9);  // 1 to 8
    arr1.reshape({2, 2, 2});
    xt::xarray<int> arr2 = xt::arange<int>(9, 17);  // 9 to 16
    arr2.reshape({2, 2, 2});

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    xt::xarray<int> result = ttml::core::concatenate(input, 0);

    // Expected: shape (4,2,2) with arr1 stacked over arr2 along axis 0
    xt::xarray<int> expected = xt::concatenate(xt::xtuple(arr1, arr2), 0);

    xt::allclose(result, expected);
}

TEST(ConcatenateTest, HigherAxis) {
    xt::xarray<int> arr1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}};
    // Both have shape (2,2,2)

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    xt::xarray<int> result = ttml::core::concatenate(input, 2);
    // Expected shape: (2,2,4)
    xt::xarray<int> expected = {{{1, 2, 9, 10}, {3, 4, 11, 12}}, {{5, 6, 13, 14}, {7, 8, 15, 16}}};

    xt::allclose(result, expected);
}

TYPED_TEST(MeshOpsTest, ConcatenateSameParametersAsCompose) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {1, 3};

    // Create a few shards: [0,1], [2,3], [4,5]
    xt::xarray<TypeParam> s1 = {TypeParam(0), TypeParam(1)};
    xt::xarray<TypeParam> s2 = {TypeParam(2), TypeParam(3)};
    xt::xarray<TypeParam> s3 = {TypeParam(4), TypeParam(5)};

    std::vector<xt::xarray<TypeParam>> shards = {s1, s2, s3};
    ttml::core::ConcatMeshToXTensor<TypeParam> composer(mesh_shape, 0);
    auto composed = ttml::core::concatenate(shards);

    xt::xarray<TypeParam> expected = {
        TypeParam(0), TypeParam(1), TypeParam(2), TypeParam(3), TypeParam(4), TypeParam(5)};
    EXPECT_TRUE(xt::allclose(composed, expected));
}
}  // namespace
