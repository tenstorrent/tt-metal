// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/xtensor_all_includes.hpp>

#include "core/distributed_mapping.hpp"

template <typename T>
class MeshOpsTest : public ::testing::Test {
protected:
    // Common setup could go here if needed
};

using TestTypes = ::testing::Types<std::uint32_t, float>;
TYPED_TEST_SUITE(MeshOpsTest, TestTypes);

TYPED_TEST(MeshOpsTest, ChunkBasic) {
    // Create a 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    // Using TypeParam ensures we test both uint32_t and float.
    xt::xarray<TypeParam> tensor = xt::arange<TypeParam>(10);

    // Chunk into 3 parts along dimension 0
    auto chunks = ttml::core::chunk(tensor, 3, 0);

    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0].shape()[0], 4u);  // first chunk size 4
    EXPECT_EQ(chunks[1].shape()[0], 3u);  // next chunk size 3
    EXPECT_EQ(chunks[2].shape()[0], 3u);  // last chunk size 3
}

TYPED_TEST(MeshOpsTest, ShardTensorToMeshBasicShard) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {1, 4};

    // A simple 1D tensor to shard across 4 devices
    auto tensor = xt::arange<TypeParam>(8);  // [0,...,7]

    ttml::core::ShardTensorToMesh<TypeParam> sharder(mesh_shape, 0);
    auto shards = sharder.map(tensor);

    // With 4 shards, each shard should have size 2
    ASSERT_EQ(shards.size(), 4u);
    for (auto& s : shards) {
        EXPECT_EQ(s.size(), 2u);
    }
}

TYPED_TEST(MeshOpsTest, ShardTensor2dMeshTwoDimSharding) {
    // Mesh shape: 2x2, total 4 devices
    tt::tt_metal::distributed::MeshShape mesh_shape = {2, 2};

    // Create a 2D tensor shape: (4,4)
    auto tensor = xt::arange<TypeParam>(16);
    tensor.reshape({4, 4});

    // Shard along row_dim=0 and col_dim=1
    ttml::core::ShardTensor2dMesh<TypeParam> sharder(mesh_shape, {0, 1});
    auto shards = sharder.map(tensor);

    ASSERT_EQ(shards.size(), 4u);
    // Check shapes of shards
    for (auto& shard : shards) {
        EXPECT_EQ(shard.shape()[0], 2u);
        EXPECT_EQ(shard.shape()[1], 2u);
    }
}

TYPED_TEST(MeshOpsTest, ReplicateTensorToMeshReplication) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {2, 2};
    int num_devices = mesh_shape.first * mesh_shape.second;  // 4

    auto tensor = xt::arange<TypeParam>(4);  // [0,1,2,3]

    ttml::core::ReplicateTensorToMesh<TypeParam> replicator(mesh_shape);
    auto replicas = replicator.map(tensor);

    ASSERT_EQ(static_cast<int>(replicas.size()), num_devices);
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

    EXPECT_TRUE(xt::allclose(composed, expected));
}

TYPED_TEST(MeshOpsTest, ConcatMeshToTensorOneDimConcatenation) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {1, 3};

    // Create a few shards: [0,1], [2,3], [4,5]
    xt::xarray<TypeParam> s1 = {TypeParam(0), TypeParam(1)};
    xt::xarray<TypeParam> s2 = {TypeParam(2), TypeParam(3)};
    xt::xarray<TypeParam> s3 = {TypeParam(4), TypeParam(5)};

    std::vector<xt::xarray<TypeParam>> shards = {s1, s2, s3};
    ttml::core::ConcatMeshToTensor<TypeParam> composer(mesh_shape, 0);
    auto composed = composer.compose(shards);

    xt::xarray<TypeParam> expected = {
        TypeParam(0), TypeParam(1), TypeParam(2), TypeParam(3), TypeParam(4), TypeParam(5)};
    EXPECT_TRUE(xt::allclose(composed, expected));
}

TYPED_TEST(MeshOpsTest, VectorMeshToTensorVectorReturn) {
    tt::tt_metal::distributed::MeshShape mesh_shape = {2, 2};
    ttml::core::VectorMeshToTensor<TypeParam> vectorComposer(mesh_shape);

    std::vector<xt::xarray<TypeParam>> shards = {
        xt::xarray<TypeParam>({TypeParam(0), TypeParam(1)}), xt::xarray<TypeParam>({TypeParam(2), TypeParam(3)})};

    auto result = vectorComposer.compose(shards);
    ASSERT_EQ(result.size(), shards.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        EXPECT_TRUE(xt::allclose(result[i], shards[i]));
    }
}
