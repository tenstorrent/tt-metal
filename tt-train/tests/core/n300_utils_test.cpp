// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/distributed/types.hpp>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed_mapping.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce.hpp"
#include "xtensor/xbuilder.hpp"

auto check_board_is_n300() {
    return tt::Cluster::instance().get_board_type(0) == BoardType::N300;
}
class N300UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!check_board_is_n300()) {
            GTEST_SKIP() << "Skipping N300 specific tests";
        }
        ttml::autograd::ctx().set_mesh_shape({1, 2});
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(N300UtilsTest, TestXTensorReplicate) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    xt::xarray<float> test_data = {30.F, 20.F, 2.F};
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 1, 3});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tensor = ttml::core::from_xtensor(xtensor, device, replicate_composer);
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto xtensors_back = ttml::core::to_xtensor(tensor, identity_composer);

    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorShardAxis3) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::arange(8);
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 2, 4});

    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ShardXTensorToMesh<float>(mesh_shape, 3);
    auto tensor = ttml::core::from_xtensor(xtensor, device, replicate_composer);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto xtensors_back = ttml::core::to_xtensor(tensor, identity_composer);

    xt::xarray<float> chunk0 = xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, 2));
    xt::xarray<float> chunk1 = xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(2, 4));

    EXPECT_TRUE(xt::allclose(chunk0, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(chunk1, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorShardAxis2) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::arange(8);
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 2, 4});

    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ShardXTensorToMesh<float>(mesh_shape, 2);
    auto tensor = ttml::core::from_xtensor(xtensor, device, replicate_composer);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto xtensors_back = ttml::core::to_xtensor(tensor, identity_composer);

    xt::xarray<float> chunk0 = xt::view(xtensor, xt::all(), xt::all(), xt::range(0, 1), xt::all());
    xt::xarray<float> chunk1 = xt::view(xtensor, xt::all(), xt::all(), xt::range(1, 2), xt::all());

    EXPECT_TRUE(xt::allclose(chunk0, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(chunk1, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorReplicateAllReduce) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    // optimized branch
    xt::xarray<float> test_data = xt::arange(64 * 64 * 4) / 100.F;
    xt::xarray<float> xtensor = test_data.reshape({2, 2, 64, 64});
    // non optimized branch
    // xt::xarray<float> test_data = xt::arange(32 * 32 * 1) / 100.F;
    // xt::xarray<float> xtensor = test_data.reshape({1, 1, 32, 32});

    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tensor = ttml::core::from_xtensor(xtensor, device, replicate_composer);

    auto sum_tensor = ttnn::experimental::all_reduce(tensor, ttnn::operations::reduction::ReduceType::Sum);
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);

    auto xtensors_back = ttml::core::to_xtensor(sum_tensor, identity_composer);
    auto reduced_tensor = xtensor + xtensor;

    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[1]));
}
