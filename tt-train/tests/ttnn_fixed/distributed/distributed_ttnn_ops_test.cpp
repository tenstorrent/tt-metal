// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <umd/device/cluster.h>

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace {

auto check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == BoardType::N300;
}

class TrivialTnnFixedDistributedTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!check_board_is_n300()) {
            GTEST_SKIP() << "Skipping N300 specific tests";
        }
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

}  // namespace

TEST_F(TrivialTnnFixedDistributedTest, TestCustomScatterDim0) {
    auto* device = &ttml::autograd::ctx().get_device();

    uint32_t size = 64U;
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    auto shape = ttnn::Shape({size, 1, 1, 1});
    auto tensor = ttml::core::from_vector(data, shape, device);

    auto scattered_tensor = ttml::ttnn_fixed::distributed::scatter(tensor, /* dim */ 0);

    auto mesh_shape = device->shape();
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);

    auto xtensors_back = ttml::core::to_xtensor(scattered_tensor, identity_composer);

    auto tensor_0 = xtensors_back[0];
    auto tensor_1 = xtensors_back[1];

    EXPECT_EQ(tensor_0.shape()[0], size / 2);
    EXPECT_EQ(tensor_1.shape()[0], size / 2);
    for (int i = 0; i < size / 2; ++i) {
        EXPECT_EQ(tensor_0(i, 0, 0, 0), i);
        EXPECT_EQ(tensor_1(i, 0, 0, 0), i + size / 2);
    }
}

TEST_F(TrivialTnnFixedDistributedTest, TestCustomScatterDim1) {
    auto* device = &ttml::autograd::ctx().get_device();

    uint32_t size = 64U;
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    auto shape = ttnn::Shape({1, size, 1, 1});
    auto tensor = ttml::core::from_vector(data, shape, device);

    auto scattered_tensor = ttml::ttnn_fixed::distributed::scatter(tensor, /* dim */ 1);

    auto mesh_shape = device->shape();
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);

    auto xtensors_back = ttml::core::to_xtensor(scattered_tensor, identity_composer);

    auto tensor_0 = xtensors_back[0];
    auto tensor_1 = xtensors_back[1];

    EXPECT_EQ(tensor_0.shape()[1], size / 2);
    EXPECT_EQ(tensor_1.shape()[1], size / 2);
    for (int i = 0; i < size / 2; ++i) {
        EXPECT_EQ(tensor_0(0, i, 0, 0), i);
        EXPECT_EQ(tensor_1(0, i, 0, 0), i + size / 2);
    }
}

TEST_F(TrivialTnnFixedDistributedTest, TestCustomScatterDim2) {
    auto* device = &ttml::autograd::ctx().get_device();

    uint32_t size = 64U;
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    auto shape = ttnn::Shape({1, 1, size, 1});
    auto tensor = ttml::core::from_vector(data, shape, device);

    auto scattered_tensor = ttml::ttnn_fixed::distributed::scatter(tensor, /* dim */ 2);

    auto mesh_shape = device->shape();
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);

    auto xtensors_back = ttml::core::to_xtensor(scattered_tensor, identity_composer);

    auto tensor_0 = xtensors_back[0];
    auto tensor_1 = xtensors_back[1];

    EXPECT_EQ(tensor_0.shape()[2], size / 2);
    EXPECT_EQ(tensor_1.shape()[2], size / 2);
    for (int i = 0; i < size / 2; ++i) {
        EXPECT_EQ(tensor_0(0, 0, i, 0), i);
        EXPECT_EQ(tensor_1(0, 0, i, 0), i + size / 2);
    }
}

TEST_F(TrivialTnnFixedDistributedTest, TestCustomScatterDim3) {
    auto* device = &ttml::autograd::ctx().get_device();

    uint32_t size = 64U;
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    auto shape = ttnn::Shape({1, 1, 1, size});
    auto tensor = ttml::core::from_vector(data, shape, device);

    auto scattered_tensor = ttml::ttnn_fixed::distributed::scatter(tensor, /* dim */ 3);

    auto mesh_shape = device->shape();
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);

    auto xtensors_back = ttml::core::to_xtensor(scattered_tensor, identity_composer);

    auto tensor_0 = xtensors_back[0];
    auto tensor_1 = xtensors_back[1];

    EXPECT_EQ(tensor_0.shape()[3], size / 2);
    EXPECT_EQ(tensor_1.shape()[3], size / 2);
    for (int i = 0; i < size / 2; ++i) {
        EXPECT_EQ(tensor_0(0, 0, 0, i), i);
        EXPECT_EQ(tensor_1(0, 0, 0, i), i + size / 2);
    }
}
