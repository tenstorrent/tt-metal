// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <ttnn/distributed/types.hpp>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/distributed_mapping.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul.hpp"
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
    // xt::xarray<float> test_data = xt::arange(64 * 64 * 4) / 100.F;
    // xt::xarray<float> xtensor = test_data.reshape({2, 2, 64, 64});
    // non optimized branch
    xt::xarray<float> test_data = xt::arange(32 * 32 * 1) / 100.F;
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 32, 32});

    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tensor = ttml::core::from_xtensor(xtensor, device, replicate_composer);

    auto sum_tensor = ttnn::experimental::all_reduce(
        tensor, ttnn::operations::reduction::ReduceType::Sum, 1, std::nullopt, ttnn::ccl::Topology::Ring);
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);

    auto xtensors_back = ttml::core::to_xtensor(sum_tensor, identity_composer);
    auto reduced_tensor = xtensor + xtensor;

    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorShardAxis2AddScalar) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    float scalar = 10.F;
    xt::xarray<float> test_data = xt::arange(8);
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 2, 4});

    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ShardXTensorToMesh<float>(mesh_shape, 2);
    auto tensor = ttml::core::from_xtensor(xtensor, device, replicate_composer);
    auto out_tensor = ttnn::add(tensor, scalar);
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto xtensors_back = ttml::core::to_xtensor(out_tensor, identity_composer);

    xt::xarray<float> chunk0 = xt::view(xtensor, xt::all(), xt::all(), xt::range(0, 1), xt::all());
    xt::xarray<float> chunk1 = xt::view(xtensor, xt::all(), xt::all(), xt::range(1, 2), xt::all());

    EXPECT_TRUE(xt::allclose(chunk0 + scalar, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(chunk1 + scalar, xtensors_back[1]));
}

/*
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const uint32_t dim,
        const CoreCoord all_gather_core_grid_offset,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config_ag = std::nullopt,
        const std::optional<size_t> num_workers = std::nullopt,
        const std::optional<size_t> num_buffers_per_channel = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);
*/
TEST_F(N300UtilsTest, TestXTensorShardAxis3Matmul) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> xtensor_a = xt::arange(128 * 64).reshape({1, 1, 128, 64}) / 100.F;
    xt::xarray<float> xtensor_b = xt::arange(64 * 96).reshape({1, 1, 64, 96}) / 100.F;

    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ShardXTensorToMesh<float>(mesh_shape, 3);
    auto tensor_a = ttml::core::from_xtensor(xtensor_a, device, replicate_composer);
    auto tensor_b = ttml::core::from_xtensor(xtensor_b, device, replicate_composer);
    auto mul_tensor = ttnn::operations::experimental::ccl::all_gather_matmul(
        tensor_a,
        tensor_b,
        3,
        {0, 4},
        1,
        ttnn::MemoryConfig{},
        std::nullopt,
        std::nullopt,
        ttnn::MemoryConfig{},
        false,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        ttml::core::ComputeKernelConfig::precise());
    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto xtensors_back = ttml::core::to_xtensor(mul_tensor[0], identity_composer);

    xt::xarray<float> mul_res = xt::linalg::dot(xtensor_a, xtensor_b);

    EXPECT_TRUE(xt::allclose(mul_res, xtensors_back[0]));
}
