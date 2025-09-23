// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <umd/device/cluster.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

using namespace ttml;

auto check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == BoardType::N300;
}

class N300UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!check_board_is_n300()) {
            GTEST_SKIP() << "Skipping N300 specific tests";
        }

        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(N300UtilsTest, TestXTensorReplicateInt32) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    xt::xarray<int32_t> test_data = {30, 20, 2};
    xt::xarray<int32_t> xtensor = test_data.reshape({1, 1, 1, 3});
    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tensor =
        ttml::core::from_xtensor<int32_t, ttnn::DataType::INT32>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto xtensors_back = ttml::core::to_xtensor<int32_t>(tensor, ttml::core::IdentityComposer{});

    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorReplicateUInt32) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    xt::xarray<uint32_t> test_data = {30U, 20U, 2U};
    xt::xarray<uint32_t> xtensor = test_data.reshape({1, 1, 1, 3});
    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tensor =
        ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto xtensors_back = ttml::core::to_xtensor<uint32_t>(tensor, ttml::core::IdentityComposer{});
    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorReplicate) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    xt::xarray<float> test_data = {30.F, 20.F, 2.F};
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 1, 3});
    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto xtensors_back = ttml::core::to_xtensor(tensor, ttml::core::IdentityComposer{});

    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(xtensor, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorShardAxis3) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::arange(8);
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 2, 4});

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());

    auto xtensors_back = ttml::core::to_xtensor(tensor, ttml::core::IdentityComposer{});

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

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 2);
    auto tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());

    auto xtensors_back = ttml::core::to_xtensor(tensor, ttml::core::IdentityComposer{});

    xt::xarray<float> chunk0 = xt::view(xtensor, xt::all(), xt::all(), xt::range(0, 1), xt::all());
    xt::xarray<float> chunk1 = xt::view(xtensor, xt::all(), xt::all(), xt::range(1, 2), xt::all());

    EXPECT_TRUE(xt::allclose(chunk0, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(chunk1, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorReplicateAllReduce) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> xtensor_data = xt::empty<float>({32 * 32});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{xtensor_data.data(), xtensor_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.05, 0.05); },
        seed);
    xt::xarray<float> xtensor = xtensor_data.reshape({1, 1, 32, 32});

    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());

    auto sum_tensor = ttnn_fixed::distributed::all_reduce(tensor);

    auto xtensors_back = ttml::core::to_xtensor(sum_tensor, ttml::core::IdentityComposer{});
    auto reduced_tensor = xtensor + xtensor;

    std::cout << "xtensors_back[0]: " << xtensors_back[0] << std::endl;
    std::cout << "xtensors_back[1]: " << xtensors_back[1] << std::endl;
    std::cout << "reduced_tensor: " << reduced_tensor << std::endl;
    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[0], /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[1], /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(N300UtilsTest, TestXTensorReplicateAllReduceBadTiles) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> xtensor_data = xt::empty<float>({32});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{xtensor_data.data(), xtensor_data.size()},
        []() { return std::uniform_real_distribution<float>(-1.F, 1.F); },
        seed);
    xt::xarray<float> xtensor = xtensor_data.reshape({1, 1, 4, 8});

    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());

    auto sum_tensor = ttnn_fixed::distributed::all_reduce(tensor);

    auto xtensors_back = ttml::core::to_xtensor(sum_tensor, ttml::core::IdentityComposer{});
    auto reduced_tensor = xtensor + xtensor;

    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[0], /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(reduced_tensor, xtensors_back[1], /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(N300UtilsTest, TestXTensorShardAxis2AddScalar) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    float scalar = 10.F;
    xt::xarray<float> test_data = xt::arange(8);
    xt::xarray<float> xtensor = test_data.reshape({1, 1, 2, 4});

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 2);
    auto tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto out_tensor = ttnn::add(tensor, scalar);
    auto xtensors_back = ttml::core::to_xtensor(out_tensor, ttml::core::IdentityComposer{});

    xt::xarray<float> chunk0 = xt::view(xtensor, xt::all(), xt::all(), xt::range(0, 1), xt::all());
    xt::xarray<float> chunk1 = xt::view(xtensor, xt::all(), xt::all(), xt::range(1, 2), xt::all());

    EXPECT_TRUE(xt::allclose(chunk0 + scalar, xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(chunk1 + scalar, xtensors_back[1]));
}

TEST_F(N300UtilsTest, TestXTensorShardAxis3Matmul) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> xtensor_a_data = xt::empty<float>({128 * 64});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{xtensor_a_data.data(), xtensor_a_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.005, 0.005); },
        seed);
    xt::xarray<float> xtensor_a = xtensor_a_data.reshape({1, 1, 128, 64});

    xt::xarray<float> xtensor_b_data = xt::empty<float>({256 * 64});
    uint32_t seed2 = rng();
    ttml::core::parallel_generate(
        std::span{xtensor_b_data.data(), xtensor_b_data.size()},
        []() { return std::uniform_real_distribution<float>(-0.005, 0.005); },
        seed2);
    xt::xarray<float> xtensor_b = xtensor_b_data.reshape({1, 1, 64, 256});

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tensor_a = ttml::core::from_xtensor(xtensor_a, device, ttnn::Layout::TILE, mapper.get());
    auto tensor_b = ttml::core::from_xtensor(xtensor_b, device, ttnn::Layout::TILE, mapper.get());

    auto gathered_ta = ttnn_fixed::distributed::all_gather(tensor_a, 3);
    fmt::print("gathered_ta shape: {}\n", gathered_ta.logical_shape());
    auto mul_tensor = ttnn::matmul(
        gathered_ta,
        tensor_b,
        false,
        false,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */ ttml::core::ComputeKernelConfig::precise(),
        /* core_grid */ ttnn::CoreGrid{7, 8},
        /* output_tile */ std::nullopt);
    auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*device, 3);
    auto xtensors_back = ttml::core::to_xtensor(mul_tensor, *composer);
    xt::xarray<float> mul_res = xt::linalg::dot(xtensor_a, xtensor_b);

    // (128, 64) X (64, 256) => (128, 256)
    EXPECT_TRUE(xt::allclose(mul_res, xtensors_back[0], /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(N300UtilsTest, DropoutDifferentSeed) {
    uint32_t dropout_seed1 = 42;
    float scale = 2.0F;
    float prob = 0.5F;
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    auto shapes = {std::vector<int>{64, 1, 256, 384}, std::vector<int>{1, 1, 32, 32}};
    for (auto& shape : shapes) {
        fmt::println("Testing shape: {}", shape);
        xt::xarray<float> xtensor = xt::ones<float>(shape);
        const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
        auto xtensor_tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());
        auto out_tensor = ttnn::experimental::dropout(xtensor_tensor, prob, scale, dropout_seed1);
        auto xtensors_back = ttml::core::to_xtensor(out_tensor, ttml::core::IdentityComposer{});
        EXPECT_FALSE(xt::allclose(xtensors_back[0], xtensors_back[1], /*rtol=*/1e-4, /*atol=*/1e-3));
    }
}

TEST_F(N300UtilsTest, MorehClipGradNorm) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    xt::xarray<float> xtensor = xt::ones<float>({4, 1, 20, 5});

    const auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tensor = ttml::core::from_xtensor(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto do_it = [&tensor]() {
        ttnn::moreh_clip_grad_norm(
            std::vector<tt::tt_metal::Tensor>{tensor},
            1.0F,
            2.0F,
            false,
            /* total_norm */ std::nullopt,
            /* memory_config */ std::nullopt,
            ttml::core::ComputeKernelConfig::precise());
    };
    // ensure that moreh clip grad norm works without throwing a
    // bad_variant_access on n300.
    EXPECT_NO_THROW(do_it());
    xt::xarray<float> expected_res = xt::full_like(xtensor, 0.05F);

    auto res_back = ttml::core::to_xtensor(tensor, ttml::core::IdentityComposer{})[0];
    EXPECT_TRUE(xt::allclose(expected_res, res_back, 2.2e-2F));
}
