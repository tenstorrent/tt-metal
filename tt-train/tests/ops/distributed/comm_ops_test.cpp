// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/distributed/comm_ops.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"

namespace {

auto check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == BoardType::N300;
}

}  // namespace

class N300CommOpsTest : public ::testing::Test {
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

TEST_F(N300CommOpsTest, TestAllReduceNotFullyTiled) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t size = 64UL;
    std::vector<float> test_data_vec(size);
    std::iota(test_data_vec.begin(), test_data_vec.end(), 0.0F);
    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    xt::xarray<float> xtensor = test_data.reshape({1U, 1U, 1U, size});
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto all_reduce_tensor = ttml::ops::distributed::all_reduce(tensor);

    auto all_reduce_xtensor =
        ttml::core::to_xtensor<float>(all_reduce_tensor->get_value(), ttml::core::IdentityComposer{});

    xt::xarray<float> all_reduce_expected =
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) +
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size));

    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-3));
    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-3));

    xt::xarray<float> grad_data = xt::empty<float>(all_reduce_expected.shape());
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{grad_data.data(), grad_data.size()},
        []() { return std::uniform_real_distribution<float>(0.F, 1.F); },
        seed);
    mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_grad_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());
    all_reduce_tensor->set_grad(tt_grad_tensor);
    all_reduce_tensor->backward();

    auto result_tensor_grad = tensor->get_grad();
    EXPECT_TRUE(ttml::core::is_tensor_initialized(result_tensor_grad));

    auto num_devices_scale = static_cast<float>(ttml::autograd::ctx().get_device().num_devices());

    auto grad_xtensor = ttml::core::to_xtensor<float>(tensor->get_grad(), ttml::core::IdentityComposer{});
    EXPECT_EQ(grad_xtensor[0].shape(), grad_xtensor[1].shape());
    EXPECT_TRUE(xt::allclose(
        grad_data * num_devices_scale,
        grad_xtensor[0],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        grad_data * num_devices_scale,
        grad_xtensor[1],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
}

TEST_F(N300CommOpsTest, TestAllReduceNanoGPT) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t batch = 64;
    size_t size = 384;
    size_t height = 256;
    std::vector<float> test_data_vec(batch * size * height);
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{test_data_vec.data(), test_data_vec.size()},
        []() { return std::uniform_real_distribution<float>{-1.F, 1.F}; },
        seed);
    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    xt::xarray<float> xtensor = test_data.reshape({batch, 1U, height, size});
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto all_reduce_tensor = ttml::ops::distributed::all_reduce(tensor);

    auto all_reduce_xtensor =
        ttml::core::to_xtensor<float>(all_reduce_tensor->get_value(), ttml::core::IdentityComposer{});

    xt::xarray<float> all_reduce_expected =
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) +
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size));

    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[0], /* rtol */ 1e-3, /* atol */ 2e-2));
    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[1], /* rtol */ 1e-3, /* atol */ 2e-2));

    xt::xarray<float> grad_data = xt::empty<float>(all_reduce_expected.shape());
    uint32_t seed2 = rng();
    ttml::core::parallel_generate(
        std::span{grad_data.data(), grad_data.size()},
        []() { return std::uniform_real_distribution<float>(0.F, 1.F); },
        seed2);
    mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_grad_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());
    all_reduce_tensor->set_grad(tt_grad_tensor);
    all_reduce_tensor->backward();

    auto result_tensor_grad = tensor->get_grad();
    EXPECT_TRUE(ttml::core::is_tensor_initialized(result_tensor_grad));

    auto num_devices_scale = static_cast<float>(ttml::autograd::ctx().get_device().num_devices());
    auto grad_xtensor = ttml::core::to_xtensor<float>(tensor->get_grad(), ttml::core::IdentityComposer{});
    EXPECT_EQ(grad_xtensor[0].shape(), grad_xtensor[1].shape());
    EXPECT_TRUE(xt::allclose(
        grad_data * num_devices_scale,
        grad_xtensor[0],
        /* rtol */ 1e-3,
        /* atol */ 2e-2));
    EXPECT_TRUE(xt::allclose(
        grad_data * num_devices_scale,
        grad_xtensor[1],
        /* rtol */ 1e-3,
        /* atol */ 2e-2));
}

TEST_F(N300CommOpsTest, TestAllReduceFullyTiled) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t size = 64UL;
    size_t height = 32UL;
    std::vector<float> test_data_vec(size * height);
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{test_data_vec.data(), test_data_vec.size()},
        []() { return std::uniform_real_distribution<float>(0.F, 0.001F); },
        seed);
    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    xt::xarray<float> xtensor = test_data.reshape({1U, 1U, height, size});
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto all_reduce_tensor = ttml::ops::distributed::all_reduce(tensor);

    auto all_reduce_xtensor =
        ttml::core::to_xtensor<float>(all_reduce_tensor->get_value(), ttml::core::IdentityComposer{});

    xt::xarray<float> all_reduce_expected =
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) +
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size));

    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    xt::xarray<float> grad_data = xt::empty<float>(all_reduce_expected.shape());
    seed = rng();
    ttml::core::parallel_generate(
        std::span{grad_data.data(), grad_data.size()},
        []() { return std::uniform_real_distribution<float>(0.F, 1.F); },
        seed);
    mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_grad_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());
    all_reduce_tensor->set_grad(tt_grad_tensor);
    all_reduce_tensor->backward();

    auto result_tensor_grad = tensor->get_grad();
    EXPECT_TRUE(ttml::core::is_tensor_initialized(result_tensor_grad));

    auto num_devices_scale = static_cast<float>(ttml::autograd::ctx().get_device().num_devices());
    auto grad_xtensor = ttml::core::to_xtensor<float>(tensor->get_grad(), ttml::core::IdentityComposer{});
    EXPECT_EQ(grad_xtensor[0].shape(), grad_xtensor[1].shape());
    EXPECT_TRUE(xt::allclose(
        grad_data * num_devices_scale,
        grad_xtensor[0],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        grad_data * num_devices_scale,
        grad_xtensor[1],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
}

TEST_F(N300CommOpsTest, TestAllGatherNotFullyTiled) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t size = 64UL;
    std::vector<float> test_data_vec(size);
    std::iota(test_data_vec.begin(), test_data_vec.end(), 0.0F);
    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    xt::xarray<float> xtensor = test_data.reshape({1U, 1U, 1U, size});
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto gathered_tensor = ttml::ops::distributed::all_gather(tensor, 3);

    auto gathered_xtensor = ttml::core::to_xtensor<float>(gathered_tensor->get_value(), ttml::core::IdentityComposer{});
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    xt::xarray<float> grad_data = xt::empty<float>(xtensor.shape());
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{grad_data.data(), grad_data.size()},
        []() { return std::uniform_real_distribution<float>(0.F, 1.F); },
        seed);
    mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_grad_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());
    gathered_tensor->set_grad(tt_grad_tensor);
    gathered_tensor->backward();

    auto result_tensor_grad = tensor->get_grad();
    EXPECT_TRUE(ttml::core::is_tensor_initialized(result_tensor_grad));

    auto num_devices_scale = static_cast<float>(ttml::autograd::ctx().get_device().num_devices());
    auto grad_xtensor = ttml::core::to_xtensor<float>(tensor->get_grad(), ttml::core::IdentityComposer{});
    EXPECT_EQ(grad_xtensor[0].shape(), grad_xtensor[1].shape());
    EXPECT_TRUE(xt::allclose(
        xt::view(grad_data, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) * num_devices_scale,
        grad_xtensor[0],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        xt::view(grad_data, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size)) * num_devices_scale,
        grad_xtensor[1],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
}

TEST_F(N300CommOpsTest, TestAllGatherFullyTiled) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t batch = 64;
    size_t size = 64UL;
    size_t height = 256UL;
    std::vector<float> test_data_vec(batch * size * height);
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{test_data_vec.data(), test_data_vec.size()},
        []() { return std::uniform_real_distribution<float>{-1.F, 1.F}; },
        seed);
    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    xt::xarray<float> xtensor = test_data.reshape({batch, 1U, height, size});
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto gathered_tensor = ttml::ops::distributed::all_gather(tensor, 3);

    auto gathered_xtensor = ttml::core::to_xtensor<float>(gathered_tensor->get_value(), ttml::core::IdentityComposer{});
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    xt::xarray<float> grad_data = xt::empty<float>(xtensor.shape());
    seed = rng();
    ttml::core::parallel_generate(
        std::span{grad_data.data(), grad_data.size()},
        []() { return std::uniform_real_distribution<float>(0.F, 1.F); },
        seed);
    mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_grad_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());
    gathered_tensor->set_grad(tt_grad_tensor);
    gathered_tensor->backward();

    auto result_tensor_grad = tensor->get_grad();
    EXPECT_TRUE(ttml::core::is_tensor_initialized(result_tensor_grad));

    auto num_devices_scale = static_cast<float>(ttml::autograd::ctx().get_device().num_devices());
    auto grad_xtensor = ttml::core::to_xtensor<float>(tensor->get_grad(), ttml::core::IdentityComposer{});
    EXPECT_EQ(grad_xtensor[0].shape(), grad_xtensor[1].shape());
    EXPECT_TRUE(xt::allclose(
        xt::view(grad_data, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) * num_devices_scale,
        grad_xtensor[0],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        xt::view(grad_data, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size)) * num_devices_scale,
        grad_xtensor[1],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
}

TEST_F(N300CommOpsTest, TestScatterNotFullyTiled) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t size = 64UL;
    std::vector<float> test_data_vec(size);
    std::iota(test_data_vec.begin(), test_data_vec.end(), 0.0F);
    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    xt::xarray<float> xtensor = test_data.reshape({1U, 1U, 1U, size});
    auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto scattered_tensor = ttml::ops::distributed::reduce_scatter(tensor, 3);

    // check forward
    auto xtensors_back = ttml::core::to_xtensor<float>(scattered_tensor->get_value(), ttml::core::IdentityComposer{});
    auto num_devices_scale = static_cast<float>(ttml::autograd::ctx().get_device().num_devices());
    EXPECT_TRUE(xt::allclose(
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) * num_devices_scale,
        xtensors_back[0]));
    EXPECT_TRUE(xt::allclose(
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size)) * num_devices_scale,
        xtensors_back[1]));

    // check backward
    xt::xarray<float> grad_data = xt::empty<float>(xtensor.shape());
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{grad_data.data(), grad_data.size()},
        []() { return std::uniform_real_distribution<float>(0.F, 1.F); },
        seed);
    mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_grad_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());
    scattered_tensor->set_grad(tt_grad_tensor);
    scattered_tensor->backward();

    auto result_tensor_grad = tensor->get_grad();
    EXPECT_TRUE(ttml::core::is_tensor_initialized(result_tensor_grad));

    auto grad_xtensor = ttml::core::to_xtensor<float>(tensor->get_grad(), ttml::core::IdentityComposer{});
    EXPECT_TRUE(grad_data.shape() == grad_xtensor[0].shape());
    EXPECT_TRUE(grad_data.shape() == grad_xtensor[1].shape());

    EXPECT_EQ(grad_xtensor[0], grad_xtensor[1]);
    EXPECT_TRUE(xt::allclose(grad_data, grad_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(grad_data, grad_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));
}

TEST_F(N300CommOpsTest, TestScatterFullyTiled) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t batch = 64;
    size_t size = 128UL;
    size_t height = 256UL;
    std::vector<float> test_data_vec(batch * size * height);
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{test_data_vec.data(), test_data_vec.size()},
        []() { return std::uniform_real_distribution<float>{-1.F, 1.F}; },
        seed);
    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    xt::xarray<float> xtensor = test_data.reshape({batch, 1U, height, size});

    auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());

    auto xtensor_after_replication = ttml::core::to_xtensor<float>(tt_tensor, ttml::core::IdentityComposer{});
    EXPECT_TRUE(xt::allclose(xtensor, xtensor_after_replication[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(xtensor, xtensor_after_replication[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto scattered_tensor = ttml::ops::distributed::reduce_scatter(tensor, 3);

    // check forward
    auto xtensors_back = ttml::core::to_xtensor<float>(scattered_tensor->get_value(), ttml::core::IdentityComposer{});
    auto num_devices_scale = static_cast<float>(ttml::autograd::ctx().get_device().num_devices());
    EXPECT_TRUE(xt::allclose(
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) * num_devices_scale,
        xtensors_back[0],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size)) * num_devices_scale,
        xtensors_back[1],
        /* rtol */ 1e-3,
        /* atol */ 1e-2));

    // check backward
    xt::xarray<float> grad_data = xt::empty<float>(xtensor.shape());
    seed = rng();
    ttml::core::parallel_generate(
        std::span{grad_data.data(), grad_data.size()},
        []() { return std::uniform_real_distribution<float>(-1.F, 1.F); },
        seed);
    mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_grad_tensor = ttml::core::from_xtensor(grad_data, device, ttnn::Layout::TILE, mapper.get());
    scattered_tensor->set_grad(tt_grad_tensor);
    scattered_tensor->backward();

    auto result_tensor_grad = tensor->get_grad();
    EXPECT_TRUE(ttml::core::is_tensor_initialized(result_tensor_grad));

    auto grad_xtensor = ttml::core::to_xtensor<float>(tensor->get_grad(), ttml::core::IdentityComposer{});
    EXPECT_TRUE(grad_data.shape() == grad_xtensor[0].shape());
    EXPECT_TRUE(grad_data.shape() == grad_xtensor[1].shape());

    EXPECT_EQ(grad_xtensor[0], grad_xtensor[1]);
    EXPECT_TRUE(xt::allclose(grad_data, grad_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(grad_data, grad_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));
}
