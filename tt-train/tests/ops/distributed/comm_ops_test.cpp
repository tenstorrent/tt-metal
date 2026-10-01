// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/distributed/comm_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <core/xtensor_utils.hpp>
#include <stdexcept>
#include <umd/device/cluster.hpp>
#include <variant>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

auto check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == tt::BoardType::N300;
}

}  // namespace

class N300CommOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!check_board_is_n300()) {
            GTEST_SKIP() << "Skipping N300 specific tests";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(2U);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(N300CommOpsTest, TestAllReduceNotFullyTiled) {
    // Test failing with watcher enabled, github issue #30521
    SKIP_FOR_WATCHER();

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
    auto tensor = ttml::autograd::create_tensor(tt_tensor, /* requires_grad */ true);
    auto all_reduce_tensor = ttml::ops::distributed::all_reduce(tensor);

    auto all_reduce_xtensor =
        ttml::core::to_xtensor<float>(all_reduce_tensor->get_value(), ttml::core::IdentityComposer{});

    xt::xarray<float> all_reduce_expected =
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) +
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size));

    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-3));
    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-3));

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> grad_data = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{1U, 1U, 1U, size / 2}, 0.0F, 1.0F, seed);
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
    // Test failing with watcher enabled, github issue #30521
    SKIP_FOR_WATCHER();

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t batch = 64;
    size_t size = 384;
    size_t height = 256;
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, height, size}, -1.0F, 1.0F, seed);
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor, /* requires_grad */ true);
    auto all_reduce_tensor = ttml::ops::distributed::all_reduce(tensor);

    auto all_reduce_xtensor =
        ttml::core::to_xtensor<float>(all_reduce_tensor->get_value(), ttml::core::IdentityComposer{});

    xt::xarray<float> all_reduce_expected =
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) +
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size));

    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[0], /* rtol */ 1e-3, /* atol */ 2e-2));
    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[1], /* rtol */ 1e-3, /* atol */ 2e-2));

    uint32_t seed2 = rng();
    xt::xarray<float> grad_data = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, height, size / 2}, 0.0F, 1.0F, seed2);
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
    // Test failing with watcher enabled, github issue #30521
    SKIP_FOR_WATCHER();

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    size_t size = 64UL;
    size_t height = 32UL;
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{1U, 1U, height, size}, 0.0F, 0.001F, seed);
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor, /* requires_grad */ true);
    auto all_reduce_tensor = ttml::ops::distributed::all_reduce(tensor);

    auto all_reduce_xtensor =
        ttml::core::to_xtensor<float>(all_reduce_tensor->get_value(), ttml::core::IdentityComposer{});

    xt::xarray<float> all_reduce_expected =
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(0, size / 2)) +
        xt::view(xtensor, xt::all(), xt::all(), xt::all(), xt::range(size / 2, size));

    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(all_reduce_expected, all_reduce_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    seed = rng();
    xt::xarray<float> grad_data = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{1U, 1U, height, size / 2}, 0.0F, 1.0F, seed);
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
    // Test failing with watcher enabled, github issue #36312
    SKIP_FOR_WATCHER();

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
    auto tensor = ttml::autograd::create_tensor(tt_tensor, /* requires_grad */ true);
    auto gathered_tensor = ttml::ops::distributed::all_gather(tensor, 3);

    auto gathered_xtensor = ttml::core::to_xtensor<float>(gathered_tensor->get_value(), ttml::core::IdentityComposer{});
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> grad_data =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, size}, 0.0F, 1.0F, seed);
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
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, height, size}, -1.0F, 1.0F, seed);
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = ttml::autograd::create_tensor(tt_tensor, /* requires_grad */ true);
    auto gathered_tensor = ttml::ops::distributed::all_gather(tensor, 3);

    auto gathered_xtensor = ttml::core::to_xtensor<float>(gathered_tensor->get_value(), ttml::core::IdentityComposer{});
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(xtensor, gathered_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    seed = rng();
    xt::xarray<float> grad_data = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, height, size}, 0.0F, 1.0F, seed);
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
    // Test failing with watcher enabled, github issue #36312
    SKIP_FOR_WATCHER();

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
    auto tensor = ttml::autograd::create_tensor(tt_tensor, /* requires_grad */ true);
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
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> grad_data =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, size}, 0.0F, 1.0F, seed);
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
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, height, size}, -1.0F, 1.0F, seed);

    auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto tt_tensor =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());

    auto xtensor_after_replication = ttml::core::to_xtensor<float>(tt_tensor, ttml::core::IdentityComposer{});
    EXPECT_TRUE(xt::allclose(xtensor, xtensor_after_replication[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(xtensor, xtensor_after_replication[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    auto tensor = ttml::autograd::create_tensor(tt_tensor, /* requires_grad */ true);
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
    seed = rng();
    xt::xarray<float> grad_data = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, height, size}, -1.0F, 1.0F, seed);
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

// synchronize_gradients decides per mesh axis whether a gradient is reduced by reading the parameter's label through
// is_sharded_on_axis, and trusts the all_reduce result's own label afterwards (the relabel-by-index it used to apply
// is gone). On the 1x2 mesh axis 1 is the only axis with more than one device.
namespace {

using MeshMapperConfig = ttnn::distributed::MeshMapperConfig;

bool all_replicate(const ttnn::Tensor& tensor) {
    const auto& placements = tensor.tensor_topology().placements();
    return std::all_of(placements.begin(), placements.end(), [](const MeshMapperConfig::Placement& placement) {
        return std::holds_alternative<MeshMapperConfig::Replicate>(placement);
    });
}

// A replicated [1, 1, 32, 32] parameter (created without a mapper) whose gradient is `grad_data` ([2, 1, 32, 32])
// distributed by `mapper`, so every device holds a different [1, 1, 32, 32] gradient.
ttml::autograd::TensorPtr parameter_with_gradient(
    const xt::xarray<float>& grad_data, const ttnn::distributed::TensorToMesh& mapper) {
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> value = xt::zeros<float>({1U, 1U, 32U, 32U});
    auto parameter = ttml::autograd::create_tensor(
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(value, device, ttnn::Layout::TILE),
        /* requires_grad */ true);
    parameter->set_grad(
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, &mapper));
    return parameter;
}

xt::xarray<float> mean_of_the_two_slices(const xt::xarray<float>& grad_data) {
    return (xt::view(grad_data, xt::range(0, 1), xt::all(), xt::all(), xt::all()) +
            xt::view(grad_data, xt::range(1, 2), xt::all(), xt::all(), xt::all())) *
           0.5F;
}

}  // namespace

TEST_F(N300CommOpsTest, SynchronizeGradientsCollapsedGradientIsAveragedAndReadsReplicate) {
    // The gradient carries the collapsed {2}, [Shard{0}] label of the whole-mesh 1-D mapper. It must read as sharded
    // on axis 1 (negative control: it read as not sharded there before, the one placement sat at index 0), be averaged
    // along axis 1, and come back labelled Replicate on every axis by the collective itself.
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    xt::xarray<float> grad_data =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{2U, 1U, 32U, 32U}, 0.0F, 1.0F, rng());
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0);
    auto parameter = parameter_with_gradient(grad_data, *mapper);

    // A parameter created without a mapper carries one placement per mesh axis.
    EXPECT_EQ(parameter->get_value().tensor_topology().placements().size(), 2U);
    EXPECT_TRUE(all_replicate(parameter->get_value()));
    EXPECT_EQ(parameter->get_grad().tensor_topology().placements().size(), 1U);
    EXPECT_TRUE(ttml::core::distributed::is_sharded_on_axis(parameter->get_grad(), 1U));
    EXPECT_FALSE(ttml::core::distributed::is_sharded_on_axis(parameter->get_grad(), 0U));
    EXPECT_FALSE(ttml::core::distributed::is_sharded_on_axis(parameter->get_value(), 1U));

    ttml::serialization::NamedParameters parameters{{"weight", parameter}};
    ttml::core::distributed::synchronize_gradients(parameters, std::vector<uint32_t>{1U});

    const auto& synced = parameter->get_grad();
    EXPECT_FALSE(ttml::core::distributed::is_sharded_on_axis(synced, 0U));
    EXPECT_FALSE(ttml::core::distributed::is_sharded_on_axis(synced, 1U));
    EXPECT_TRUE(all_replicate(synced));
    auto shards = ttml::core::to_xtensor<float>(synced, ttml::core::IdentityComposer{});
    const xt::xarray<float> expected = mean_of_the_two_slices(grad_data);
    ASSERT_EQ(shards.size(), 2U);
    EXPECT_TRUE(xt::allclose(expected, shards[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected, shards[1], /* rtol */ 1e-2, /* atol */ 1e-2));
}

TEST_F(N300CommOpsTest, SynchronizeGradientsNDGradientIsAveragedAndReadsReplicate) {
    // The same with an N-D [Replicate, Shard{0}] gradient label: averaged along axis 1, [Replicate, Replicate] out.
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    xt::xarray<float> grad_data =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{2U, 1U, 32U, 32U}, 0.0F, 1.0F, rng());
    const auto mapper = ttnn::distributed::create_mesh_mapper(
        *device, MeshMapperConfig{.placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{0}}});
    auto parameter = parameter_with_gradient(grad_data, *mapper);

    EXPECT_EQ(parameter->get_grad().tensor_topology().placements().size(), 2U);
    EXPECT_TRUE(ttml::core::distributed::is_sharded_on_axis(parameter->get_grad(), 1U));
    EXPECT_FALSE(ttml::core::distributed::is_sharded_on_axis(parameter->get_grad(), 0U));

    ttml::serialization::NamedParameters parameters{{"weight", parameter}};
    ttml::core::distributed::synchronize_gradients(parameters, std::vector<uint32_t>{1U});

    const auto& synced = parameter->get_grad();
    EXPECT_EQ(synced.tensor_topology().placements().size(), 2U);
    EXPECT_TRUE(all_replicate(synced));
    auto shards = ttml::core::to_xtensor<float>(synced, ttml::core::IdentityComposer{});
    const xt::xarray<float> expected = mean_of_the_two_slices(grad_data);
    ASSERT_EQ(shards.size(), 2U);
    EXPECT_TRUE(xt::allclose(expected, shards[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected, shards[1], /* rtol */ 1e-2, /* atol */ 1e-2));
}

TEST_F(N300CommOpsTest, IsShardedOnAxisReadsTensorParallelShardsPerAxis) {
    // A TP weight sharded on dim 3 along axis 1, spelled N-D and collapsed, reads as sharded on axis 1 only; the
    // collapsed spelling read as not sharded on axis 1 before this change. A label that does not cover the mesh is
    // refused rather than guessed. A reduction request over the size-1 axis 0 is a no-op (synchronize_tensor's early
    // return) and leaves the TP shard alone.
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> weight = xt::zeros<float>({1U, 1U, 32U, 64U});
    const auto nd_mapper = ttnn::distributed::create_mesh_mapper(
        *device, MeshMapperConfig{.placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{3}}});
    auto nd =
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(weight, device, ttnn::Layout::TILE, nd_mapper.get());
    EXPECT_TRUE(ttml::core::distributed::is_sharded_on_axis(nd, 1U));
    EXPECT_FALSE(ttml::core::distributed::is_sharded_on_axis(nd, 0U));

    const auto collapsed_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 3);
    auto collapsed = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        weight, device, ttnn::Layout::TILE, collapsed_mapper.get());
    EXPECT_EQ(collapsed.tensor_topology().placements().size(), 1U);
    EXPECT_TRUE(ttml::core::distributed::is_sharded_on_axis(collapsed, 1U));
    EXPECT_FALSE(ttml::core::distributed::is_sharded_on_axis(collapsed, 0U));

    auto forged = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(weight, device, ttnn::Layout::TILE);
    const auto coords = forged.tensor_topology().mesh_coords();
    forged.update_tensor_topology(tt::tt_metal::TensorTopology(
        tt::tt_metal::distributed::MeshShape(1U),
        ttsl::SmallVector<MeshMapperConfig::Placement>{MeshMapperConfig::Shard{0}},
        std::vector<tt::tt_metal::distributed::MeshCoordinate>{coords.front()}));
    EXPECT_THROW(ttml::core::distributed::is_sharded_on_axis(forged, 1U), std::runtime_error);
    EXPECT_THROW(ttml::core::distributed::is_sharded_on_axis(nd, 2U), std::runtime_error);

    auto parameter = ttml::autograd::create_tensor(nd, /* requires_grad */ true);
    parameter->set_grad(nd);
    ttml::serialization::NamedParameters parameters{{"weight", parameter}};
    ttml::core::distributed::synchronize_gradients(parameters, std::vector<uint32_t>{0U});
    EXPECT_TRUE(ttml::core::distributed::is_sharded_on_axis(parameter->get_grad(), 1U));
}
