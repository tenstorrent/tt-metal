// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Test for ring_shift distributed operation.
 *
 * Ring shift moves tensor data between adjacent devices in a ring:
 * - Forward: device i sends to device (i+1) % ring_size
 * - Backward: device i sends to device (i-1+ring_size) % ring_size
 */

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <umd/device/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

auto check_32_chips() {
    auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
    auto all_chips = cluster_desc->get_all_chips();
    return all_chips.size() == 32;
}

class GalaxyRingShiftTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!check_32_chips()) {
            GTEST_SKIP() << "Skipping Galaxy specific tests";
        }
    }
};

static void TestRingShift(
    const uint32_t mesh_rows,
    const uint32_t mesh_cols,
    const size_t batch,
    const size_t seq,
    const size_t hidden,
    const uint32_t cluster_axis,
    const uint32_t shard_dim,
    const bool forward,
    const bool test_backward_grad = false,
    const float rtol = 1e-3F,
    const float atol = 1e-5F) {
    using namespace ttml;

    const uint32_t num_devices = mesh_rows * mesh_cols;

    // Setup device
    ttnn_fixed::distributed::enable_fabric(num_devices);
    autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(mesh_rows, mesh_cols));
    autograd::ctx().set_seed(42);

    auto* device = &autograd::ctx().get_device();

    // Create input tensor with unique values - full tensor that will be sharded
    std::vector<float> test_data_vec(batch * seq * hidden);
    auto& rng = autograd::ctx().get_generator();
    const auto seed = rng();
    core::parallel_generate(
        std::span{test_data_vec.data(), test_data_vec.size()},
        []() { return std::uniform_real_distribution<float>{0.F, 2.F}; },
        seed);

    xt::xarray<float> test_data = xt::adapt(test_data_vec);
    const xt::xarray<float> xtensor = test_data.reshape({batch, 1UL, seq, hidden});

    // Shard tensor across devices along the appropriate dimension
    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, shard_dim, cluster_axis);
    const auto tt_tensor =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
    auto tensor = autograd::create_tensor(tt_tensor);

    // Get original sharded tensors for comparison
    const auto original_xtensors = core::to_xtensor<float>(tensor->get_value(), core::IdentityComposer{});

    // Perform ring shift
    const auto shifted_tensor = ops::distributed::ring_shift(tensor, cluster_axis, forward);

    // Get output back to xtensor
    const auto shifted_xtensors = core::to_xtensor<float>(shifted_tensor->get_value(), core::IdentityComposer{});

    // Verify shape is preserved
    for (size_t i = 0; i < num_devices; ++i) {
        EXPECT_EQ(shifted_xtensors[i].shape(), original_xtensors[i].shape()) << "Shape mismatch on device " << i;
    }

    // Verify ring shift correctness:
    // After forward shift, device i should have data from device (i-1+ring_size) % ring_size
    // After backward shift, device i should have data from device (i+1) % ring_size
    for (uint32_t row = 0; row < mesh_rows; ++row) {
        for (uint32_t col = 0; col < mesh_cols; ++col) {
            const size_t device_idx = row * mesh_cols + col;

            uint32_t src_row = row;
            uint32_t src_col = col;
            if (cluster_axis == 0) {
                src_row = forward ? (row + mesh_rows - 1) % mesh_rows : (row + 1) % mesh_rows;
            } else {
                src_col = forward ? (col + mesh_cols - 1) % mesh_cols : (col + 1) % mesh_cols;
            }
            size_t src_device_idx = src_row * mesh_cols + src_col;

            EXPECT_TRUE(xt::allclose(original_xtensors[src_device_idx], shifted_xtensors[device_idx], rtol, atol))
                << "Value mismatch on device " << device_idx << " (expected data from device " << src_device_idx << ")";
        }
    }

    // Optionally test backward gradient flow
    if (test_backward_grad) {
        xt::xarray<float> grad_data = xt::empty<float>(xtensor.shape());
        const auto seed = rng();
        core::parallel_generate(
            std::span{grad_data.data(), grad_data.size()},
            []() { return std::uniform_real_distribution<float>{0.F, 1.F}; },
            seed);

        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, shard_dim, cluster_axis);
        const auto tt_grad_tensor =
            core::from_xtensor<float, ttnn::DataType::BFLOAT16>(grad_data, device, ttnn::Layout::TILE, mapper.get());

        const auto original_grad_xtensors = core::to_xtensor<float>(tt_grad_tensor, core::IdentityComposer{});

        shifted_tensor->set_grad(tt_grad_tensor);
        shifted_tensor->backward();
        const auto grad_xtensors = core::to_xtensor<float>(tensor->get_grad(), core::IdentityComposer{});

        // Gradient flows in opposite direction
        for (uint32_t row = 0; row < mesh_rows; ++row) {
            for (uint32_t col = 0; col < mesh_cols; ++col) {
                const size_t device_idx = row * mesh_cols + col;

                uint32_t src_row = row;
                uint32_t src_col = col;
                if (cluster_axis == 0) {
                    src_row = forward ? (row + 1) % mesh_rows : (row + mesh_rows - 1) % mesh_rows;
                } else {
                    src_col = forward ? (col + 1) % mesh_cols : (col + mesh_cols - 1) % mesh_cols;
                }
                const size_t src_device_idx = src_row * mesh_cols + src_col;

                EXPECT_EQ(grad_xtensors[device_idx].shape(), original_grad_xtensors[device_idx].shape());
                EXPECT_TRUE(xt::allclose(original_grad_xtensors[src_device_idx], grad_xtensors[device_idx], rtol, atol))
                    << "Gradient mismatch on device " << device_idx;
            }
        }
    }

    autograd::ctx().close_device();
}

TEST_F(GalaxyRingShiftTest, ForwardAlongColumns) {
    // hidden=64 sharded across 8 cols -> 8 per device
    TestRingShift(1, 32, 1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, /*forward=*/true);
}

TEST_F(GalaxyRingShiftTest, ForwardAlongRows) {
    // batch=4 sharded across 4 rows -> 1 per device
    TestRingShift(4, 8, 4, 32, 64, /*cluster_axis=*/0, /*shard_dim=*/0, /*forward=*/true);
}

TEST_F(GalaxyRingShiftTest, ForwardBig) {
    TestRingShift(
        4, 8, 16, 1024, 8192, /*cluster_axis=*/1, /*shard_dim=*/3, /*forward=*/false, /*test_backward_grad=*/false);
}

TEST_F(GalaxyRingShiftTest, Llama8bSeqLen8192Tp8Cp8Bs16) {
    TestRingShift(
        4, 8, 16, 1024, 512, /*cluster_axis=*/1, /*shard_dim=*/0, /*forward=*/false, /*test_backward_grad=*/false);
}

TEST_F(GalaxyRingShiftTest, Llama8bSeqLen8192Tp4Cp4Bs16) {
    TestRingShift(
        8, 4, 16, 1024, 512, /*cluster_axis=*/1, /*shard_dim=*/0, /*forward=*/false, /*test_backward_grad=*/false);
}

TEST_F(GalaxyRingShiftTest, BackwardAlongColumns) {
    TestRingShift(4, 8, 1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, /*forward=*/false);
}

TEST_F(GalaxyRingShiftTest, ForwardWithGradient) {
    TestRingShift(4, 8, 1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, /*forward=*/true, /*test_backward_grad=*/true);
}

TEST_F(GalaxyRingShiftTest, BackwardWithGradient) {
    TestRingShift(4, 8, 1, 32, 64, /*cluster_axis=*/1, /*shard_dim=*/3, /*forward=*/false, /*test_backward_grad=*/true);
}

TEST_F(GalaxyRingShiftTest, BackwardBig) {
    TestRingShift(
        4, 8, 16, 1024, 8192, /*cluster_axis=*/1, /*shard_dim=*/3, /*forward=*/true, /*test_backward_grad=*/true);
}
