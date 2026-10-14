// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/distributed/rmsnorm_op.hpp"

#include <gtest/gtest.h>

#include <array>
#include <core/xtensor_utils.hpp>
#include <numeric>
#include <optional>
#include <tt-metalium/host_api.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

constexpr float kEpsilon = 1e-5F;

bool has_two_devices() {
    return tt::tt_metal::GetNumAvailableDevices() == 2;
}

struct RMSNormReference {
    xt::xarray<float> out;
    xt::xarray<float> dx;
    xt::xarray<float> dgamma;
};

RMSNormReference rmsnorm_reference(
    const xt::xarray<float>& x, const xt::xarray<float>& gamma, const xt::xarray<float>& dy, float epsilon) {
    xt::xarray<float> rms = xt::sqrt(xt::mean(x * x, {3}, xt::keep_dims) + epsilon);
    xt::xarray<float> gained = gamma * dy / rms;
    xt::xarray<float> scale = xt::mean(x * gained, {3}, xt::keep_dims);
    return RMSNormReference{
        .out = gamma * x / rms,
        .dx = gained - x * scale / (rms * rms),
        .dgamma = xt::sum(dy * x / rms, {0, 1, 2}, xt::keep_dims),
    };
}

// Shards the last dim over `cluster_axis` and replicates over the other mesh axis; std::nullopt shards over all
// devices as a 1D mesh.
ttnn::Tensor shard_last_dim(const xt::xarray<float>& xtensor, std::optional<uint32_t> cluster_axis) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(
        *device, 3, cluster_axis.has_value() ? std::optional<int>(cluster_axis.value()) : std::nullopt);
    return ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(xtensor, device, ttnn::Layout::TILE, mapper.get());
}

// Inverse of shard_last_dim: one full tensor per replica, i.e. per device line along `cluster_axis`.
std::vector<xt::xarray<float>> unshard_last_dim(const ttnn::Tensor& tensor, std::optional<uint32_t> cluster_axis) {
    auto shards = ttml::core::to_xtensor<float>(tensor, ttml::core::IdentityComposer{});
    auto concat = [&](const std::vector<size_t>& indices) {
        std::vector<xt::xarray<float>> line;
        line.reserve(indices.size());
        for (size_t index : indices) {
            line.push_back(shards[index]);
        }
        return ttml::core::concat(line, 3);
    };

    if (!cluster_axis.has_value()) {
        std::vector<size_t> indices(shards.size());
        std::iota(indices.begin(), indices.end(), 0U);
        return {concat(indices)};
    }

    // Shards are returned in row-major mesh order.
    const auto mesh_shape = ttml::autograd::ctx().get_device().shape();
    const size_t rows = mesh_shape[0];
    const size_t cols = mesh_shape[1];
    const bool along_rows = cluster_axis.value() == 0U;
    const size_t num_lines = along_rows ? cols : rows;
    const size_t line_size = along_rows ? rows : cols;

    std::vector<xt::xarray<float>> lines;
    lines.reserve(num_lines);
    for (size_t line = 0; line < num_lines; ++line) {
        std::vector<size_t> indices;
        indices.reserve(line_size);
        for (size_t i = 0; i < line_size; ++i) {
            indices.push_back(along_rows ? i * cols + line : line * cols + i);
        }
        lines.push_back(concat(indices));
    }
    return lines;
}

void expect_all_close(
    const ttnn::Tensor& tensor,
    const xt::xarray<float>& expected,
    std::optional<uint32_t> cluster_axis,
    float rtol,
    float atol) {
    const auto lines = unshard_last_dim(tensor, cluster_axis);
    for (size_t line = 0; line < lines.size(); ++line) {
        EXPECT_TRUE(xt::allclose(lines[line], expected, rtol, atol))
            << "mismatch on replica " << line << ", max abs diff " << xt::amax(xt::abs(lines[line] - expected))();
    }
}

void run_distributed_rmsnorm(
    uint32_t batch, uint32_t seq_len, uint32_t hidden, std::optional<uint32_t> cluster_axis = std::nullopt) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    xt::xarray<float> x = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());
    xt::xarray<float> gamma =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, hidden}, 0.5F, 1.5F, rng());
    xt::xarray<float> dy = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());

    auto tensor = autograd::create_tensor(shard_last_dim(x, cluster_axis), /* requires_grad */ true);
    auto tt_gamma = autograd::create_tensor(shard_last_dim(gamma, cluster_axis), /* requires_grad */ true);

    auto out = ops::distributed::rmsnorm(tensor, tt_gamma, kEpsilon, cluster_axis);
    out->set_grad(shard_last_dim(dy, cluster_axis));
    out->backward();

    ASSERT_TRUE(core::is_tensor_initialized(tensor->get_grad()));
    ASSERT_TRUE(core::is_tensor_initialized(tt_gamma->get_grad()));

    const auto expected = rmsnorm_reference(x, gamma, dy, kEpsilon);
    expect_all_close(out->get_value(), expected.out, cluster_axis, /* rtol */ 3e-2F, /* atol */ 3e-2F);
    expect_all_close(tensor->get_grad(), expected.dx, cluster_axis, /* rtol */ 5e-2F, /* atol */ 5e-2F);
    // dL/dgamma sums over B * S rows, so its absolute error grows with the row count.
    const float dgamma_atol = 5e-2F * static_cast<float>(batch * seq_len) / 32.0F;
    expect_all_close(tt_gamma->get_grad(), expected.dgamma, cluster_axis, /* rtol */ 5e-2F, dgamma_atol);
}

}  // namespace

class TwoDeviceDistributedRMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!has_two_devices()) {
            GTEST_SKIP() << "Skipping test that needs 2 devices";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(2U);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(TwoDeviceDistributedRMSNormTest, Small) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(1, 32, 128);
}

TEST_F(TwoDeviceDistributedRMSNormTest, Batched) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 64, 256);
}

// A wider row makes the full-row mean depend on the fp32 stats rather than on rounding of a short sum.
// 1024 per device stays under the L1 cap of rms_norm_post_all_gather_bw, which keeps the whole local row in L1
// (1184 on Wormhole, 1248 on Blackhole).
TEST_F(TwoDeviceDistributedRMSNormTest, WideHidden) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(1, 64, 2048);
}

TEST_F(TwoDeviceDistributedRMSNormTest, NonTileAlignedSeqLen) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 33, 128);
}

TEST_F(TwoDeviceDistributedRMSNormTest, ExplicitClusterAxis) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(1, 64, 256, /* cluster_axis */ 1U);
}

// Axis 0 of a 1x2 mesh has size 1: hidden is unsharded and replicated on both devices, so the op falls back to
// the single-device kernel and no gather runs.
TEST_F(TwoDeviceDistributedRMSNormTest, TrivialClusterAxisReplicates) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 64, 128, /* cluster_axis */ 0U);
}

TEST_F(TwoDeviceDistributedRMSNormTest, RejectsOutOfRangeClusterAxis) {
    auto x = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 32U, 128U}), std::nullopt));
    auto gamma = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 1U, 128U}), std::nullopt));
    EXPECT_THROW(ttml::ops::distributed::rmsnorm(x, gamma, kEpsilon, /* cluster_axis */ 2U), std::invalid_argument);
}

TEST_F(TwoDeviceDistributedRMSNormTest, RejectsNonTileAlignedLocalHidden) {
    // 96 split over two devices leaves 48 per device.
    auto x = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 32U, 96U}), std::nullopt));
    auto gamma = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 1U, 96U}), std::nullopt));
    EXPECT_THROW(ttml::ops::distributed::rmsnorm(x, gamma, kEpsilon), std::invalid_argument);
}

TEST_F(TwoDeviceDistributedRMSNormTest, RejectsUnshardedGamma) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    auto x = autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 32U, 128U}), std::nullopt));
    auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    auto full_gamma = autograd::create_tensor(core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        xt::ones<float>({1U, 1U, 1U, 128U}), device, ttnn::Layout::TILE, mapper.get()));
    EXPECT_THROW(ops::distributed::rmsnorm(x, full_gamma, kEpsilon), std::invalid_argument);
}

// Runs on a T3K (2x4) or Galaxy (8x4): hidden is sharded over one mesh axis and replicated over the other.
class MeshDistributedRMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices == 32U) {
            mesh_shape_ = tt::tt_metal::distributed::MeshShape(8, 4);
        } else if (num_devices == 8U) {
            mesh_shape_ = tt::tt_metal::distributed::MeshShape(2, 4);
        } else {
            GTEST_SKIP() << "Skipping test that needs an 8 or 32 device mesh";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(num_devices);
        ttml::autograd::ctx().open_device(mesh_shape_.value());
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        if (mesh_shape_.has_value()) {
            ttml::autograd::ctx().close_device();
        }
    }

    std::optional<tt::tt_metal::distributed::MeshShape> mesh_shape_;
};

TEST_F(MeshDistributedRMSNormTest, ShardedAlongAxis1) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 64, 256, /* cluster_axis */ 1U);
}

TEST_F(MeshDistributedRMSNormTest, ShardedAlongAxis0) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 64, 256, /* cluster_axis */ 0U);
}

TEST_F(MeshDistributedRMSNormTest, RejectsMissingClusterAxis) {
    auto x = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 32U, 256U}), 1U));
    auto gamma = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 1U, 256U}), 1U));
    EXPECT_THROW(ttml::ops::distributed::rmsnorm(x, gamma, kEpsilon), std::invalid_argument);
}

class SingleDeviceDistributedRMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// One device: falls back to ops::rmsnorm.
TEST_F(SingleDeviceDistributedRMSNormTest, MatchesReferenceWithoutGather) {
    run_distributed_rmsnorm(2, 64, 128);
}
