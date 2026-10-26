// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/distributed/rmsnorm_op.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <core/xtensor_utils.hpp>
#include <numeric>
#include <optional>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_program_factory.hpp"
#include "ops/rmsnorm_op.hpp"
#include "test_utils/random_data.hpp"
#include "test_utils/rmsnorm_reference.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

constexpr float kEpsilon = 1e-5F;

// The system mesh is bounded by the mesh graph descriptor, not by the chip count: on a 4-chip box run
// under a 1x2 descriptor only 2 devices are usable, and opening a bigger mesh is a hard failure.
size_t system_mesh_size() {
    return tt::tt_metal::distributed::SystemMesh::instance().shape().mesh_size();
}

bool has_at_least_devices(size_t num_devices) {
    return system_mesh_size() >= num_devices;
}

// Outputs and dx are bf16 (relative step 2^-8); the row reductions behind them read fp32 operands through tf32.
constexpr float kOutRtol = 1e-2F;
constexpr float kOutAtol = 1e-2F;
constexpr float kDxRtol = 2e-2F;
constexpr float kDxAtol = 2e-2F;
// Per-row absolute error of dgamma; the tolerance grows with sqrt(number of rows).
constexpr float kDgammaRtol = 2e-2F;
constexpr float kDgammaRowAtol = 1e-2F;
// Two device results are each rounded to bf16, so they can differ by two bf16 steps (relative 2^-7). dx from the
// distributed and the single-device backward differ only in how the row sum is assembled, so its absolute tolerance is
// tighter than against the fp32 reference; this is what catches a wrong cross-shard term.
constexpr float kDeviceVsDeviceRtol = 2e-2F;
constexpr float kDeviceVsDeviceDxAtol = 1e-2F;

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
    const char* what,
    const ttnn::Tensor& tensor,
    const xt::xarray<float>& expected,
    std::optional<uint32_t> cluster_axis,
    float rtol,
    float atol) {
    const auto lines = unshard_last_dim(tensor, cluster_axis);
    for (size_t line = 0; line < lines.size(); ++line) {
        EXPECT_TRUE(xt::allclose(lines[line], expected, rtol, atol))
            << what << " mismatch on replica " << line << ", max abs diff "
            << xt::amax(xt::abs(lines[line] - expected))();
    }
}

void run_distributed_rmsnorm(
    uint32_t batch,
    uint32_t seq_len,
    uint32_t hidden,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    bool zero_first_row = false) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    xt::xarray<float> x = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());
    xt::xarray<float> gamma =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, hidden}, 0.5F, 1.5F, rng());
    xt::xarray<float> dy = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());
    if (zero_first_row) {
        // rms = sqrt(eps) on this row; every gradient must stay finite.
        xt::view(x, 0, 0, 0, xt::all()) = 0.F;
    }

    auto tensor = autograd::create_tensor(shard_last_dim(x, cluster_axis), /* requires_grad */ true);
    auto tt_gamma = autograd::create_tensor(shard_last_dim(gamma, cluster_axis), /* requires_grad */ true);

    auto out = ops::distributed::rmsnorm(tensor, tt_gamma, kEpsilon, cluster_axis);
    out->set_grad(shard_last_dim(dy, cluster_axis));
    out->backward();

    ASSERT_TRUE(core::is_tensor_initialized(tensor->get_grad()));
    ASSERT_TRUE(core::is_tensor_initialized(tt_gamma->get_grad()));

    const auto expected = test_utils::rmsnorm_reference(x, gamma, dy, kEpsilon);
    expect_all_close("out", out->get_value(), expected.out, cluster_axis, kOutRtol, kOutAtol);
    expect_all_close("dx", tensor->get_grad(), expected.dx, cluster_axis, kDxRtol, kDxAtol);
    const float dgamma_atol = test_utils::dgamma_atol(batch * seq_len, kDgammaRowAtol);
    expect_all_close("dgamma", tt_gamma->get_grad(), expected.dgamma, cluster_axis, kDgammaRtol, dgamma_atol);
}

// Compares the distributed op over the whole mesh (1D) with ops::rmsnorm on the unsharded row, replicated on every
// device.
void run_against_single_device_rmsnorm(uint32_t batch, uint32_t seq_len, uint32_t hidden) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    auto& rng = autograd::ctx().get_generator();
    const xt::xarray<float> x = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());
    const xt::xarray<float> gamma =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, hidden}, 0.5F, 1.5F, rng());
    const xt::xarray<float> dy = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());

    auto sharded_x = autograd::create_tensor(shard_last_dim(x, std::nullopt), /* requires_grad */ true);
    auto sharded_gamma = autograd::create_tensor(shard_last_dim(gamma, std::nullopt), /* requires_grad */ true);
    auto sharded_out = ops::distributed::rmsnorm(sharded_x, sharded_gamma, kEpsilon);
    sharded_out->set_grad(shard_last_dim(dy, std::nullopt));
    sharded_out->backward();

    auto replicate = [&](const xt::xarray<float>& values) {
        auto mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(values, device, ttnn::Layout::TILE, mapper.get());
    };
    auto full_x = autograd::create_tensor(replicate(x), /* requires_grad */ true);
    auto full_gamma = autograd::create_tensor(replicate(gamma), /* requires_grad */ true);
    auto full_out = ops::rmsnorm(full_x, full_gamma, kEpsilon);
    full_out->set_grad(replicate(dy));
    full_out->backward();

    auto first_replica = [](const ttnn::Tensor& tensor) {
        return core::to_xtensor<float>(tensor, core::IdentityComposer{}).front();
    };
    expect_all_close(
        "out",
        sharded_out->get_value(),
        first_replica(full_out->get_value()),
        std::nullopt,
        kDeviceVsDeviceRtol,
        kOutAtol);
    expect_all_close(
        "dx",
        sharded_x->get_grad(),
        first_replica(full_x->get_grad()),
        std::nullopt,
        kDeviceVsDeviceRtol,
        kDeviceVsDeviceDxAtol);
    expect_all_close(
        "dgamma",
        sharded_gamma->get_grad(),
        first_replica(full_gamma->get_grad()),
        std::nullopt,
        kDgammaRtol,
        test_utils::dgamma_atol(batch * seq_len, kDgammaRowAtol));
}

// Layout the ExternalScale backward picks for one device's [batch, 1, seq_len, local_width] shard.
ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardLayout external_scale_layout(
    uint32_t batch, uint32_t seq_len, uint32_t local_width, uint32_t num_shards) {
    using namespace ttml::metal::ops::rmsnorm_bw::device;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto x = ttml::core::zeros(ttnn::Shape({batch, 1U, seq_len, local_width}), device);
    const auto gamma = ttml::core::zeros(ttnn::Shape({1U, 1U, 1U, local_width}), device);
    const auto sums = ttml::core::zeros(
        ttnn::Shape({batch, 1U, seq_len, num_shards * tt::constants::TILE_WIDTH}), device, ttnn::DataType::FLOAT32);
    return compute_layout(
        operation_attributes_t{.mode = RMSNormBackwardMode::ExternalScale, .full_width = num_shards * local_width},
        tensor_args_t{.input = x, .gamma = gamma, .dL_dout = x, .external_sums = sums});
}

}  // namespace

class TwoDeviceDistributedRMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!has_at_least_devices(2U)) {
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
// A local width past the forward's whole-row L1 cap is rejected before the ttnn forward
// (see RejectsLocalWidthBeyondForwardL1). RMSNormBwShardedModesTest covers the streaming backward.
TEST_F(TwoDeviceDistributedRMSNormTest, WideHidden) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(1, 64, 2048);
}

TEST_F(TwoDeviceDistributedRMSNormTest, BatchedNonTileAlignedMatchesSingleDevice) {
    SKIP_FOR_WATCHER();
    run_against_single_device_rmsnorm(2, 33, 1024);
}

// 4480 per device sits just under the forward's whole-row L1 cap, where the ExternalScale backward already streams.
TEST_F(TwoDeviceDistributedRMSNormTest, NearForwardWidthCapMatchesSingleDevice) {
    SKIP_FOR_WATCHER();
    constexpr uint32_t kSeqLen = 256U;
    constexpr uint32_t kLocalWidth = 4480U;
    EXPECT_FALSE(external_scale_layout(1U, kSeqLen, kLocalWidth, 2U).everything_fits_in_l1)
        << "the ExternalScale backward should stream a " << kLocalWidth << "-wide shard";
    run_against_single_device_rmsnorm(1U, kSeqLen, 2U * kLocalWidth);
}

// Wan2.2-14B hidden at a training-length sequence: 256 tile rows per device.
TEST_F(TwoDeviceDistributedRMSNormTest, LongSequenceWanHidden) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(1, 8192, 5120);
}

TEST_F(TwoDeviceDistributedRMSNormTest, LongSequenceWanHiddenMatchesSingleDevice) {
    SKIP_FOR_WATCHER();
    run_against_single_device_rmsnorm(1, 8192, 5120);
}

TEST_F(TwoDeviceDistributedRMSNormTest, ZeroRowStaysFinite) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 32, 256, std::nullopt, /* zero_first_row */ true);
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

// 32768 split over two devices is 16384 per device, past the forward's whole-row L1 cap.
TEST_F(TwoDeviceDistributedRMSNormTest, RejectsLocalWidthBeyondForwardL1) {
    constexpr uint32_t kFullWidth = 32768U;
    auto x = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 32U, kFullWidth}), std::nullopt));
    auto gamma = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 1U, kFullWidth}), std::nullopt));
    EXPECT_THROW(ttml::ops::distributed::rmsnorm(x, gamma, kEpsilon), std::invalid_argument);
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

// Runs on a Blackhole QuietBox 2 (4 chips): TP=4 over a 1x4 line.
class FourDeviceDistributedRMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!has_at_least_devices(4U)) {
            GTEST_SKIP() << "Skipping test that needs 4 devices";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(4U);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 4));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(FourDeviceDistributedRMSNormTest, Small) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(1, 32, 512);
}

TEST_F(FourDeviceDistributedRMSNormTest, NonTileAlignedSeqLen) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 33, 512);
}

// Wan2.2-14B hidden size: 1280 per device.
TEST_F(FourDeviceDistributedRMSNormTest, WanHidden) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(1, 256, 5120);
}

// Runs on a Blackhole QuietBox 2 as a 2x2 mesh: hidden is sharded over one axis and replicated over the other.
class TwoByTwoMeshDistributedRMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!has_at_least_devices(4U)) {
            GTEST_SKIP() << "Skipping test that needs 4 devices";
        }
        ttml::ttnn_fixed::distributed::enable_fabric(4U);
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(2, 2));
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(TwoByTwoMeshDistributedRMSNormTest, ShardedAlongAxis1) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 64, 256, /* cluster_axis */ 1U);
}

TEST_F(TwoByTwoMeshDistributedRMSNormTest, ShardedAlongAxis0) {
    SKIP_FOR_WATCHER();
    run_distributed_rmsnorm(2, 64, 256, /* cluster_axis */ 0U);
}

TEST_F(TwoByTwoMeshDistributedRMSNormTest, RejectsMissingClusterAxis) {
    auto x = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 32U, 256U}), 1U));
    auto gamma = ttml::autograd::create_tensor(shard_last_dim(xt::ones<float>({1U, 1U, 1U, 256U}), 1U));
    EXPECT_THROW(ttml::ops::distributed::rmsnorm(x, gamma, kEpsilon), std::invalid_argument);
}

// Runs on a T3K (2x4) or Galaxy (8x4): hidden is sharded over one mesh axis and replicated over the other.
class MeshDistributedRMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto num_devices = system_mesh_size();
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
