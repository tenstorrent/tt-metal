// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the LocalSum and ExternalScale modes of rmsnorm_bw on one device by splitting the hidden dim into
// num_shards chunks. Concatenating the per-chunk LocalSum outputs along the last dim is what the all-gather does
// on a mesh, so ExternalScale on every chunk must reproduce the single-device backward of the full row.

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "test_utils/random_data.hpp"

namespace {

constexpr float kEpsilon = 1e-5F;

class RMSNormBwShardedModesTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

ttnn::Tensor slice_last_dim(const ttnn::Tensor& tensor, uint32_t begin, uint32_t end) {
    const auto& shape = tensor.logical_shape();
    return ttnn::slice(
        tensor,
        ttsl::SmallVector<uint32_t>{0U, 0U, 0U, begin},
        ttsl::SmallVector<uint32_t>{shape[0], shape[1], shape[2], end},
        ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
}

xt::xarray<float> to_host(const ttnn::Tensor& tensor) {
    return ttml::core::to_xtensor<float>(tensor);
}

void run_sharded_modes(uint32_t batch, uint32_t seq_len, uint32_t hidden, uint32_t num_shards) {
    using namespace ttml;
    ASSERT_EQ(hidden % (num_shards * 32U), 0U);
    auto* device = &autograd::ctx().get_device();
    auto& rng = autograd::ctx().get_generator();

    auto x = core::from_xtensor(
        test_utils::make_uniform_xarray<float>(
            std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng()),
        device);
    auto gamma = core::from_xtensor(
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, hidden}, 0.5F, 1.5F, rng()),
        device);
    auto dy = core::from_xtensor(
        test_utils::make_uniform_xarray<float>(
            std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng()),
        device);
    auto rms = metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEpsilon)[1].value();

    const uint32_t shard_width = hidden / num_shards;
    std::vector<ttnn::Tensor> local_sums;
    for (uint32_t shard = 0; shard < num_shards; ++shard) {
        const uint32_t begin = shard * shard_width;
        const uint32_t end = begin + shard_width;
        local_sums.push_back(metal::rmsnorm_bw_local_sum(
            slice_last_dim(x, begin, end), slice_last_dim(gamma, begin, end), slice_last_dim(dy, begin, end)));
        EXPECT_EQ(local_sums.back().dtype(), ttnn::DataType::FLOAT32);
    }
    auto gathered_sums = ttnn::concat(local_sums, /* dim */ 3);

    std::vector<ttnn::Tensor> dx_shards;
    std::vector<ttnn::Tensor> dgamma_shards;
    for (uint32_t shard = 0; shard < num_shards; ++shard) {
        const uint32_t begin = shard * shard_width;
        const uint32_t end = begin + shard_width;
        auto grads = metal::rmsnorm_bw_external_scale(
            slice_last_dim(x, begin, end),
            slice_last_dim(gamma, begin, end),
            rms,
            slice_last_dim(dy, begin, end),
            gathered_sums,
            hidden);
        dx_shards.push_back(grads[0].value());
        dgamma_shards.push_back(grads[1].value());
    }

    // Reference: the fp32 formula on the bf16 values the device actually read.
    const auto x_h = to_host(x);
    const auto gamma_h = to_host(gamma);
    const auto dy_h = to_host(dy);
    const xt::xarray<float> rms_h = xt::sqrt(xt::mean(x_h * x_h, {3}, xt::keep_dims) + kEpsilon);
    const xt::xarray<float> gained = gamma_h * dy_h / rms_h;
    const xt::xarray<float> scale = xt::mean(x_h * gained, {3}, xt::keep_dims);
    const xt::xarray<float> expected_dx = gained - x_h * scale / (rms_h * rms_h);
    const xt::xarray<float> expected_dgamma = xt::sum(dy_h * x_h / rms_h, {0, 1, 2}, xt::keep_dims);
    const xt::xarray<float> expected_sum0 = xt::sum(
        xt::view(x_h * gamma_h * dy_h, xt::all(), xt::all(), xt::all(), xt::range(0, shard_width)), {3}, xt::keep_dims);

    // The matmul row reduce reads its fp32 operand through tf32 source registers, so the absolute error of the
    // sum grows with the number of summed terms.
    const float sum_atol = 1e-3F * std::sqrt(static_cast<float>(shard_width));
    const auto sum0 = to_host(local_sums[0]);
    EXPECT_TRUE(
        xt::allclose(xt::view(sum0, xt::all(), xt::all(), xt::all(), xt::range(0, 1)), expected_sum0, 1e-3F, sum_atol))
        << "LocalSum max abs diff "
        << xt::amax(xt::abs(xt::view(sum0, xt::all(), xt::all(), xt::all(), xt::range(0, 1)) - expected_sum0))();
    EXPECT_TRUE(xt::all(xt::equal(xt::view(sum0, xt::all(), xt::all(), xt::all(), xt::range(1, 32)), 0.F)))
        << "LocalSum must leave columns 1..31 zero so that gathered tiles can be summed";

    const auto dx = to_host(ttnn::concat(dx_shards, 3));
    const auto dgamma = to_host(ttnn::concat(dgamma_shards, 3));
    const auto full = metal::rmsnorm_bw(x, gamma, rms, dy);
    const auto full_dx = to_host(full[0].value());
    const auto full_dgamma = to_host(full[1].value());

    EXPECT_TRUE(xt::allclose(dx, expected_dx, 2e-2F, 2e-2F))
        << "dx vs reference max abs diff " << xt::amax(xt::abs(dx - expected_dx))();
    EXPECT_TRUE(xt::allclose(dx, full_dx, 2e-2F, 1e-2F))
        << "dx vs Full mode max abs diff " << xt::amax(xt::abs(dx - full_dx))();
    const float dgamma_atol = 1e-2F * std::sqrt(static_cast<float>(batch * seq_len));
    EXPECT_TRUE(xt::allclose(dgamma, expected_dgamma, 2e-2F, dgamma_atol))
        << "dgamma vs reference max abs diff " << xt::amax(xt::abs(dgamma - expected_dgamma))();
    EXPECT_TRUE(xt::allclose(dgamma, full_dgamma, 2e-2F, dgamma_atol))
        << "dgamma vs Full mode max abs diff " << xt::amax(xt::abs(dgamma - full_dgamma))();
}

// The tensors are views whose padded rows hold real data, so dgamma is only right if the kernel drops the rows past
// the logical sequence length. The dgamma partials must come back as one tile row per core.
void run_padded_rows_ignored(uint32_t batch, uint32_t seq_len, uint32_t hidden, bool expect_on_chip_dgamma) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    auto& rng = autograd::ctx().get_generator();
    const uint32_t padded_seq_len = tt::round_up(seq_len, tt::constants::TILE_HEIGHT);
    ASSERT_NE(padded_seq_len, seq_len);

    const std::array<std::size_t, 4> padded_shape{batch, 1U, padded_seq_len, hidden};
    xt::xarray<float> x_h = test_utils::make_uniform_xarray<float>(padded_shape, -1.0F, 1.0F, rng());
    xt::xarray<float> dy_h = test_utils::make_uniform_xarray<float>(padded_shape, -1.0F, 1.0F, rng());
    xt::view(dy_h, xt::all(), xt::all(), xt::range(seq_len, padded_seq_len), xt::all()) = 50.F;
    const xt::xarray<float> gamma_h =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, hidden}, 0.5F, 1.5F, rng());

    auto as_logical = [&](const xt::xarray<float>& values) {
        auto tensor = core::from_xtensor(values, device);
        return ttnn::reshape(tensor, ttnn::Shape({batch, 1U, seq_len, hidden}), tensor.padded_shape());
    };
    auto x = as_logical(x_h);
    auto dy = as_logical(dy_h);
    auto gamma = core::from_xtensor(gamma_h, device);
    auto rms = metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEpsilon)[1].value();

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t expected_num_cores =
        std::min<uint32_t>(grid.x * grid.y, batch * (padded_seq_len / tt::constants::TILE_HEIGHT));
    const auto parts = ttnn::prim::ttml_rmsnorm_bw(x, gamma, rms, dy);
    const auto expected_parts_shape =
        expect_on_chip_dgamma ? ttnn::Shape({1U, 1U, expected_num_cores * tt::constants::TILE_HEIGHT, hidden})
                              : ttnn::Shape({batch, 1U, seq_len, hidden});
    EXPECT_EQ(parts[1].logical_shape(), expected_parts_shape);

    const auto dgamma = to_host(metal::rmsnorm_bw(x, gamma, rms, dy)[1].value());
    const auto x_bf16 = xt::eval(
        xt::view(to_host(core::from_xtensor(x_h, device)), xt::all(), xt::all(), xt::range(0, seq_len), xt::all()));
    const auto dy_bf16 = xt::eval(
        xt::view(to_host(core::from_xtensor(dy_h, device)), xt::all(), xt::all(), xt::range(0, seq_len), xt::all()));
    const xt::xarray<float> rms_ref = xt::sqrt(xt::mean(x_bf16 * x_bf16, {3}, xt::keep_dims) + kEpsilon);
    const xt::xarray<float> expected_dgamma = xt::sum(dy_bf16 * x_bf16 / rms_ref, {0, 1, 2}, xt::keep_dims);
    const float dgamma_atol = 1e-2F * std::sqrt(static_cast<float>(batch * seq_len));
    EXPECT_TRUE(xt::allclose(dgamma, expected_dgamma, 2e-2F, dgamma_atol))
        << "dgamma vs reference max abs diff " << xt::amax(xt::abs(dgamma - expected_dgamma))();
}

}  // namespace

TEST_F(RMSNormBwShardedModesTest, PaddedRowsIgnoredFitsInL1) {
    run_padded_rows_ignored(2, 33, 512, /* expect_on_chip_dgamma */ true);
}

// 8192 streams the row but the 256 fp32 accumulator tiles (1 MB) still fit in L1.
TEST_F(RMSNormBwShardedModesTest, PaddedRowsIgnoredStreaming) {
    run_padded_rows_ignored(1, 50, 8192, /* expect_on_chip_dgamma */ true);
}

// 512 fp32 accumulator tiles (2 MB) do not fit, so dgamma falls back to per-row components.
TEST_F(RMSNormBwShardedModesTest, PaddedRowsIgnoredComponentsFallback) {
    run_padded_rows_ignored(1, 50, 16384, /* expect_on_chip_dgamma */ false);
}

TEST_F(RMSNormBwShardedModesTest, TwoShards) {
    run_sharded_modes(2, 64, 256, 2);
}

TEST_F(RMSNormBwShardedModesTest, FourShardsNonTileAlignedSeqLen) {
    run_sharded_modes(2, 33, 512, 4);
}

// 256 tile rows exceed the core count, so every core processes several rows in both modes.
TEST_F(RMSNormBwShardedModesTest, ManyRowsPerCore) {
    run_sharded_modes(1, 8192, 512, 2);
}

// Three tiles per shard: odd Wt runs with block size 1.
TEST_F(RMSNormBwShardedModesTest, OddTilesPerShard) {
    run_sharded_modes(1, 64, 192, 2);
}

// Wan2.2-14B hidden size at TP=2 and TP=4: shards of 2560 and 1280.
TEST_F(RMSNormBwShardedModesTest, WanHiddenTwoShards) {
    run_sharded_modes(1, 128, 5120, 2);
}

TEST_F(RMSNormBwShardedModesTest, WanHiddenFourShards) {
    run_sharded_modes(1, 128, 5120, 4);
}

// 8192 per shard does not fit in L1, so both modes take the streaming path.
TEST_F(RMSNormBwShardedModesTest, StreamingShards) {
    run_sharded_modes(1, 64, 16384, 2);
}
