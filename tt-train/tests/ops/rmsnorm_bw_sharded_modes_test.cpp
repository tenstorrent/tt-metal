// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the LocalSum and ExternalScale modes of rmsnorm_bw on one device by splitting the hidden dim into
// num_shards chunks. Concatenating the per-chunk LocalSum outputs along the last dim is what the all-gather does
// on a mesh, so ExternalScale on every chunk must reproduce the single-device backward of the full row.

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_program_factory.hpp"
#include "test_utils/random_data.hpp"
#include "test_utils/rmsnorm_reference.hpp"

namespace {

using ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardMode;

constexpr float kEpsilon = 1e-5F;

// dx is bf16 (relative step 2^-8); its row reduction reads fp32 operands through tf32 source registers.
constexpr float kDxRtol = 2e-2F;
constexpr float kDxAtol = 2e-2F;
// ExternalScale and Full dx come from the same kernel arithmetic, so they agree more closely than either agrees with
// the fp32 reference. This catches a wrong cross-shard term, which is small next to dx itself.
constexpr float kDxVsFullAtol = 1e-2F;
// Per-row absolute error of dgamma; the tolerance grows with sqrt(number of rows). The per-row components path
// rounds every row's contribution to bf16 before the sum.
constexpr float kDgammaRtol = 2e-2F;
constexpr float kDgammaRowAtol = 1e-2F;

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

struct ExpectedLayout {
    bool everything_fits_in_l1{};
    bool dgamma_on_chip{};
};

void expect_layout(
    const char* what,
    RMSNormBackwardMode mode,
    const ttnn::Tensor& x,
    const ttnn::Tensor& gamma,
    const std::optional<ttnn::Tensor>& rms,
    const ttnn::Tensor& dy,
    const std::optional<ttnn::Tensor>& external_sums,
    uint32_t full_width,
    ExpectedLayout expected) {
    using namespace ttml::metal::ops::rmsnorm_bw::device;
    const auto layout = compute_layout(
        operation_attributes_t{.mode = mode, .full_width = full_width},
        tensor_args_t{.input = x, .gamma = gamma, .rms = rms, .dL_dout = dy, .external_sums = external_sums});
    EXPECT_EQ(layout.everything_fits_in_l1, expected.everything_fits_in_l1) << what << ": everything_fits_in_l1";
    EXPECT_EQ(layout.dgamma_on_chip, expected.dgamma_on_chip) << what << ": dgamma_on_chip";
}

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

xt::xarray<float> last_dim_range(const xt::xarray<float>& values, uint32_t begin, uint32_t end) {
    return xt::view(values, xt::all(), xt::all(), xt::all(), xt::range(begin, end));
}

void run_sharded_modes(
    uint32_t batch,
    uint32_t seq_len,
    uint32_t hidden,
    uint32_t num_shards,
    ExpectedLayout shard_layout,
    ExpectedLayout full_layout) {
    using namespace ttml;
    ASSERT_EQ(hidden % (num_shards * 32U), 0U);
    auto* device = &autograd::ctx().get_device();
    auto& rng = autograd::ctx().get_generator();

    const xt::xarray<float> x_h = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());
    const xt::xarray<float> gamma_h =
        test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, 1U, hidden}, 0.5F, 1.5F, rng());
    const xt::xarray<float> dy_h = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{batch, 1U, seq_len, hidden}, -1.0F, 1.0F, rng());
    auto x = core::from_xtensor(x_h, device);
    auto gamma = core::from_xtensor(gamma_h, device);
    auto dy = core::from_xtensor(dy_h, device);
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
        const auto x_shard = slice_last_dim(x, begin, end);
        const auto gamma_shard = slice_last_dim(gamma, begin, end);
        const auto dy_shard = slice_last_dim(dy, begin, end);
        if (shard == 0U) {
            expect_layout(
                "ExternalScale shard",
                RMSNormBackwardMode::ExternalScale,
                x_shard,
                gamma_shard,
                rms,
                dy_shard,
                gathered_sums,
                hidden,
                shard_layout);
        }
        auto grads = metal::rmsnorm_bw_external_scale(x_shard, gamma_shard, rms, dy_shard, gathered_sums, hidden);
        dx_shards.push_back(grads[0].value());
        dgamma_shards.push_back(grads[1].value());
    }
    expect_layout("Full", RMSNormBackwardMode::Full, x, gamma, rms, dy, std::nullopt, 0U, full_layout);

    const auto expected = test_utils::rmsnorm_reference(x_h, gamma_h, dy_h, kEpsilon);
    const auto expected_sum0 = test_utils::rmsnorm_reference(
                                   last_dim_range(x_h, 0U, shard_width),
                                   last_dim_range(gamma_h, 0U, shard_width),
                                   last_dim_range(dy_h, 0U, shard_width),
                                   kEpsilon)
                                   .local_sum;

    // The matmul row reduce reads its fp32 operand through tf32 source registers, so the absolute error of the
    // sum grows with the number of summed terms.
    const float sum_atol = 1e-3F * std::sqrt(static_cast<float>(shard_width));
    const auto sum0 = to_host(local_sums[0]);
    const auto sum0_col0 = last_dim_range(sum0, 0U, 1U);
    EXPECT_TRUE(xt::allclose(sum0_col0, expected_sum0, 1e-3F, sum_atol))
        << "LocalSum max abs diff " << xt::amax(xt::abs(sum0_col0 - expected_sum0))();
    EXPECT_TRUE(xt::all(xt::equal(last_dim_range(sum0, 1U, 32U), 0.F)))
        << "LocalSum must leave columns 1..31 zero so that gathered tiles can be summed";

    const auto dx = to_host(ttnn::concat(dx_shards, 3));
    const auto dgamma = to_host(ttnn::concat(dgamma_shards, 3));
    const auto full = metal::rmsnorm_bw(x, gamma, rms, dy);
    const auto full_dx = to_host(full[0].value());
    const auto full_dgamma = to_host(full[1].value());

    EXPECT_TRUE(xt::allclose(dx, expected.dx, kDxRtol, kDxAtol))
        << "ExternalScale dx vs reference max abs diff " << xt::amax(xt::abs(dx - expected.dx))();
    EXPECT_TRUE(xt::allclose(full_dx, expected.dx, kDxRtol, kDxAtol))
        << "Full dx vs reference max abs diff " << xt::amax(xt::abs(full_dx - expected.dx))();
    EXPECT_TRUE(xt::allclose(dx, full_dx, kDxRtol, kDxVsFullAtol))
        << "ExternalScale dx vs Full max abs diff " << xt::amax(xt::abs(dx - full_dx))();
    const float dgamma_atol = test_utils::dgamma_atol(batch * seq_len, kDgammaRowAtol);
    EXPECT_TRUE(xt::allclose(dgamma, expected.dgamma, kDgammaRtol, dgamma_atol))
        << "ExternalScale dgamma vs reference max abs diff " << xt::amax(xt::abs(dgamma - expected.dgamma))();
    EXPECT_TRUE(xt::allclose(full_dgamma, expected.dgamma, kDgammaRtol, dgamma_atol))
        << "Full dgamma vs reference max abs diff " << xt::amax(xt::abs(full_dgamma - expected.dgamma))();
    EXPECT_TRUE(xt::allclose(dgamma, full_dgamma, kDgammaRtol, dgamma_atol))
        << "ExternalScale dgamma vs Full max abs diff " << xt::amax(xt::abs(dgamma - full_dgamma))();
}

// The tensors are views whose padded rows hold real data, so dgamma is only right if the kernel drops the rows past
// the logical sequence length. On-chip dgamma partials must come back as one row per core.
void run_padded_rows_ignored(uint32_t batch, uint32_t seq_len, uint32_t hidden, ExpectedLayout expected_layout) {
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
    expect_layout("Full", RMSNormBackwardMode::Full, x, gamma, rms, dy, std::nullopt, 0U, expected_layout);

    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t expected_num_cores =
        std::min<uint32_t>(grid.x * grid.y, batch * (padded_seq_len / tt::constants::TILE_HEIGHT));
    const auto parts = ttnn::prim::ttml_rmsnorm_bw(x, gamma, rms, dy);
    const auto expected_parts_shape = expected_layout.dgamma_on_chip ? ttnn::Shape({1U, 1U, expected_num_cores, hidden})
                                                                     : ttnn::Shape({batch, 1U, seq_len, hidden});
    EXPECT_EQ(parts[1].logical_shape(), expected_parts_shape);

    const auto dgamma = to_host(metal::rmsnorm_bw(x, gamma, rms, dy)[1].value());
    auto logical_rows = [&](const xt::xarray<float>& values) {
        return xt::eval(xt::view(values, xt::all(), xt::all(), xt::range(0, seq_len), xt::all()));
    };
    const auto expected = test_utils::rmsnorm_reference(logical_rows(x_h), gamma_h, logical_rows(dy_h), kEpsilon);
    const float dgamma_atol = test_utils::dgamma_atol(batch * seq_len, kDgammaRowAtol);
    EXPECT_TRUE(xt::allclose(dgamma, expected.dgamma, kDgammaRtol, dgamma_atol))
        << "dgamma vs reference max abs diff " << xt::amax(xt::abs(dgamma - expected.dgamma))();
}

template <ttnn::DataType TensorType = ttnn::DataType::BFLOAT16>
ttnn::Tensor filled_tensor(std::vector<size_t> shape, float value) {
    const xt::xarray<float> host(shape, value);
    return ttml::core::from_xtensor<float, TensorType>(host, &ttml::autograd::ctx().get_device());
}

constexpr ExpectedLayout kFitsOnChip{.everything_fits_in_l1 = true, .dgamma_on_chip = true};
constexpr ExpectedLayout kStreamOnChip{.everything_fits_in_l1 = false, .dgamma_on_chip = true};
constexpr ExpectedLayout kStreamComponents{.everything_fits_in_l1 = false, .dgamma_on_chip = false};

}  // namespace

TEST_F(RMSNormBwShardedModesTest, PaddedRowsIgnoredFitsInL1) {
    run_padded_rows_ignored(2, 33, 512, kFitsOnChip);
}

// 8192 streams the row but the 256 fp32 accumulator tiles (1 MB) still fit in L1.
TEST_F(RMSNormBwShardedModesTest, PaddedRowsIgnoredStreaming) {
    run_padded_rows_ignored(1, 50, 8192, kStreamOnChip);
}

// 512 fp32 accumulator tiles (2 MB) do not fit, so dgamma falls back to per-row components.
TEST_F(RMSNormBwShardedModesTest, PaddedRowsIgnoredComponentsFallback) {
    run_padded_rows_ignored(1, 50, 16384, kStreamComponents);
}

// 512 tile rows over the grid: every core accumulates full rows before and after each padded tile row.
TEST_F(RMSNormBwShardedModesTest, PaddedRowsIgnoredManyRowsPerCore) {
    run_padded_rows_ignored(256, 33, 512, kFitsOnChip);
}

TEST_F(RMSNormBwShardedModesTest, TwoShards) {
    run_sharded_modes(2, 64, 256, 2, kFitsOnChip, kFitsOnChip);
}

TEST_F(RMSNormBwShardedModesTest, FourShardsNonTileAlignedSeqLen) {
    run_sharded_modes(2, 33, 512, 4, kFitsOnChip, kFitsOnChip);
}

// 256 tile rows exceed the core count, so every core processes several rows in both modes.
TEST_F(RMSNormBwShardedModesTest, ManyRowsPerCore) {
    run_sharded_modes(1, 8192, 512, 2, kFitsOnChip, kFitsOnChip);
}

// 512 tile rows from 256 sequences of 33: several rows per core, each sequence ending in a padded tile row.
TEST_F(RMSNormBwShardedModesTest, ManySequencesSmallWidth) {
    run_sharded_modes(256, 33, 256, 2, kFitsOnChip, kFitsOnChip);
}

// Three tiles per shard: odd Wt runs with block size 1.
TEST_F(RMSNormBwShardedModesTest, OddTilesPerShard) {
    run_sharded_modes(1, 64, 192, 2, kFitsOnChip, kFitsOnChip);
}

// Wan2.2-14B hidden size at TP=2 and TP=4: shards of 2560 and 1280. The unsharded row streams.
TEST_F(RMSNormBwShardedModesTest, WanHiddenTwoShards) {
    run_sharded_modes(1, 128, 5120, 2, kFitsOnChip, kStreamOnChip);
}

TEST_F(RMSNormBwShardedModesTest, WanHiddenFourShards) {
    run_sharded_modes(1, 128, 5120, 4, kFitsOnChip, kStreamOnChip);
}

// 8192 per shard does not fit in L1, so both modes take the streaming path.
TEST_F(RMSNormBwShardedModesTest, StreamingShards) {
    run_sharded_modes(1, 64, 16384, 2, kStreamOnChip, kStreamComponents);
}

// Shards of 4608 stream, and 144 tile rows exceed the core count of every Blackhole grid (at most 130), so part of
// the grid streams a second row in both modes.
TEST_F(RMSNormBwShardedModesTest, StreamingTwoRowsPerCore) {
    run_sharded_modes(1, 4608, 9216, 2, kStreamOnChip, kStreamOnChip);
}

// The same widths at a training-length sequence: 256 tile rows, so every core streams several rows.
TEST_F(RMSNormBwShardedModesTest, NIGHTLY_StreamingManyRowsPerCore) {
    run_sharded_modes(1, 8192, 9216, 2, kStreamOnChip, kStreamOnChip);
}

// 340 fp32 accumulator tiles per shard do not fit next to the streaming buffers, so ExternalScale streams the row
// and writes per-row dgamma components.
TEST_F(RMSNormBwShardedModesTest, ExternalScaleStreamingComponents) {
    run_sharded_modes(1, 64, 21760, 2, kStreamComponents, kStreamComponents);
}

TEST_F(RMSNormBwShardedModesTest, RejectsRmsWiderThanOneTile) {
    auto x = filled_tensor({1U, 1U, 32U, 64U}, 1.F);
    auto gamma = filled_tensor({1U, 1U, 1U, 64U}, 1.F);
    auto wide_rms = filled_tensor({1U, 1U, 32U, 64U}, 1.F);
    EXPECT_THROW(ttnn::prim::ttml_rmsnorm_bw(x, gamma, wide_rms, x), std::runtime_error);
}

// One sums tile is the ungathered LocalSum output. full_width of two shards must not scale dx by it.
TEST_F(RMSNormBwShardedModesTest, RejectsExternalScaleSumWidthMismatch) {
    auto x = filled_tensor({1U, 1U, 32U, 64U}, 1.F);
    auto gamma = filled_tensor({1U, 1U, 1U, 64U}, 1.F);
    auto rms = filled_tensor({1U, 1U, 32U, 1U}, 1.F);
    auto sums = filled_tensor<ttnn::DataType::FLOAT32>({1U, 1U, 32U, 32U}, 0.F);
    EXPECT_THROW(
        ttnn::prim::ttml_rmsnorm_bw(x, gamma, rms, x, sums, RMSNormBackwardMode::ExternalScale, /* full_width */ 128U),
        std::runtime_error);
}

// A small width would otherwise accumulate dgamma on chip. The preallocated buffer selects components.
TEST_F(RMSNormBwShardedModesTest, PreallocatedDgammaComponentsSelectsComponentsLayout) {
    auto x = filled_tensor({1U, 1U, 32U, 64U}, 1.F);
    auto gamma = filled_tensor({1U, 1U, 1U, 64U}, 1.F);
    auto rms = filled_tensor({1U, 1U, 32U, 1U}, 1.F);
    auto components = filled_tensor({1U, 1U, 32U, 64U}, 0.F);
    const auto parts = ttnn::prim::ttml_rmsnorm_bw(
        x,
        gamma,
        rms,
        x,
        /* external_sums */ std::nullopt,
        RMSNormBackwardMode::Full,
        /* full_width */ 0U,
        /* preallocated_da */ std::nullopt,
        components);
    ASSERT_EQ(parts.size(), 2U);
    EXPECT_EQ(parts[1].dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(parts[1].logical_shape(), x.logical_shape());
    EXPECT_EQ(parts[1].buffer()->address(), components.buffer()->address());
}

TEST_F(RMSNormBwShardedModesTest, RejectsPreallocatedDgammaComponentsShape) {
    auto x = filled_tensor({1U, 1U, 32U, 64U}, 1.F);
    auto gamma = filled_tensor({1U, 1U, 1U, 64U}, 1.F);
    auto rms = filled_tensor({1U, 1U, 32U, 1U}, 1.F);
    auto components = filled_tensor({1U, 1U, 32U, 32U}, 0.F);
    EXPECT_THROW(
        ttnn::prim::ttml_rmsnorm_bw(
            x,
            gamma,
            rms,
            x,
            /* external_sums */ std::nullopt,
            RMSNormBackwardMode::Full,
            /* full_width */ 0U,
            /* preallocated_da */ std::nullopt,
            components),
        std::runtime_error);
}
