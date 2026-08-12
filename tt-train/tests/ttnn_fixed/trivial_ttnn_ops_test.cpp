// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_fixed/trivial_ttnn_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <umd/device/cluster.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class TrivialTnnFixedTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(TrivialTnnFixedTest, TestMaxNegativeOne) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(24, -1.F);
    auto shape = ttnn::Shape({1, 2, 3, 4});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto res = ttnn::max(tensor, /* dim */ 3, /* keepdim */ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_EQ(res_vector.size(), 6);
    bool all_equal =
        std::all_of(res_vector.begin(), res_vector.end(), [](float v) { return std::fabs(v + 1.F) <= 1e-2F; });
    EXPECT_TRUE(all_equal);
}

TEST_F(TrivialTnnFixedTest, TestMaxNegativeBatch) {
    auto* device = &ttml::autograd::ctx().get_device();

    auto shape = ttnn::Shape({4, 1, 1, 4});
    std::vector<float> data(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            data[i * 4 + j] = -static_cast<float>(i + 1);
        }
    }
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto res = ttnn::max(tensor, /* dim */ 3, /* keepdim */ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_EQ(res_vector.size(), 4);
    bool all_equal = true;
    for (int i = 0; i < 4 && all_equal; ++i) {
        if (std::fabs(res_vector[i] - (-static_cast<float>(i + 1))) > 1e-2) {
            all_equal = false;
        }
    }
    EXPECT_TRUE(all_equal);
}

TEST_F(TrivialTnnFixedTest, TestStableSoftmax_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 2U;
    std::vector<float> data(batch_size * features);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = 100.F + static_cast<float>(i);
    }
    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], 100.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], 101.F, 1e-2);

    auto res = ttml::ttnn_fixed::softmax(tensor, /* dim */ 3);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_NEAR(res_vector[0], 0.2689F, 2e-2);
    EXPECT_NEAR(res_vector[1], 0.7311F, 2e-2);
}

TEST_F(TrivialTnnFixedTest, TestOriginalStableSoftmax_AllNegative) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 2U;
    std::vector<float> data(batch_size * features);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = -100.F + static_cast<float>(i);
    }
    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], -100.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], -99.F, 1e-2);
    auto compute_kernel_config = ttml::core::ComputeKernelConfig::precise();
    auto res = ttnn::softmax(
        tensor,
        /* dim */ 3,
        /*memory_config */ std::nullopt,
        compute_kernel_config,
        /*stable*/ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_NEAR(res_vector[0], 0.2689F, 2e-2);
    EXPECT_NEAR(res_vector[1], 0.7311F, 2e-2);
}

TEST_F(TrivialTnnFixedTest, TestStableSoftmax_2) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 10U;
    std::vector<float> data(batch_size * features, 0.F);
    data[0] = 1.0F;
    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], 1.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], 0.F, 1e-2);

    auto res = ttml::ttnn_fixed::softmax(tensor, /* dim */ 3);
    auto res_vector = ttml::core::to_vector(res);

    auto exp_sum = 0.0F;
    for (auto& elem : data) {
        exp_sum += std::exp(elem);
    }

    for (int i = 0; i < res_vector.size(); ++i) {
        EXPECT_NEAR(res_vector[i], std::exp(data[i]) / exp_sum, 1e-2);
    }
}

TEST_F(TrivialTnnFixedTest, TestSumOverBatch_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 10U;
    const size_t features = 4U;
    std::vector<float> data(batch_size * features);
    std::iota(data.begin(), data.end(), 0);

    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_shape = tensor.logical_shape();
    EXPECT_EQ(tensor_shape[0], batch_size);
    EXPECT_EQ(tensor_shape[1], 1U);
    EXPECT_EQ(tensor_shape[2], 1U);
    EXPECT_EQ(tensor_shape[3], features);

    auto result = ttml::ttnn_fixed::sum_over_batch(tensor);
    const auto& result_shape = result.logical_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], 1U);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);
}

TEST_F(TrivialTnnFixedTest, TestDivide) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 2U;
    const size_t features = 64U;
    std::vector<float> lhs(batch_size * features);
    std::vector<float> rhs(batch_size * features);

    for (int i = 0; i < lhs.size(); ++i) {
        lhs[i] = static_cast<float>(i);
        rhs[i] = static_cast<float>(i + 1);
    }

    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto lhs_tensor = ttml::core::from_vector(lhs, shape, device);
    auto rhs_tensor = ttml::core::from_vector(rhs, shape, device);

    auto result = ttml::ttnn_fixed::divide(lhs_tensor, rhs_tensor);
    const auto& result_shape = result.logical_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], batch_size);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);

    std::vector<float> resulting_vector = ttml::core::to_vector(result);
    EXPECT_EQ(resulting_vector.size(), batch_size * features);
    for (int i = 0; i < resulting_vector.size(); ++i) {
        EXPECT_NEAR(resulting_vector[i], static_cast<float>(i) / static_cast<float>(i + 1), 1e-2);
    }
}

TEST_F(TrivialTnnFixedTest, TestSumOverBatch_1) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 2U;
    const size_t features = 64U;
    std::vector<float> data(batch_size * features);
    float step = 0.1F;
    float value = 0.0F;
    for (int i = 0; i < data.size(); ++i) {
        data[i] = value;
        value += step;
    }

    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_shape = tensor.logical_shape();
    EXPECT_EQ(tensor_shape[0], batch_size);
    EXPECT_EQ(tensor_shape[1], 1U);
    EXPECT_EQ(tensor_shape[2], 1U);
    EXPECT_EQ(tensor_shape[3], features);

    auto result = ttml::ttnn_fixed::sum_over_batch(tensor);
    const auto& result_shape = result.logical_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], 1U);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);

    std::vector<float> resulting_vector = ttml::core::to_vector(result);
    EXPECT_EQ(resulting_vector.size(), features);
    const float eps = 1.0F;
    for (int i = 0; i < resulting_vector.size(); ++i) {
        float expected_value = 0.F;
        for (int j = 0; j < batch_size; ++j) {
            expected_value += static_cast<float>(i + j * features) * step;
        }

        EXPECT_NEAR(expected_value, resulting_vector[i], eps);
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingZeroTemperatureNoMask) {
    // xarray of shape {1, 1, 32, 32} with max along the diagonal
    xt::xarray<float>::shape_type shape = {1, 1, 32, 32};
    xt::xarray<float> a = xt::zeros<float>(shape);
    // Set diagonal max: for each row i, set a(0,0,i,i) = 1000.0f
    for (size_t i = 0; i < 32; ++i) {
        a(0, 0, i, i) = 1000.0f;
    }
    std::vector<uint32_t> expected_b(32);
    for (size_t i = 0; i < 32; ++i) {
        expected_b[i] = i;
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto tensor_b = ttml::ttnn_fixed::sample(tensor_a, 0.0F, 42);
    auto vector_b = ttml::core::to_vector<uint32_t>(tensor_b);
    EXPECT_EQ(vector_b, expected_b);
}

TEST_F(TrivialTnnFixedTest, TestSamplingPositiveTemperatureNoMask) {
    // Test sampling with positive temperature, no mask, and xarray of shape {1, 1, 32, 64}
    xt::xarray<float>::shape_type shape = {1, 1, 32, 64};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, 42U);
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    float temperature = 1.0F;
    auto tensor_b = ttml::ttnn_fixed::sample(tensor_a, temperature, 42);
    auto vector_b = ttml::core::to_vector<uint32_t>(tensor_b);
    // The output should have shape {1, 1, 32} (one sample per row)
    EXPECT_EQ(vector_b.size(), 32);
    // All values should be in the range [0, 63] (since last dim is 64)
    for (auto v : vector_b) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 64);
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingPositiveTemperatureWithMask) {
    // TODO: Accuracy issue with BH. Tracking issue: https://github.com/tenstorrent/tt-metal/issues/37342
    auto board = tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0);
    if (board == tt::BoardType::P100 || board == tt::BoardType::P150) {
        GTEST_SKIP() << "Skipping on P100/P150 boards";
    }
    // Test sampling with positive temperature, with mask, and xarray of shape {1, 1, 32, 65}
    xt::xarray<float>::shape_type shape = {1, 1, 32, 65};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, 84U);
    // Create a mask: mask out the last column (set to large negative value)
    xt::xarray<float> mask = xt::zeros<float>(shape);
    for (size_t i = 0; i < 32; ++i) {
        mask(0, 0, i, 64) = 1e4F;
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto tensor_mask = ttml::core::from_xtensor(mask, &ttml::autograd::ctx().get_device());
    float temperature = 1.0F;
    auto tensor_b = ttml::ttnn_fixed::sample(tensor_a, temperature, 42, tensor_mask);
    auto vector_b = ttml::core::to_vector<uint32_t>(tensor_b);
    // The output should have shape {1, 1, 32} (one sample per row)
    EXPECT_EQ(vector_b.size(), 32);
    // All values should be in the range [0, 63] (since last dim is 65, but last index is masked)
    for (auto v : vector_b) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 64);
    }
}

namespace {

// sample() draws U ~ Uniform[2^-32, 1) and applies the Gumbel transform -log(-log(U)), so the noise
// added to the scaled logits is bounded to roughly [-3.1, +16.6]. The tests below size their logit
// gaps against this span so that "the scaled logits must win" is a guarantee, not a coin flip.
constexpr float kGumbelNoiseSpan = 20.0F;

}  // namespace

TEST_F(TrivialTnnFixedTest, TestSamplingDoesNotMutateInputs) {
    // sample() runs the Gumbel chain, the temperature scaling and the mask subtraction in place.
    // Only the noise buffer it allocates itself may be written -- the caller's logits and mask must
    // come back untouched. The size/bounds assertions in the tests above cannot see an in-place op
    // that picked the wrong operand, so check the inputs directly.
    constexpr uint32_t kRows = 32;
    constexpr uint32_t kVocab = 64;
    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, 42U);
    xt::xarray<float> m = xt::zeros<float>(shape);
    for (uint32_t i = 0; i < kRows; ++i) {
        m(0, 0, i, kVocab - 1) = 1e4F;
    }

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_mask = ttml::core::from_xtensor(m, device);

    const auto logits_before = ttml::core::to_vector(tensor_a);
    const auto mask_before = ttml::core::to_vector(tensor_mask);

    // Positive temperature, no mask: the fused add writes into the noise buffer, reading the logits
    // through an activation. Nothing should land back on the logits.
    (void)ttml::ttnn_fixed::sample(tensor_a, 1.0F, 42);
    EXPECT_EQ(ttml::core::to_vector(tensor_a), logits_before) << "positive temperature, no mask";

    // Positive temperature with a mask: the subtract is in place, but on a buffer sample() owns.
    (void)ttml::ttnn_fixed::sample(tensor_a, 1.0F, 42, tensor_mask);
    EXPECT_EQ(ttml::core::to_vector(tensor_a), logits_before) << "positive temperature, with mask";
    EXPECT_EQ(ttml::core::to_vector(tensor_mask), mask_before) << "positive temperature, mask operand";

    // Zero temperature with a mask: nothing has been allocated yet, so the working tensor still
    // aliases the caller's logits and the subtract must NOT be in place. This is the path the
    // out_is_owned flag exists to protect, and it is otherwise untested.
    (void)ttml::ttnn_fixed::sample(tensor_a, 0.0F, 42, tensor_mask);
    EXPECT_EQ(ttml::core::to_vector(tensor_a), logits_before) << "zero temperature, with mask";
    EXPECT_EQ(ttml::core::to_vector(tensor_mask), mask_before) << "zero temperature, mask operand";
}

TEST_F(TrivialTnnFixedTest, TestSamplingTemperatureScalesLogitsNotNoise) {
    // sample() computes logits/temperature + noise, with the 1/temperature factor carried as an
    // activation on the *logits* operand of a single fused add. If that factor were attached to the
    // noise operand instead, or inverted, both directions below flip.
    constexpr uint32_t kRows = 32;
    constexpr uint32_t kVocab = 64;
    constexpr float kWinnerLogit = 1.0F;
    constexpr float kBackgroundMax = 0.25F;

    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, kBackgroundMax, 7U);
    std::vector<uint32_t> expected(kRows);
    for (uint32_t i = 0; i < kRows; ++i) {
        // Deliberately not the row index and not a constant, so a degenerate result is still wrong.
        const uint32_t winner = (i * 7U + 3U) % kVocab;
        a(0, 0, i, winner) = kWinnerLogit;
        expected[i] = winner;
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Cold: the gap is 0.75, so at temperature 0.005 the scaled gap is 150 -- far beyond the noise
    // span. Sampling must collapse onto the argmax, deterministically.
    const float cold_temperature = (kWinnerLogit - kBackgroundMax) / (10.0F * kGumbelNoiseSpan);
    auto cold = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, cold_temperature, 42));
    EXPECT_EQ(cold, expected) << "low temperature must reduce to argmax";

    // Hot: the logits are scaled down to <= 1e-4 and the noise dominates, so the argmax must not win
    // every row. All 32 rows agreeing by chance would be (1/64)^32.
    auto hot = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0e4F, 42));
    ASSERT_EQ(hot.size(), expected.size());
    EXPECT_NE(hot, expected) << "high temperature must not reduce to argmax";
}

TEST_F(TrivialTnnFixedTest, TestSamplingGumbelMatchesSoftmaxDistribution) {
    // The Gumbel-max trick guarantees P(argmax == i) == softmax(logits / temperature)_i. This is the
    // only assertion in the suite that actually pins down the -log(-log(U)) chain: dropping or
    // reordering a step still yields in-range indices, so the shape and bounds checks stay green.
    constexpr uint32_t kRows = 2048;
    constexpr uint32_t kVocab = 32;
    constexpr uint32_t kActive = 4;
    // Unnormalized weights on the first four columns; every other column is pushed far enough down
    // that the bounded noise can never lift it (-60 + 16.6 is still well below 0 - 3.1).
    constexpr std::array<float, kActive> kWeights = {8.0F, 4.0F, 2.0F, 1.0F};
    constexpr float kWeightTotal = 15.0F;
    constexpr float kSuppressed = -60.0F;

    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(kSuppressed);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t c = 0; c < kActive; ++c) {
            a(0, 0, r, c) = std::log(kWeights[c]);
        }
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Pool several seeds so the result does not hinge on the internal structure of one RNG stream.
    const std::vector<uint32_t> seeds = {1U, 2U, 3U, 4U};
    std::array<uint32_t, kActive> counts{};
    uint32_t total_samples = 0U;
    for (auto seed : seeds) {
        auto picks = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, seed));
        ASSERT_EQ(picks.size(), kRows);
        for (auto pick : picks) {
            ASSERT_LT(pick, kActive) << "sampled a column whose logit was " << kSuppressed;
            ++counts[pick];
            ++total_samples;
        }
    }

    // Five sigma on Binomial(total_samples, p): flakes at ~1e-6 per column, while a broken Gumbel
    // chain moves these counts by tens of sigma.
    for (uint32_t c = 0; c < kActive; ++c) {
        const double p = static_cast<double>(kWeights[c]) / kWeightTotal;
        const double expected_count = p * total_samples;
        const double tolerance = 5.0 * std::sqrt(total_samples * p * (1.0 - p));
        EXPECT_NEAR(static_cast<double>(counts[c]), expected_count, tolerance)
            << "column " << c << " selected " << counts[c] << " / " << total_samples;
    }

    // A nonzero seed is contractually reproducible, and distinct seeds must actually decorrelate --
    // both are properties the fused in-place chain could silently break.
    auto first = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 1234U));
    auto again = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 1234U));
    EXPECT_EQ(first, again) << "same seed must reproduce the same samples";
    auto other = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 5678U));
    EXPECT_NE(first, other) << "different seeds must produce different samples";
}
