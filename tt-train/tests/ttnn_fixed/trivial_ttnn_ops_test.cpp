// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_fixed/trivial_ttnn_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

TEST_F(TrivialTnnFixedTest, TestSamplingSubnormalTemperatureIsExactGreedy) {
    // 1e-39F is positive and finite, so it passes validation and used to select the noise kernel --
    // but 1/1e-39 overflows FLT_MAX, and the resulting +inf scale factor collapsed every positive
    // logit to the same +inf bit pattern (and every zero logit to NaN, which float32_greater never
    // picks). The "sampled" argmax then returned the FIRST positive column instead of the max one.
    // Sub-reciprocal-overflow temperatures now route to the greedy kernel, the exact limit a
    // temperature anneal approaches.
    //
    // Each row is built so the two behaviors disagree: an early positive DECOY column with a small
    // logit and a later WINNER column with the true max. The old path returns the decoy (first
    // positive), greedy returns the winner -- so exact equality here is the regression check.
    constexpr uint32_t kRows = 32U;
    constexpr uint32_t kVocab = 64U;
    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);
    std::vector<uint32_t> expected(kRows);
    for (uint32_t i = 0; i < kRows; ++i) {
        const uint32_t decoy = i % 8U;
        const uint32_t winner = 8U + ((i * 3U) % (kVocab - 8U));
        a(0, 0, i, decoy) = 0.5F;
        a(0, 0, i, winner) = 2.0F;
        expected[i] = winner;
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto got = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1e-39F, 42));
    EXPECT_EQ(got, expected);
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
    // Test sampling with positive temperature, with mask, and xarray of shape {1, 1, 32, 65}
    xt::xarray<float>::shape_type shape = {1, 1, 32, 65};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, 84U);
    // Mask out the last column. Shape is {1, 1, 1, 65}: one row broadcast across every token, which
    // is what callers build (padding columns do not depend on token position).
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, 65};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    mask(0, 0, 0, 64) = 1e4F;
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
    // [1, 1, 1, V]: the broadcast shape every caller passes.
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> m = xt::zeros<float>(mask_shape);
    m(0, 0, 0, kVocab - 1) = 1e4F;

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
    // aliases the caller's logits and the subtract must NOT be in place.
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

TEST_F(TrivialTnnFixedTest, TestSamplingBroadcastPaddingMask) {
    // The padding mask every real caller builds is [1, 1, 1, V] -- ONE row, because which vocab
    // columns are padding does not depend on the token position (see _sample_logits_mask in
    // generate.py and _build_logits_mask in llama_completer.py). It must apply to every token row.
    //
    // The mask is independent of the BATCH for the same reason one level up: every sequence is
    // decoded by the same lm_head, so the same columns are padding for all of them. kBatch > 1 makes
    // that explicit -- one [1, 1, 1, V] mask has to cover both entries, and the reader addresses mask
    // tiles by column alone, with no batch stride to get wrong.
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kRows = 32;          // must exceed 1, or the unmasked rows do not exist
    constexpr uint32_t kVocab = 64;         // real vocabulary
    constexpr uint32_t kPaddedVocab = 128;  // what a TP-padded LM head actually emits
    constexpr uint32_t kBestRealId = 42;

    // Real columns are negative (kBestRealId least so); padding columns sit at 0.0, exactly as
    // zero-filled LM-head rows do. So an UNMASKED argmax lands on kVocab -- the first padding
    // column -- and a correctly masked one lands on kBestRealId.
    xt::xarray<float>::shape_type logits_shape = {kBatch, 1, kRows, kPaddedVocab};
    xt::xarray<float> logits = xt::zeros<float>(logits_shape);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t c = 0; c < kVocab; ++c) {
                logits(b, 0, r, c) = -1.0F;
            }
            logits(b, 0, r, kBestRealId) = -0.5F;
            // columns [kVocab, kPaddedVocab) stay at 0.0
        }
    }

    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kPaddedVocab};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    for (uint32_t c = kVocab; c < kPaddedVocab; ++c) {
        mask(0, 0, 0, c) = 1e4F;
    }

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_logits = ttml::core::from_xtensor(logits, device);
    auto tensor_mask = ttml::core::from_xtensor(mask, device);

    // Greedy: exact and deterministic, so assert the strongest thing -- every row picks the best
    // REAL column. A row that missed the mask would report kVocab instead.
    auto greedy = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_logits, 0.0F, 42, tensor_mask));
    ASSERT_EQ(greedy.size(), kBatch * kRows);
    EXPECT_EQ(greedy, std::vector<uint32_t>(kBatch * kRows, kBestRealId))
        << "a [1, 1, 1, V] mask must apply to every token row of every batch entry, not just row 0";

    // Positive temperature compiles a different kernel (the noise and the scaling are no longer
    // compiled out), so the broadcast has to hold there too. The noise makes the winner among the
    // real columns unpredictable, but the 1e4 penalty is far beyond the noise span, so a padding
    // column must never win.
    auto sampled = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_logits, 1.0F, 4242, tensor_mask));
    ASSERT_EQ(sampled.size(), kBatch * kRows);
    for (uint32_t r = 0; r < kBatch * kRows; ++r) {
        EXPECT_LT(sampled[r], kVocab) << "row " << r << " sampled a masked padding column";
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingRaggedShapes) {
    // Every other sampling test uses tile-aligned dimensions (32 or 2048 tokens, 32/64 vocab) and a
    // single batch entry, which leaves three pieces of the op untested:
    //
    //   * tokens % 32 != 0  -- the last tile row of each batch entry is partly padding, so the
    //                          writer must emit only `valid_rows` results for it.
    //   * V % 32 != 0       -- the last vocab tile is partly padding, so the argmax scan must stop
    //                          at the logical width (`cols_to_scan`).
    //   * batch > 1         -- output pages are indexed as batch_index * tokens + first_token, and
    //                          with one batch entry that term is always zero.
    //
    // Padding is what makes this a real test rather than a shape smoke test: from_xtensor zero-fills
    // the padded region, and every REAL logit here is negative, so any padding element the scan
    // wrongly visits (0.0) beats the whole row and shows up as an out-of-range index.
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTokens = 37U;  // 37 = 32 + 5 -> Ht = 2, last tile row has 5 valid rows
    constexpr uint32_t kVocab = 77U;   // 77 = 2*32 + 13 -> Wt = 3, last tile has 13 valid columns

    xt::xarray<float>::shape_type shape = {kBatch, 1, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    // A distinct winner per (batch, token) so a row/page mix-up cannot pass by coincidence.
    std::vector<uint32_t> expected(kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            const uint32_t winner = ((b * kTokens + t) * 7U) % kVocab;
            a(b, 0, t, winner) = -0.5F;
            expected[b * kTokens + t] = winner;
        }
    }

    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Greedy is exact, so assert the full result vector: one id per token, in [batch, token] order.
    auto greedy = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(greedy.size(), kBatch * kTokens) << "one sampled id per token, across all batch entries";
    EXPECT_EQ(greedy, expected);

    // The scan bounds also have to hold in the sampled kernel, which is a separate binary
    // (the noise compile-time arg). The noise makes the winner unpredictable, but no index may ever leave the
    // logical vocabulary -- reaching the zero-filled padding would.
    auto sampled = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 99));
    ASSERT_EQ(sampled.size(), kBatch * kTokens);
    for (uint32_t i = 0; i < sampled.size(); ++i) {
        EXPECT_LT(sampled[i], kVocab) << "index " << i << " left the logical vocabulary";
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingHonoursBufferPlacement) {
    // TensorAccessorArgs bakes buffer placement into the reader/writer COMPILE-TIME args: it sets
    // ArgConfig::IsDram from buffer->is_dram() and emits the buffer's aligned_page_size, both of
    // which differ between DRAM and L1 (see tensor_accessor_args.cpp). Two calls that differ ONLY in
    // placement therefore need two different programs -- but they have identical shapes, dtypes and
    // mask-ness, so they collide in the program cache unless placement is part of its key. On the
    // second call the cache hit patches addresses only, leaving accessors compiled for the wrong
    // memory space pointed at the other one's addresses.
    //
    // Order matters: the DRAM call must run FIRST so it is the entry the L1 call then collides with.
    constexpr uint32_t kRows = 32;
    constexpr uint32_t kVocab = 64;
    constexpr uint32_t kDecoy = kVocab - 1;  // always the raw argmax; only the mask can dethrone it

    auto* device = &ttml::autograd::ctx().get_device();

    xt::xarray<float>::shape_type shape = {1, 1, kRows, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);
    std::vector<uint32_t> expected(kRows);
    for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t winner = (r * 7U + 3U) % (kVocab - 1U);  // never the decoy
        a(0, 0, r, winner) = 1.0F;
        expected[r] = winner;
    }

    // ---- logits placement ----
    auto dram_logits = ttml::core::from_xtensor(a, device);
    ASSERT_EQ(dram_logits.memory_config().buffer_type(), tt::tt_metal::BufferType::DRAM);
    auto from_dram = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(dram_logits, 0.0F, 42));
    EXPECT_EQ(from_dram, expected) << "DRAM logits";

    auto l1_logits = ttml::ttnn_fixed::to_l1_interleaved(dram_logits);
    ASSERT_EQ(l1_logits.memory_config().buffer_type(), tt::tt_metal::BufferType::L1);
    auto from_l1 = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(l1_logits, 0.0F, 42));
    EXPECT_EQ(from_l1, expected) << "L1 logits must sample identically to DRAM logits";

    // ---- mask placement ----
    // The mask's placement is hashed (placement_of) precisely because accessors are compiled per memory space — this
    // guards that. The decoy column outranks every real winner, so a mask read from the wrong memory space cannot go
    // unnoticed.
    xt::xarray<float> decoyed = a;
    for (uint32_t r = 0; r < kRows; ++r) {
        decoyed(0, 0, r, kDecoy) = 2.0F;
    }
    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> m = xt::zeros<float>(mask_shape);
    m(0, 0, 0, kDecoy) = 1e4F;

    auto decoyed_logits = ttml::core::from_xtensor(decoyed, device);
    auto dram_mask = ttml::core::from_xtensor(m, device);
    ASSERT_EQ(dram_mask.memory_config().buffer_type(), tt::tt_metal::BufferType::DRAM);
    auto masked_dram = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(decoyed_logits, 0.0F, 42, dram_mask));
    EXPECT_EQ(masked_dram, expected) << "DRAM mask";

    auto l1_mask = ttml::ttnn_fixed::to_l1_interleaved(dram_mask);
    ASSERT_EQ(l1_mask.memory_config().buffer_type(), tt::tt_metal::BufferType::L1);
    auto masked_l1 = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(decoyed_logits, 0.0F, 42, l1_mask));
    EXPECT_EQ(masked_l1, expected) << "L1 mask must suppress the decoy exactly as a DRAM mask does";
}

namespace {

// Positions as the op wants them: [B, 1, 1, 1] UINT32 ROW_MAJOR. Note the explicit layout --
// core::from_vector defaults to TILE, which the op rejects (a tiled [B,1,1,1] pads to a 32x32 tile
// and would make every page read 4 KB instead of one aligned word).
ttnn::Tensor make_positions(const std::vector<uint32_t>& positions) {
    return ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        positions,
        ttnn::Shape({static_cast<uint32_t>(positions.size()), 1U, 1U, 1U}),
        &ttml::autograd::ctx().get_device(),
        ttnn::Layout::ROW_MAJOR);
}

}  // namespace

TEST_F(TrivialTnnFixedTest, TestSamplingAtPerRowPositions) {
    // Prefill wants ONE token per sequence, taken at that sequence's own prompt end -- a different
    // row for every batch entry. Passing those positions makes the op read only the tiles holding
    // them and return [B, 1, 1, 1]. What has to hold is that the shortcut is exactly equivalent:
    // sampling at position p must give what sampling everything would have given at row p.
    //
    // The shapes are deliberately ragged (tokens and vocab both mid-tile) and the positions are
    // spread across tile rows -- the first tile row, a middle one, and the partly-padded last one --
    // because the position picks BOTH the source page and the row inside that tile, and a wrong
    // row/tile split would still land on a real row for tile-aligned positions.
    constexpr uint32_t kBatch = 3U;
    constexpr uint32_t kTokens = 70U;  // Ht = 3; the last tile row has 6 valid rows
    constexpr uint32_t kVocab = 77U;   // Wt = 3; the last tile has 13 valid columns

    const std::vector<uint32_t> positions = {0U, 45U, 69U};  // tile rows 0, 1, 2; rows 0, 13, 5

    xt::xarray<float>::shape_type shape = {kBatch, 1, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    // Distinct winner per (batch, token), so reading the wrong row or the wrong batch entry lands on
    // a different id rather than coincidentally matching.
    std::vector<uint32_t> expected_all(kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            const uint32_t winner = ((b * kTokens + t) * 11U) % kVocab;
            a(b, 0, t, winner) = -0.5F;
            expected_all[b * kTokens + t] = winner;
        }
    }

    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Sample everything first. Besides producing the reference, this seeds the program cache with
    // the no-positions program: the positioned call that follows has the same shapes, dtype and
    // mask-ness, so it collides with it unless the cache key knows about positions -- and a
    // collision would reuse a program whose output is [B, 1, tokens, 1].
    auto greedy_all = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(greedy_all.size(), kBatch * kTokens);
    EXPECT_EQ(greedy_all, expected_all);

    std::vector<uint32_t> expected_at_positions(kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        expected_at_positions[b] = expected_all[b * kTokens + positions[b]];
    }

    auto greedy_at = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
        tensor_a, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, make_positions(positions)));
    ASSERT_EQ(greedy_at.size(), kBatch) << "one sampled id per batch entry, not per token";
    EXPECT_EQ(greedy_at, expected_at_positions);

    // The scan bounds have to hold in the sampled kernel too, which is a separate binary
    // (the noise compile-time arg). The noise makes the winner unpredictable, but every real logit here is
    // negative while from_xtensor zero-fills the padding, so any index that leaves the logical
    // vocabulary means the scan walked into padding.
    auto sampled_at = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
        tensor_a, 1.0F, 99, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, make_positions(positions)));
    ASSERT_EQ(sampled_at.size(), kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_LT(sampled_at[b], kVocab) << "batch entry " << b << " left the logical vocabulary";
    }

    // Every entry pointed at the SAME row exercises the other extreme of the work split: all three
    // entries now read the same tile row of their own shard, and the boundary-merge path sees three
    // groups that each span whatever cores the split handed them.
    const std::vector<uint32_t> uniform(kBatch, kTokens - 1U);
    auto greedy_uniform = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
        tensor_a, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, make_positions(uniform)));
    ASSERT_EQ(greedy_uniform.size(), kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_EQ(greedy_uniform[b], expected_all[b * kTokens + (kTokens - 1U)]);
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingAtPerRowPositionsAcrossTokenCounts) {
    // With positions supplied, the program is independent of the token dimension -- the work split
    // is one tile row per batch entry however many tokens the logits carry -- so the cache key
    // normalizes that dimension away and ONE program serves every prompt length. That is what makes
    // prefill affordable: a GRPO rollout rounds its prompts to a new length most generates, and each
    // distinct length used to cost a fresh JIT build of all three kernels (~6 s, against ~3 ms for
    // the dispatch itself).
    //
    // The price is that the second call below reuses the first call's program with only its RUNTIME
    // args patched -- and Ht is now one of those. If it is not re-applied, the reader resolves a
    // batch entry's tile row as entry * Ht_stale + position / 32, which for a later entry still
    // lands inside that entry's own data and inside the buffer: a real token row, no fault, just
    // the wrong one. So this test needs all three of: two different token counts, a position in the
    // last tile row, and assertions on a LATER batch entry. Entry 0 cannot see the bug at all,
    // because entry * Ht is zero whatever Ht is.
    constexpr uint32_t kBatch = 3U;
    constexpr uint32_t kVocab = 77U;

    auto run = [](uint32_t tokens, const std::vector<uint32_t>& positions) {
        xt::xarray<float>::shape_type shape = {kBatch, 1U, tokens, kVocab};
        xt::xarray<float> a = xt::zeros<float>(shape);
        a.fill(-1.0F);

        std::vector<uint32_t> expected_all(kBatch * tokens);
        for (uint32_t b = 0; b < kBatch; ++b) {
            for (uint32_t t = 0; t < tokens; ++t) {
                const uint32_t winner = ((b * tokens + t) * 11U) % kVocab;
                a(b, 0, t, winner) = -0.5F;
                expected_all[b * tokens + t] = winner;
            }
        }

        auto tensor = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
        auto got = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
            tensor, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, make_positions(positions)));

        std::vector<uint32_t> expected(kBatch);
        for (uint32_t b = 0; b < kBatch; ++b) {
            expected[b] = expected_all[b * tokens + positions[b]];
        }
        return std::pair<std::vector<uint32_t>, std::vector<uint32_t>>{std::move(got), std::move(expected)};
    };

    // Builds the program at Ht = 3 (70 tokens pad to 96).
    const auto first = run(70U, {0U, 37U, 69U});
    ASSERT_EQ(first.first.size(), kBatch);
    EXPECT_EQ(first.first, first.second) << "70-token call";

    // Same batch and vocabulary, so under the normalized key this reuses the program above -- but it
    // needs Ht = 5 (134 tokens pad to 160). Entry 2 at token 133 belongs to tile row 14; replayed
    // with the stale Ht it would resolve to row 10, which is still entry 2 and still in bounds, but
    // holds token 0.
    const auto second = run(134U, {0U, 70U, 133U});
    ASSERT_EQ(second.first.size(), kBatch);
    EXPECT_EQ(second.first, second.second) << "134-token call reusing the 70-token program";
}

TEST_F(TrivialTnnFixedTest, TestSamplingAtPerRowPositionsLargeBatch) {
    // Positions live in a small tensor each core stages into L1, so the batch is bounded only by memory.
    // This test also exercises the positions CB at a size where an off-by-one in its bound would trip watcher.
    constexpr uint32_t kBatch = 512U;
    constexpr uint32_t kTokens = 64U;
    constexpr uint32_t kVocab = 33U;

    xt::xarray<float>::shape_type shape = {kBatch, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    std::vector<uint32_t> positions(kBatch);
    std::vector<uint32_t> expected(kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        positions[b] = (b * 13U) % kTokens;
        const uint32_t winner = (b * 7U) % kVocab;
        a(b, 0, positions[b], winner) = -0.5F;
        expected[b] = winner;
    }

    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto got = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
        tensor_a, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, make_positions(positions)));
    ASSERT_EQ(got.size(), kBatch);
    EXPECT_EQ(got, expected);
}

TEST_F(TrivialTnnFixedTest, TestSamplingRepatchesPositionsBufferOnCacheHit) {
    // The positions BUFFER ADDRESS is a runtime arg, and every prefill builds a new tensor. A cached
    // program replayed against a stale address reads whatever DRAM now occupies that region: in
    // bounds, no fault, a plausible-looking token. Nothing else in this file varies the positions
    // buffer across two calls that share a program, so a dropped re-patch passes every other test.
    //
    // The first tensor is deallocated before the second is built, so the allocator is likely to hand
    // back the same region -- which is exactly the case where a stale address looks healthy.
    constexpr uint32_t kBatch = 4U;
    constexpr uint32_t kTokens = 96U;
    constexpr uint32_t kVocab = 40U;

    xt::xarray<float>::shape_type shape = {kBatch, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);
    std::vector<uint32_t> winner_at(kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            const uint32_t winner = ((b * kTokens + t) * 3U) % kVocab;
            a(b, 0, t, winner) = -0.5F;
            winner_at[b * kTokens + t] = winner;
        }
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    auto sample_at = [&](const std::vector<uint32_t>& positions) {
        auto positions_tt = make_positions(positions);
        auto got = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
            tensor_a, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, positions_tt));
        positions_tt.deallocate(/* force */ true);
        return got;
    };

    const std::vector<uint32_t> first_positions = {0U, 31U, 64U, 95U};
    const std::vector<uint32_t> second_positions = {95U, 64U, 31U, 0U};  // same shapes, different values

    auto first = sample_at(first_positions);
    auto second = sample_at(second_positions);

    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_EQ(first[b], winner_at[b * kTokens + first_positions[b]]) << "first call, entry " << b;
        EXPECT_EQ(second[b], winner_at[b * kTokens + second_positions[b]])
            << "second call reused the first call's program, entry " << b;
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingClampsOutOfRangePosition) {
    // Positions live in device memory, so the host cannot range-check them on the dispatch path.
    // The kernels clamp each position to the LAST REAL TOKEN -- reader and writer apply the same
    // clamp to the value before splitting it into their disjoint bit fields (>> 5 and & 31), so
    // the expectations here are EXACT row identities, not just "stayed inside the vocabulary".
    // Two failure modes hide behind that weaker check:
    //
    //  * A position in the tile-padding band [tokens, Ht*32) -- e.g. the classic off-by-one of
    //    position == prompt length on a mid-tile prompt -- passes any tile-row-only bound and
    //    lands on a ZERO-FILLED padding row. Greedy argmax over zeros returns token 0: in range,
    //    silently wrong. kTokens is deliberately mid-tile so the band exists at all; a
    //    tile-aligned token count has none (which is why this test previously could not catch it).
    //  * A position past Ht*32 resolves to a page outside the logits buffer entirely: interleaved
    //    accessors bounds-check nothing, and watcher validates the whole DRAM window rather than
    //    the buffer.
    //
    // Every winner below is nonzero, so a scan of a zeroed padding row (which returns 0) can never
    // masquerade as a pass.
    if (std::getenv("TT_METAL_WATCHER") != nullptr) {
        GTEST_SKIP() << "out-of-range positions deliberately trip the kernels' watcher ASSERT; "
                        "under watcher the loud path replaces the clamp being tested here";
    }

    constexpr uint32_t kBatch = 4U;
    constexpr uint32_t kTokens = 70U;  // Ht = 3, mid-tile: the padding band is [70, 96)
    constexpr uint32_t kVocab = 40U;

    xt::xarray<float>::shape_type shape = {kBatch, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);
    std::vector<uint32_t> winner_at(kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            const uint32_t winner = ((b * kTokens + t) * 5U) % kVocab;
            a(b, 0, t, winner) = -0.5F;
            winner_at[b * kTokens + t] = winner;
        }
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Entry 0: in-range control (the last real token itself -- the clamp must not disturb it).
    // Entry 1: the off-by-one, first value of the padding band; SAME tile row as real data, so a
    //          tile-row-only clamp passes it straight through to the padding row.
    // Entry 2: deep in the padding band, still inside the last real tile.
    // Entry 3: far past the padded extent, in a tile row that does not exist.
    const std::vector<uint32_t> positions = {kTokens - 1U, kTokens, 90U, 10U * kTokens};
    auto got = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
        tensor_a, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, make_positions(positions)));
    ASSERT_EQ(got.size(), kBatch);

    EXPECT_EQ(got[0], winner_at[0 * kTokens + (kTokens - 1U)]) << "in-range entry disturbed by the clamp";
    for (uint32_t b = 1; b < kBatch; ++b) {
        EXPECT_EQ(got[b], winner_at[b * kTokens + (kTokens - 1U)])
            << "entry " << b << " (position " << positions[b] << ") must clamp to the last real token's row";
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingRejectsOutOfRangeSeedAxis) {
    // seeded_linear_index() skips mesh axes it cannot find, so before this was validated an
    // out-of-range seed axis (a typo, or a config reused across mesh topologies) silently degraded
    // to "no axis seeded": every data-parallel device drew byte-identical noise and a GRPO rollout
    // emitted duplicate completions with zero-variance advantages. The op must reject it loudly
    // instead. Axis 7 is out of range on any mesh this suite runs on.
    xt::xarray<float>::shape_type shape = {1, 1, 32, 64};
    xt::xarray<float> a = ttml::test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, 42U);
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    EXPECT_ANY_THROW(ttml::ttnn_fixed::sample(
        tensor_a, 1.0F, 42, /* mask */ std::nullopt, /* seed_axes */ std::vector<uint32_t>{7U}));
}

TEST_F(TrivialTnnFixedTest, TestSamplingWithoutPositionsUnchangedByAccessorChain) {
    // The non-position path gained a NULL TensorAccessorArgs append so the accessor chain's length is
    // the same in both modes. If that append is dropped, or the hard-coded offsets drift, the next
    // accessor misdecodes its page size as the config flags -- silently, not as a build error. Cover
    // the reader chain at its new length both with and without a mask.
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTokens = 37U;
    constexpr uint32_t kVocab = 77U;

    xt::xarray<float>::shape_type shape = {kBatch, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);
    std::vector<uint32_t> expected(kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            const uint32_t winner = ((b * kTokens + t) * 7U) % kVocab;
            a(b, 0, t, winner) = -0.5F;
            expected[b * kTokens + t] = winner;
        }
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    auto no_mask = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(no_mask.size(), kBatch * kTokens) << "no-positions output must stay [B, 1, tokens, 1]";
    EXPECT_EQ(no_mask, expected);

    // A mask sits between the logits and positions accessors in the chain, so it is the case where a
    // length mismatch shows up.
    xt::xarray<float> m = xt::zeros<float>(xt::xarray<float>::shape_type{1U, 1U, 1U, kVocab});
    auto mask = ttml::core::from_xtensor(m, &ttml::autograd::ctx().get_device());
    auto with_mask = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7, mask));
    ASSERT_EQ(with_mask.size(), kBatch * kTokens);
    EXPECT_EQ(with_mask, expected) << "an all-zero mask must not change the result";
}

TEST_F(TrivialTnnFixedTest, TestSamplingPerRowMask) {
    // A [B, 1, 1, V] mask gives each batch entry its own bias row (per-request logit bias / banned
    // ids), broadcast down token positions -- served by the same program as the shared [1, 1, 1, V]
    // mask via a runtime page stride. Each entry bans a DIFFERENT column, so reading another entry's
    // mask row (the stride bug this test exists to catch: wrong entry -> wrong page, in bounds, no
    // fault) changes which token wins.
    constexpr uint32_t kBatch = 3U;
    constexpr uint32_t kTokens = 70U;  // Ht = 3, so entry != tile_row: the entry derivation is exercised
    constexpr uint32_t kVocab = 77U;   // Wt = 3, ragged last tile

    xt::xarray<float>::shape_type shape = {kBatch, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    // Per (entry, token): best column b+1, runner-up 0. Entry e's mask bans column e+1, so with the
    // mask the winner must flip to 0 for ALL tokens of that entry -- but only that entry's rows.
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            a(b, 0, t, b + 1) = -0.25F;
            a(b, 0, t, 0) = -0.5F;
        }
    }
    xt::xarray<float> m = xt::zeros<float>(xt::xarray<float>::shape_type{kBatch, 1U, 1U, kVocab});
    for (uint32_t b = 0; b < kBatch; ++b) {
        m(b, 0, 0, b + 1) = 1e4F;
    }

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_m = ttml::core::from_xtensor(m, device);

    // Unmasked: entry b picks b+1 everywhere (sanity that the setup is what we think).
    auto greedy = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(greedy.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            ASSERT_EQ(greedy[b * kTokens + t], b + 1) << "unmasked winner, entry " << b << " token " << t;
        }
    }

    // A SHARED all-zero [1, 1, 1, V] mask first, at the same logits shape. This is not a smoke
    // call: it seeds the program cache with the mask-present program built at stride 0, so the
    // per-row call below is a CACHE HIT that only works if override_runtime_arguments re-patches
    // the stride (0 -> Wt). With a stale stride every entry reads entry 0's mask row, so entries
    // 1..B-1 keep their unbanned winners -- a deterministic failure.
    xt::xarray<float> shared = xt::zeros<float>(xt::xarray<float>::shape_type{1U, 1U, 1U, kVocab});
    auto tensor_shared = ttml::core::from_xtensor(shared, device);
    auto shared_greedy = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7, tensor_shared));
    ASSERT_EQ(shared_greedy.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        ASSERT_EQ(shared_greedy[b * kTokens], b + 1) << "zero shared mask must not change the winner";
    }

    // Per-row masked: every entry's own winner is banned, so 0 must win everywhere -- and if entry
    // e were served entry f's mask row (f != e), e's winner e+1 would survive and this fails.
    auto masked = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7, tensor_m));
    ASSERT_EQ(masked.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            EXPECT_EQ(masked[b * kTokens + t], 0U) << "per-row mask missed entry " << b << " token " << t;
        }
    }

    // Back to the shared mask on the same program: the reverse stride re-patch (Wt -> 0). A stale
    // Wt stride here sends entries past the shared mask's Wt pages, so the winners it produces are
    // garbage-dependent rather than deterministic -- the assertion still holds on a correct patch
    // and the forward (0 -> Wt) direction above is the deterministic guard on the patch line.
    auto shared_again = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7, tensor_shared));
    ASSERT_EQ(shared_again.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_EQ(shared_again[b * kTokens], b + 1) << "stride must re-patch back to 0 for a shared mask";
    }

    // The per-row mask must also hold under NOISE (a separate kernel binary): the banned column
    // carries -1e4 after the subtract, so it can never win whatever the Gumbel draw.
    auto sampled = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 99, tensor_m));
    ASSERT_EQ(sampled.size(), kBatch * kTokens);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t t = 0; t < kTokens; ++t) {
            EXPECT_NE(sampled[b * kTokens + t], b + 1) << "banned column sampled, entry " << b;
            EXPECT_LT(sampled[b * kTokens + t], kVocab);
        }
    }

    // Same mask through POSITION mode: entry derivation there is virtual_tile / Wt, a different
    // code path from the tile-row derivation above.
    const std::vector<uint32_t> positions = {0U, 37U, 69U};
    auto positioned = ttml::core::to_vector<uint32_t>(
        ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7, tensor_m, /* seed_axes */ std::nullopt, make_positions(positions)));
    ASSERT_EQ(positioned.size(), kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_EQ(positioned[b], 0U) << "per-row mask in position mode, entry " << b;
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingGumbelMatchesSoftmaxDistribution) {
    // The Gumbel-max trick guarantees P(argmax == i) == softmax(logits / temperature)_i. This is the
    // only assertion in the suite that actually pins down the -log(-log(U)) chain: dropping or
    // reordering a step still yields in-range indices, so the shape and bounds checks stay green.
    //
    // The shape is deliberately awkward on every axis, so the distribution has to survive the same
    // padding and page arithmetic the rest of the op relies on:
    //   kBatch = 2       -> results are gathered across batch entries (page = batch * tokens + t)
    //   kRows  % 32 = 1  -> the last of 33 tile rows per batch entry contributes a SINGLE sample
    //   kVocab % 32 = 24 -> the last of 4 vocab tiles is partly padding
    // kBatch * kRows * seeds still totals 8200 samples, so the tolerances below are unchanged.
    constexpr uint32_t kBatch = 2;
    constexpr uint32_t kRows = 1025;
    constexpr uint32_t kVocab = 120;
    constexpr uint32_t kActive = 4;
    // The active columns sit in FOUR DIFFERENT vocab tiles and in both half-faces of a tile (a tile
    // is 32 wide and splits into 16-column faces). With every weight packed into column 0..3 they
    // would all land in one face of one tile, and a running argmax that failed to carry its maximum
    // across tile or face boundaries would still pass.
    //   5 -> tile 0 face 0 | 52 -> tile 1 face 1 | 70 -> tile 2 face 0 | 115 -> tile 3 face 1
    constexpr std::array<uint32_t, kActive> kActiveCols = {5U, 52U, 70U, 115U};
    // Unnormalized weights on those columns; every other column is pushed far enough down that the
    // bounded noise can never lift it (-60 + 16.6 is still well below 0 - 3.1).
    constexpr std::array<float, kActive> kWeights = {8.0F, 4.0F, 2.0F, 1.0F};
    constexpr float kWeightTotal = 15.0F;
    constexpr float kSuppressed = -60.0F;

    xt::xarray<float>::shape_type shape = {kBatch, 1, kRows, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(kSuppressed);
    for (uint32_t b = 0; b < kBatch; ++b) {
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t c = 0; c < kActive; ++c) {
                a(b, 0, r, kActiveCols[c]) = std::log(kWeights[c]);
            }
        }
    }
    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Reverse map so a sampled column can be attributed to its weight.
    std::vector<int> col_to_slot(kVocab, -1);
    for (uint32_t c = 0; c < kActive; ++c) {
        col_to_slot[kActiveCols[c]] = static_cast<int>(c);
    }

    // Pool several seeds so the result does not hinge on the internal structure of one RNG stream.
    const std::vector<uint32_t> seeds = {1U, 2U, 3U, 4U};
    std::array<uint32_t, kActive> counts{};
    uint32_t total_samples = 0U;
    for (auto seed : seeds) {
        auto picks = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, seed));
        ASSERT_EQ(picks.size(), kBatch * kRows);
        for (auto pick : picks) {
            // Past kVocab is tile padding, which from_xtensor zero-fills -- reaching it would beat
            // the weight-1 column outright, so this also guards the ragged-width scan bound.
            ASSERT_LT(pick, kVocab) << "sampled index left the logical vocabulary";
            ASSERT_NE(col_to_slot[pick], -1) << "sampled column " << pick << ", whose logit was " << kSuppressed;
            ++counts[static_cast<uint32_t>(col_to_slot[pick])];
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
            << "column " << kActiveCols[c] << " (weight " << kWeights[c] << ") selected " << counts[c] << " / "
            << total_samples;
    }

    // A nonzero seed is contractually reproducible, and distinct seeds must actually decorrelate --
    // both are properties the fused in-place chain could silently break.
    auto first = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 1234U));
    auto again = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 1234U));
    EXPECT_EQ(first, again) << "same seed must reproduce the same samples";
    auto other = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 1.0F, 5678U));
    EXPECT_NE(first, other) << "different seeds must produce different samples";

    // ---- the same guarantee through the positions + mask path ----
    //
    // Position mode is a different program: its own work split (NC * Wt virtual tiles), its own RNG
    // stream layout, and the single-row writer path. Nothing above proves the distribution survives
    // it, so it is re-proven here with the BATCH as the sample axis: 2050 entries x 4 seeds is the
    // same 8200 samples, so the 5-sigma bounds carry over unchanged.
    //
    // Two tripwires ride along:
    //   * every NON-target row holds its mass on a sentinel column instead of the four weighted
    //     ones, so reading the wrong row samples the sentinel almost surely and trips col_to_slot;
    //   * a decoy column outweighs every active column but is suppressed by the padding mask, so a
    //     dropped or misapplied mask hands the decoy ~98.5% of the samples and shreds every bound.
    constexpr uint32_t kPosBatch = 2050U;
    constexpr uint32_t kPosTokens = 70U;          // Ht = 3; the last tile row keeps only 6 real rows
    constexpr uint32_t kSentinelCol = 20U;        // tile 0, face 1 -- not an active column
    constexpr uint32_t kDecoyCol = 100U;          // tile 3, face 0 -- masked below
    const float decoy_logit = std::log(1000.0F);  // outranks log(8) by far more than the noise span

    xt::xarray<float>::shape_type pos_shape = {kPosBatch, 1, kPosTokens, kVocab};
    xt::xarray<float> pos_logits = xt::zeros<float>(pos_shape);
    pos_logits.fill(kSuppressed);
    std::vector<uint32_t> entry_positions(kPosBatch);
    for (uint32_t b = 0; b < kPosBatch; ++b) {
        entry_positions[b] = (b * 13U) % kPosTokens;  // 13 is coprime with 70: every row gets hit
        for (uint32_t t = 0; t < kPosTokens; ++t) {
            if (t == entry_positions[b]) {
                for (uint32_t c = 0; c < kActive; ++c) {
                    pos_logits(b, 0, t, kActiveCols[c]) = std::log(kWeights[c]);
                }
                pos_logits(b, 0, t, kDecoyCol) = decoy_logit;
            } else {
                pos_logits(b, 0, t, kSentinelCol) = 0.0F;
            }
        }
    }

    xt::xarray<float>::shape_type mask_shape = {1, 1, 1, kVocab};
    xt::xarray<float> mask = xt::zeros<float>(mask_shape);
    mask(0, 0, 0, kDecoyCol) = 1e4F;

    auto* device = &ttml::autograd::ctx().get_device();
    auto pos_tensor = ttml::core::from_xtensor(pos_logits, device);
    auto mask_tensor = ttml::core::from_xtensor(mask, device);
    auto positions_tensor = make_positions(entry_positions);

    std::array<uint32_t, kActive> pos_counts{};
    uint32_t pos_total = 0U;
    for (auto seed : seeds) {
        auto picks = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
            pos_tensor, 1.0F, seed, mask_tensor, /* seed_axes */ std::nullopt, positions_tensor));
        ASSERT_EQ(picks.size(), kPosBatch);
        for (auto pick : picks) {
            ASSERT_LT(pick, kVocab) << "sampled index left the logical vocabulary";
            ASSERT_NE(col_to_slot[pick], -1) << "sampled column " << pick << ": a wrong row (sentinel " << kSentinelCol
                                             << "), an unmasked decoy (" << kDecoyCol << "), or a suppressed column";
            ++pos_counts[static_cast<uint32_t>(col_to_slot[pick])];
            ++pos_total;
        }
    }

    for (uint32_t c = 0; c < kActive; ++c) {
        const double p = static_cast<double>(kWeights[c]) / kWeightTotal;
        const double expected_count = p * pos_total;
        const double tolerance = 5.0 * std::sqrt(pos_total * p * (1.0 - p));
        EXPECT_NEAR(static_cast<double>(pos_counts[c]), expected_count, tolerance)
            << "positions+mask: column " << kActiveCols[c] << " (weight " << kWeights[c] << ") selected "
            << pos_counts[c] << " / " << pos_total;
    }
}

TEST_F(TrivialTnnFixedTest, TestSamplingWideRowManyOwners) {
    // The other shapes in this suite split a row over at most a few cores, so a row's owner merges
    // at most ~3 foreign records. A single wide tile row spread over the whole grid is the other
    // extreme: every core holding a shard of the row sends a record to the one owner, so the
    // owner's exact-count semaphore wait, the host-assigned slot addressing and the merge loop all
    // run at grid-scale fan-in.
    constexpr uint32_t kTokens = 2U;    // one tile row (Ht = 1); both real rows ride the same merge
    constexpr uint32_t kVocab = 4100U;  // Wt = 129; the last tile keeps 4 valid columns

    xt::xarray<float>::shape_type shape = {1U, 1U, kTokens, kVocab};
    xt::xarray<float> a = xt::zeros<float>(shape);
    a.fill(-1.0F);

    // Distinct winner per token row, in different vocab tiles -- one mid-run, one inside the
    // ragged final tile -- so a merge that drops, duplicates or mis-slots records cannot pass by
    // coincidence, and the ragged scan bound is exercised through the merge path too.
    constexpr std::array<uint32_t, kTokens> kWinners = {1234U, kVocab - 1U};
    std::vector<uint32_t> expected(kTokens);
    for (uint32_t t = 0; t < kTokens; ++t) {
        a(0, 0, t, kWinners[t]) = -0.5F;
        expected[t] = kWinners[t];
    }

    auto tensor_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());

    // Greedy is exact, so the winners must come back verbatim through the full-fan-in merge.
    auto greedy = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7));
    ASSERT_EQ(greedy.size(), kTokens);
    EXPECT_EQ(greedy, expected);

    // Position mode reruns the same wide-row merge under its own work split (NC * Wt virtual
    // tiles) and the single-row writer path. One batch entry means one position per call; select
    // each row in turn.
    for (uint32_t t = 0; t < kTokens; ++t) {
        auto at = ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(
            tensor_a,
            0.0F,
            7,
            /* mask */ std::nullopt,
            /* seed_axes */ std::nullopt,
            make_positions(std::vector<uint32_t>{t})));
        ASSERT_EQ(at.size(), 1U);
        EXPECT_EQ(at[0], expected[t]) << "position " << t;
    }
}

namespace {

// gumbel_sfpu.h's approximate log, re-derived on the host. The four constants are DUPLICATED from
// ttml::metal::sfpu::gumbel_noise_log in gumbel_sfpu.h -- keep them in sync with that header. The
// header itself is TRISC-only, so the invariants the kernel relies on are pinned here by
// reconstruction.
constexpr float kApproxLogLn2 = 0.693359375F;
constexpr float kApproxLogB = -0.240234375F;
constexpr float kApproxLogC = 1.4140625F;
constexpr float kApproxLogD = -0x1.2c801p+0F;

// The mantissa polynomial p(m) = m*(m*B + C) + D on the octave [1, 2), in double.
double approx_log_poly(double m) {
    return m * (m * static_cast<double>(kApproxLogB) + static_cast<double>(kApproxLogC)) +
           static_cast<double>(kApproxLogD);
}

// The full approximation log(v) ~= e*ln2 + p(m) for v = m * 2^e, m in [1, 2), matching the
// setexp/exexp split the SFPI pass performs.
template <typename T>
T approx_log(T v) {
    int exponent = 0;
    const T half_mantissa = std::frexp(v, &exponent);  // v = half_mantissa * 2^exponent, in [0.5, 1)
    const T m = half_mantissa * T(2);
    const T e = static_cast<T>(exponent - 1);
    const T poly = m * (m * T(kApproxLogB) + T(kApproxLogC)) + T(kApproxLogD);
    return e * T(kApproxLogLn2) + poly;
}

}  // namespace

TEST(GumbelSfpuHostTest, TestGumbelApproxLogInvariants) {
    constexpr double kTwoPowNeg20 = 0x1p-20;

    // Endpoint ties, exact: p(1) = -2^-20 and p(2) = ln2_c - 2^-20. These are what make
    // e*ln2_c + p(m) continuous across octave boundaries, and the shared -2^-20 shift is what
    // keeps -log(U) strictly positive without a zero guard.
    EXPECT_EQ(approx_log_poly(1.0), -kTwoPowNeg20);
    EXPECT_EQ(approx_log_poly(2.0), static_cast<double>(kApproxLogLn2) - kTwoPowNeg20);

    // p rises across a dense sweep of the octave, including the fp32 neighbours of both
    // endpoints, and stays within the fitted error bound of the exact log. A monotone transform
    // of U cannot reorder samples, so this is the property that preserves argmax semantics.
    constexpr int kGridPoints = 1'000'000;
    std::vector<double> grid;
    grid.reserve(kGridPoints + 4);
    grid.push_back(1.0);
    grid.push_back(static_cast<double>(std::nextafterf(1.0F, 2.0F)));
    for (int i = 1; i < kGridPoints; ++i) {
        grid.push_back(1.0 + static_cast<double>(i) / kGridPoints);
    }
    grid.push_back(static_cast<double>(std::nextafterf(2.0F, 1.0F)));
    grid.push_back(2.0);
    std::sort(grid.begin(), grid.end());

    uint32_t monotonicity_violations = 0U;
    double max_error = 0.0;
    double prev = approx_log_poly(grid.front());
    for (double m : grid) {
        const double p = approx_log_poly(m);
        if (p < prev) {
            ++monotonicity_violations;
        }
        prev = p;
        if (m < 2.0) {
            max_error = std::max(max_error, std::abs(p - std::log(m)));
        }
    }
    EXPECT_EQ(monotonicity_violations, 0U) << "p(m) must be nondecreasing on [1, 2]";
    EXPECT_LE(max_error, 5.5e-3) << "|p(m) - ln(m)| left the fitted bound";

    // Across octave boundaries: the full approximation, evaluated at fp32-adjacent points spanning
    // powers of two, must be nondecreasing. The exponent range comfortably covers everything the
    // noise chain feeds it: U in [2^-32, 1) and -log(U) in [~1e-6, ~22].
    for (int e = -40; e <= 32; e += 8) {
        const float x = std::ldexp(1.0F, e);
        const float below = std::nextafterf(x, 0.0F);
        const float above = std::nextafterf(x, HUGE_VALF);
        EXPECT_LE(approx_log<double>(static_cast<double>(below)), approx_log<double>(static_cast<double>(x)))
            << "octave boundary below 2^" << e;
        EXPECT_LE(approx_log<double>(static_cast<double>(x)), approx_log<double>(static_cast<double>(above)))
            << "octave boundary above 2^" << e;
    }

    // Noise ceiling. At the raw upper bound 1 - 2^-24 the fused chain stays finite (the whole
    // point of bounding U below 1.0); the generator's ATTAINABLE top of range sits one fp32 step
    // lower still, because the factory shrinks rand's closed-interval scale by one ULP when
    // from + scale would round past the bound (compute_rand_scale_bits), and there the ceiling is
    // near 13.75 -- the approximate-log analogue of the exact log's ~16.6.
    const float u_raw_max = std::nextafterf(1.0F, 0.0F);  // kGumbelUniformUpperBound
    const float raw_inner = approx_log<float>(u_raw_max);
    ASSERT_LT(raw_inner, 0.0F) << "log(U) must stay strictly negative below 1.0";
    EXPECT_LE(-approx_log<float>(-raw_inner), 13.9F);

    const float lower = 0x1p-32F;  // kGumbelUniformLowerBound
    float scale = u_raw_max - lower;
    uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (lower + scale > u_raw_max && scale_bits != 0U) {
        --scale_bits;
        scale = std::bit_cast<float>(scale_bits);
    }
    const float u_top = lower + scale;
    const float top_inner = approx_log<float>(u_top);
    ASSERT_LT(top_inner, 0.0F);
    EXPECT_LE(-approx_log<float>(-top_inner), 13.8F) << "noise ceiling left the documented ~13.75 cap";
}
