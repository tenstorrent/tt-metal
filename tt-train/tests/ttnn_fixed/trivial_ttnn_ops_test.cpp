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

TEST_F(TrivialTnnFixedTest, TestSamplingBroadcastPaddingMask) {
    // The padding mask every real caller builds is [1, 1, 1, V] -- ONE row, because which vocab
    // columns are padding does not depend on the token position (see _sample_logits_mask in
    // generate.py and _build_logits_mask in llama_completer.py). It must apply to every token row.
    //
    // The other masked tests in this file hand in a FULL-SIZE [1, 1, rows, V] mask, which exercises
    // a shape no caller passes and cannot catch a missing broadcast. This one can: in TILE layout a
    // height-1 tensor carries data only in row 0 of each tile, with rows 1..31 zero-filled, so an
    // implementation that subtracts tile-for-tile masks only token row 0 and lets every other row
    // argmax onto the first padding column.
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
    // TODO: shares the BH accuracy issue guarding TestSamplingPositiveTemperatureWithMask
    // (https://github.com/tenstorrent/tt-metal/issues/37342); the greedy half above still runs.
    auto board = tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0);
    if (board != tt::BoardType::P100 && board != tt::BoardType::P150) {
        auto sampled =
            ttml::core::to_vector<uint32_t>(ttml::ttnn_fixed::sample(tensor_logits, 1.0F, 4242, tensor_mask));
        ASSERT_EQ(sampled.size(), kBatch * kRows);
        for (uint32_t r = 0; r < kBatch * kRows; ++r) {
            EXPECT_LT(sampled[r], kVocab) << "row " << r << " sampled a masked padding column";
        }
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
    // (DO_GUMBEL_NOISE). The noise makes the winner unpredictable, but no index may ever leave the
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
    // Only the mask's has_value() reaches the program hash, never where it lives. The decoy column
    // outranks every real winner, so a mask read from the wrong memory space cannot go unnoticed.
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

    auto greedy_at = ttml::core::to_vector<uint32_t>(
        ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, positions));
    ASSERT_EQ(greedy_at.size(), kBatch) << "one sampled id per batch entry, not per token";
    EXPECT_EQ(greedy_at, expected_at_positions);

    // The scan bounds have to hold in the sampled kernel too, which is a separate binary
    // (DO_GUMBEL_NOISE). The noise makes the winner unpredictable, but every real logit here is
    // negative while from_xtensor zero-fills the padding, so any index that leaves the logical
    // vocabulary means the scan walked into padding.
    auto sampled_at = ttml::core::to_vector<uint32_t>(
        ttml::ttnn_fixed::sample(tensor_a, 1.0F, 99, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, positions));
    ASSERT_EQ(sampled_at.size(), kBatch);
    for (uint32_t b = 0; b < kBatch; ++b) {
        EXPECT_LT(sampled_at[b], kVocab) << "batch entry " << b << " left the logical vocabulary";
    }

    // Every entry pointed at the SAME row exercises the other extreme of the work split: all three
    // entries now read the same tile row of their own shard, and the boundary-merge path sees three
    // groups that each span whatever cores the split handed them.
    const std::vector<uint32_t> uniform(kBatch, kTokens - 1U);
    auto greedy_uniform = ttml::core::to_vector<uint32_t>(
        ttml::ttnn_fixed::sample(tensor_a, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, uniform));
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
            tensor, 0.0F, 7, /* mask */ std::nullopt, /* seed_axes */ std::nullopt, positions));

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
}
