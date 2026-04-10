// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <random>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

class GatherLogitTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// CPU reference: for each (n, h), output[n, 0, h, 0] = logit[n, 0, h, c] if c in [first_v, last_v), else 0.
static xt::xarray<float> gather_logit_reference(
    const xt::xarray<float>& logit,      // [N, 1, H, V]
    const xt::xarray<uint32_t>& target,  // [N, H]
    uint32_t first_v,
    uint32_t last_v) {
    const uint32_t N = static_cast<uint32_t>(target.shape(0));
    const uint32_t H = static_cast<uint32_t>(target.shape(1));
    xt::xarray<float> result = xt::zeros<float>({N, 1U, H, 1U});
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            const uint32_t c = target(n, h);
            if (c >= first_v && c < last_v) {
                result(n, 0U, h, 0U) = logit(n, 0U, h, c);
            }
        }
    }
    return result;
}

TEST_F(GatherLogitTest, SmallFullVocab) {
    using namespace ttml;

    // logit [1, 1, 1, 8], target [1, 1], full vocab
    xt::xarray<float> logit_t = {{{{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 1U});
    target_t(0, 0) = 3U;

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::gather_logit(logit_dev, target_dev, 0U, 8U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = gather_logit_reference(logit_t, target_t, 0U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/1e-3F, /*atol=*/1e-3F));
}

TEST_F(GatherLogitTest, SmallPartialVocab) {
    using namespace ttml;

    // logit [1, 1, 2, 8], target [1, 2]
    // first_v=4, last_v=8 — only targets >= 4 should be gathered
    xt::xarray<float> logit_t = {
        {{{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F}, {-1.F, -2.F, -3.F, -4.F, -5.F, -6.F, -7.F, -8.F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 2U});
    target_t(0, 0) = 2U;  // out of shard [4, 8) → output should be 0
    target_t(0, 1) = 5U;  // in shard        → output should be logit[0,0,1,5] = -6

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::gather_logit(logit_dev, target_dev, 4U, 8U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = gather_logit_reference(logit_t, target_t, 4U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/1e-3F, /*atol=*/1e-3F));

    // Explicit checks
    EXPECT_NEAR(result_xt(0, 0, 0, 0), 0.F, 1e-3F);   // out of shard → 0
    EXPECT_NEAR(result_xt(0, 0, 1, 0), -6.F, 1e-3F);  // logit[0,0,1,5] = -6
}

TEST_F(GatherLogitTest, BatchedNonAlignedShape) {
    using namespace ttml;

    // Non-tile-aligned H and V — mirrors cross_entropy_fw_op_test pattern
    const uint32_t N = 2U, H = 91U, V = 157U;

    std::mt19937 gen(42);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, H, V});
    std::uniform_real_distribution<float> float_dist(-5.F, 5.F);
    for (auto& v : logit_t) {
        v = float_dist(gen);
    }

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, H});
    std::uniform_int_distribution<uint32_t> idx_dist(0U, V - 1U);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            target_t(n, h) = idx_dist(gen);
        }
    }

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::gather_logit(logit_dev, target_dev, 0U, V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = gather_logit_reference(logit_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(GatherLogitTest, BatchedPartialVocabShard) {
    // Simulates a TP shard: only the second half of vocab is "local"
    using namespace ttml;

    const uint32_t N = 2U, H = 64U, V = 128U;
    const uint32_t first_v = V / 2;  // 64
    const uint32_t last_v = V;       // 128

    std::mt19937 gen(7);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, H, V});
    std::uniform_real_distribution<float> float_dist(-3.F, 3.F);
    for (auto& v : logit_t) {
        v = float_dist(gen);
    }

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, H});
    // Half the targets fall in [0, 64), half in [64, 128)
    std::uniform_int_distribution<uint32_t> idx_dist(0U, V - 1U);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            target_t(n, h) = idx_dist(gen);
        }
    }

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::gather_logit(logit_dev, target_dev, first_v, last_v);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = gather_logit_reference(logit_t, target_t, first_v, last_v);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(GatherLogitTest, LargeVocab) {
    using namespace ttml;

    const uint32_t N = 1U, H = 1U, V = 32768U;

    std::mt19937 gen(99);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, H, V});
    std::uniform_real_distribution<float> float_dist(-10.F, 10.F);
    for (auto& v : logit_t) {
        v = float_dist(gen);
    }

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, H});
    target_t(0, 0) = 12345U;

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::gather_logit(logit_dev, target_dev, 0U, V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = gather_logit_reference(logit_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(GatherLogitTest, NIGHTLY_LargeBatchLargeVocab) {
    using namespace ttml;

    const uint32_t N = 64U, H = 32U, V = 128000U;

    std::mt19937 gen(42);
    xt::xarray<float> logit_t = xt::empty<float>({N, 1U, H, V});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{logit_t.data(), logit_t.size()},
        []() { return std::uniform_real_distribution<float>(-10.F, 10.F); },
        seed);

    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, H});
    std::uniform_int_distribution<uint32_t> idx_dist(0U, V - 1U);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            target_t(n, h) = idx_dist(gen);
        }
    }

    auto logit_dev = core::from_xtensor(logit_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::gather_logit(logit_dev, target_dev, 0U, V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = gather_logit_reference(logit_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}
