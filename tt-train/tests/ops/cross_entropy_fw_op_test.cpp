
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"

class CrossEntropyForwardTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }
};

xt::xarray<float> calculate_cross_entropy_loss(const xt::xarray<float>& input, const xt::xarray<uint32_t>& target) {
    const uint32_t N = target.shape(0);
    const uint32_t C = 1U;
    const uint32_t H = target.shape(1);
    const uint32_t W = 1U;
    xt::xarray<float> target_inputs = xt::zeros<float>({N, C, H, W});

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            size_t class_index = target(n, h);
            target_inputs(n, 0, h, 0) = input(n, 0, h, class_index);
        }
    }

    xt::xarray<float> max_input = xt::amax(input, -1, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_input;
    xt::xarray<float> log_exp_sum_test = xt::log(xt::sum(xt::exp(shifted_input), -1, xt::keep_dims));
    xt::xarray<float> result = -target_inputs + max_input + log_exp_sum_test;
    return result;
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Small_Forward) {
    using namespace ttml;

    const uint32_t N = 1, H = 1;

    xt::xarray<float> input_tensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 1U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Negetive_Values) {
    using namespace ttml;

    const uint32_t N = 1, H = 2;

    xt::xarray<float> input_tensor = {{{{-100.F, -101.F, -102.F, -103.F}, {-5.01F, -5.02F, -0.3F, -7.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 0;
    target_tensor(0, 1) = 2U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 157U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

// Disabled: non-deterministic accuracy failures — https://github.com/tenstorrent/tt-metal/issues/46121
TEST_F(CrossEntropyForwardTest, DISABLED_CrossEntropyForward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1017U, W = 1018U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Forward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 65536U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_ProgramCache_MissAndHit) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    device->enable_program_cache();

    auto run_and_check = [&](uint32_t N, uint32_t H, uint32_t W, uint32_t seed) {
        xt::xarray<float> input_tensor =
            ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, 1U, H, W}, -10.0F, 10.0F, seed);
        xt::xarray<uint32_t> target_tensor =
            ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

        auto input = core::from_xtensor(input_tensor, device);
        auto target =
            core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(target_tensor, device, ttnn::Layout::ROW_MAJOR);

        auto result = ttml::metal::cross_entropy_fw(input, target);

        auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);
        auto result_xtensor = core::to_xtensor(result);
        EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
    };

    // Two distinct inner dimensions (W) generate different compile-time args, so each must compile a fresh
    // program (cache miss) rather than reuse the other's cached program.
    const auto entries_start = device->num_program_cache_entries();
    run_and_check(/*N=*/2U, /*H=*/32U, /*W=*/64U, /*seed=*/101U);
    const auto entries_after_first = device->num_program_cache_entries();
    EXPECT_GT(entries_after_first, entries_start) << "first shape did not populate the program cache";

    run_and_check(/*N=*/2U, /*H=*/32U, /*W=*/128U, /*seed=*/103U);
    const auto entries_after_second = device->num_program_cache_entries();
    EXPECT_GT(entries_after_second, entries_after_first)
        << "a distinct input shape reused the first shape's cached program (incomplete program-cache key)";

    // Re-running the first shape must reuse its cached program (cache hit), leaving the entry count unchanged.
    run_and_check(/*N=*/2U, /*H=*/32U, /*W=*/64U, /*seed=*/101U);
    const auto entries_after_repeat = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_repeat, entries_after_second)
        << "re-running the first shape compiled a new program instead of hitting the cache";
}

TEST_F(CrossEntropyForwardTest, NIGHTLY_CrossEntropyForward_Huge_Forward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}
