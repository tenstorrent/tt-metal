// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <utility>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "cross_entropy_test_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/cross_entropy_bw/device/cross_entropy_bw_device_operation.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

class CrossEntropyBackwardTest : public ::testing::Test {
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

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

xt::xarray<float> calculate_cross_entropy_backward(
    const xt::xarray<float>& input, const xt::xarray<uint32_t>& target, const float scaler = 1.0F) {
    const uint32_t N = target.shape(0);
    const uint32_t H = target.shape(1);

    const auto input_shape = input.shape();
    xt::xarray<float> target_inputs = xt::zeros<float>(input_shape);

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            size_t class_index = target(n, h);
            target_inputs(n, 0, h, class_index) = 1.0F;
        }
    }

    xt::xarray<float> scaler_tensor(input_shape);
    scaler_tensor.fill(scaler);

    xt::xarray<float> max_input = xt::amax(input, -1, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_input;
    xt::xarray<float> exp_shifted_input = xt::exp(shifted_input);
    xt::xarray<float> exp_sum = xt::sum(exp_shifted_input, -1, xt::keep_dims);
    xt::xarray<float> result = exp_shifted_input / exp_sum - target_inputs;
    return result * scaler_tensor;
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Small_Backward) {
    using namespace ttml;

    const uint32_t N = 1U, H = 1U;

    xt::xarray<float> input_tensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 1U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});
    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 91U, W = 187U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1024, W = 1024U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Large_Backward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 32U, W = 128007U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, NIGHTLY_CrossEntropyBackward_Huge_Backward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 64, W = 128000U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyForwardBackward_ReduceMeanVsNone) {
    using namespace ttml;

    const uint32_t N = 5U, C = 1U, H = 91U, W = 187U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = ttml::autograd::create_tensor(
        core::from_xtensor(input_tensor, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto target = ttml::autograd::create_tensor(core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR));

    auto result_none = ttml::ops::cross_entropy_loss(input, target, ttml::ops::ReduceType::NONE);
    auto result_none_with_mean_after = ttml::ops::mean(result_none);
    auto result_mean = ttml::ops::cross_entropy_loss(input, target, ttml::ops::ReduceType::MEAN);

    result_mean->backward();
    result_none_with_mean_after->backward();

    auto result_mean_grad = core::to_xtensor(result_mean->get_grad());
    auto result_none_with_mean_after_grad = core::to_xtensor(result_none_with_mean_after->get_grad());

    auto result_none_after_mean_xtensor = core::to_xtensor(result_none_with_mean_after->get_value());
    auto result_mean_xtensor = core::to_xtensor(result_mean->get_value());

    assert((result_none_after_mean_xtensor.shape() == result_mean_xtensor.shape()));
    EXPECT_TRUE(xt::allclose(result_none_after_mean_xtensor, result_mean_xtensor, 3e-2F, 1e-2F));
    assert((result_none_with_mean_after_grad.shape() == result_mean_grad.shape()));
    EXPECT_TRUE(xt::allclose(result_none_with_mean_after_grad, result_mean_grad, 3e-2F, 1e-2F));
}

namespace {

using namespace ttml::test_utils::cross_entropy;

// The gradient scales with `scaler`, so compare at unit scale: the absolute tolerance would accept anything for a
// MEAN-sized scaler.
void expect_grad_matches(const ttnn::Tensor& result, const LaunchTensors& in, float scaler, const std::string& launch) {
    const xt::xarray<float> expected =
        calculate_cross_entropy_backward(in.input_values, in.target_values, scaler) / scaler;
    const xt::xarray<float> actual = ttml::core::to_xtensor(result) / scaler;
    ASSERT_EQ(actual.shape(), expected.shape()) << launch;
    EXPECT_TRUE(xt::allclose(actual, expected, 3e-2F, 1e-2F)) << launch;
}

}  // namespace

// Calls the prim because the public wrapper multiplies by grad: its op output is a temporary that the next launch can
// get back at the same address, which would hide a stale output address.
TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_CacheHitFreshTensors_TwoCoreGroups) {
    constexpr float kScaler = 1.0F;  // ReduceType::NONE
    run_cache_hit_fresh_tensors(
        /*W=*/187U,
        [](const LaunchTensors& in) { return ttnn::prim::ttml_cross_entropy_bw(in.input, in.target, kScaler); },
        [](const ttnn::Tensor& result, const LaunchTensors& in, const std::string& label) {
            expect_grad_matches(result, in, kScaler, label);
        });
}

// Preallocated outputs start at 5, which no gradient can be.
TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_PreallocatedOutput_CacheHit) {
    constexpr float kScaler = 1.0F;  // ReduceType::NONE
    run_preallocated_output_cache_hits(
        /*W=*/61U,
        /*output_width=*/61U,
        /*sentinel=*/5.0F,
        [](const LaunchTensors& in, const std::optional<ttnn::Tensor>& output) {
            return ttnn::prim::ttml_cross_entropy_bw(in.input, in.target, kScaler, output);
        },
        [](const ttnn::Tensor& result, const LaunchTensors& in, const std::string& label) {
            expect_grad_matches(result, in, kScaler, label);
        });
}

// scaler is compiled into the writer and compute kernels, so it must be part of the program key: without it, the second
// scaler's launch would replay the first scaler's program. Each scaler's second launch must hit its own program.
TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_ScalerIsPartOfProgramCacheKey) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();

    const uint32_t N = 2U, H = test_utils::two_core_group_seq_len(7U), W = 61U;
    const float none_scaler = 1.0F;                                                    // ReduceType::NONE
    const float mean_scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));  // ReduceType::MEAN
    auto& rng = autograd::ctx().get_generator();
    const uint32_t seed = rng();
    // Returns the result and how many program cache entries the launch added.
    const auto launch = [&](const LaunchTensors& in, float scaler) {
        const auto entries_before = device.num_program_cache_entries();
        auto result = ttnn::prim::ttml_cross_entropy_bw(in.input, in.target, scaler);
        return std::pair{std::move(result), device.num_program_cache_entries() - entries_before};
    };

    const auto a = make_inputs(N, H, W, seed);
    const auto [result_a, new_entries_a] = launch(a, none_scaler);
    EXPECT_GT(new_entries_a, 0U) << "first scaler did not populate the cache";

    const auto b = make_inputs(N, H, W, seed + 2U);
    const auto [result_b, new_entries_b] = launch(b, mean_scaler);
    EXPECT_GT(new_entries_b, 0U) << "second scaler reused the first scaler's program";

    const auto c = make_inputs(N, H, W, seed + 4U);
    const auto [result_c, new_entries_c] = launch(c, none_scaler);
    EXPECT_EQ(new_entries_c, 0U) << "first scaler's second launch was not a cache hit";

    const auto d = make_inputs(N, H, W, seed + 6U);
    const auto [result_d, new_entries_d] = launch(d, mean_scaler);
    EXPECT_EQ(new_entries_d, 0U) << "second scaler's second launch was not a cache hit";

    expect_grad_matches(result_a, a, none_scaler, "first scaler (miss)");
    expect_grad_matches(result_b, b, mean_scaler, "second scaler (miss)");
    expect_grad_matches(result_c, c, none_scaler, "first scaler (hit)");
    expect_grad_matches(result_d, d, mean_scaler, "second scaler (hit)");
}

// Two training steps through ops::cross_entropy_loss on fresh tensors, the first step's kept alive: the second step
// builds no programs and must read its own prediction and target. Both ops' outputs are temporaries freed within each
// step, so their output addresses are covered by each op's prim-level tests, not here. Checks the gradient that
// reaches the prediction, which ReduceMeanVsNone does not.
TEST_F(CrossEntropyBackwardTest, CrossEntropyLoss_MeanBackward_CacheHitFreshTensors) {
    using namespace ttml;
    auto& device = test_utils::device_with_empty_program_cache();

    const uint32_t N = 2U, H = test_utils::two_core_group_seq_len(7U), W = 61U;
    const float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));  // what MEAN passes to the bw op
    auto& rng = autograd::ctx().get_generator();
    const uint32_t seed = rng();
    const auto step = [](const LaunchTensors& in) {
        auto prediction = autograd::create_tensor(in.input, /* requires_grad */ true);
        auto target = autograd::create_tensor(in.target);
        auto loss = ops::cross_entropy_loss(prediction, target, ops::ReduceType::MEAN);
        loss->backward();
        return prediction;
    };

    const auto a = make_inputs(N, H, W, seed);
    const auto prediction_a = step(a);
    const auto entries_after_first_step = device.num_program_cache_entries();

    const auto b = make_inputs(N, H, W, seed + 2U);
    const auto prediction_b = step(b);
    EXPECT_EQ(device.num_program_cache_entries(), entries_after_first_step) << "second step built new programs";

    expect_grad_matches(prediction_b->get_grad(), b, scaler, "second step");
    expect_grad_matches(prediction_a->get_grad(), a, scaler, "first step, re-read after the second");
}
