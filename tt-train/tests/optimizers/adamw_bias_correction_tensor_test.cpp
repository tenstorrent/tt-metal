// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Covers the ttml::metal::adamw overload that takes beta1^t / beta2^t as
// single-element tensors instead of floats. The two overloads must agree: the
// tensor path derives step_size and 1 / bias_correction2 inside the kernel
// rather than on host, but computes the same update.

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/optimizers/adamw/adamw.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace {

constexpr float kLr = 1e-3F;
constexpr float kBeta1 = 0.9F;
constexpr float kBeta2 = 0.999F;
constexpr float kEpsilon = 1e-8F;
constexpr float kWeightDecay = 0.01F;

// One run of the optimizer step, so the float and tensor overloads can be driven
// over the same inputs and compared.
struct StepInputs {
    ttnn::Tensor param;
    ttnn::Tensor grad;
    ttnn::Tensor exp_avg;
    ttnn::Tensor exp_avg_sq;
    std::optional<ttnn::Tensor> max_exp_avg_sq;
};

class AdamWBiasCorrectionTensorTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }

    static std::vector<float> random_values(size_t count, float low, float high, uint32_t seed) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(low, high);
        std::vector<float> values(count);
        for (float& value : values) {
            value = distribution(generator);
        }
        return values;
    }

    static ttnn::Tensor make_fp32(const ttnn::Shape& shape, float low, float high, uint32_t seed) {
        return ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(
            random_values(shape.volume(), low, high, seed), shape, &ttml::autograd::ctx().get_device());
    }

    static ttnn::Tensor make_bf16(const ttnn::Shape& shape, float low, float high, uint32_t seed) {
        return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(
            random_values(shape.volume(), low, high, seed), shape, &ttml::autograd::ctx().get_device());
    }

    static ttnn::Tensor make_param(
        const ttnn::Shape& shape, float low, float high, uint32_t seed, ttnn::DataType dtype) {
        return dtype == ttnn::DataType::FLOAT32 ? make_fp32(shape, low, high, seed) : make_bf16(shape, low, high, seed);
    }

    // A single-element tensor holding `value`, which is what the overload takes.
    static ttnn::Tensor make_scalar(float value, ttnn::DataType dtype) {
        const ttnn::Shape shape({1U, 1U, 1U, 1U});
        auto* device = &ttml::autograd::ctx().get_device();
        if (dtype == ttnn::DataType::FLOAT32) {
            return ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(std::vector<float>{value}, shape, device);
        }
        return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(std::vector<float>{value}, shape, device);
    }

    // Fresh state for one run. Same seeds produce identical inputs, so two calls
    // give the two overloads the same starting point (the op mutates in place).
    static StepInputs make_inputs(const ttnn::Shape& shape, ttnn::DataType param_dtype, bool amsgrad) {
        StepInputs inputs{
            .param = make_param(shape, -1.0F, 1.0F, /* seed = */ 2U, param_dtype),
            .grad = make_bf16(shape, -1.0F, 1.0F, /* seed = */ 1U),
            .exp_avg = make_param(shape, -1.0F, 1.0F, /* seed = */ 3U, param_dtype),
            .exp_avg_sq = make_param(shape, 0.0F, 1.0F, /* seed = */ 4U, param_dtype),
            .max_exp_avg_sq = std::nullopt,
        };
        if (amsgrad) {
            inputs.max_exp_avg_sq = make_param(shape, 0.0F, 1.0F, /* seed = */ 5U, param_dtype);
        }
        return inputs;
    }

    static ttnn::Tensor run_float(const StepInputs& inputs, float beta1_pow, float beta2_pow, float lr) {
        return ttml::metal::adamw(
            inputs.param,
            inputs.grad,
            inputs.exp_avg,
            inputs.exp_avg_sq,
            inputs.max_exp_avg_sq,
            lr,
            kBeta1,
            kBeta2,
            beta1_pow,
            beta2_pow,
            kEpsilon,
            kWeightDecay);
    }

    // Takes the bias tensors as-is, so a caller can hand the same two buffers to
    // several successive steps instead of allocating a fresh pair each time.
    static ttnn::Tensor run_tensor_with(
        const StepInputs& inputs, const ttnn::Tensor& beta1_pow, const ttnn::Tensor& beta2_pow, float lr) {
        return ttml::metal::adamw(
            inputs.param,
            inputs.grad,
            inputs.exp_avg,
            inputs.exp_avg_sq,
            inputs.max_exp_avg_sq,
            lr,
            kBeta1,
            kBeta2,
            beta1_pow,
            beta2_pow,
            kEpsilon,
            kWeightDecay);
    }

    static ttnn::Tensor run_tensor(
        const StepInputs& inputs, float beta1_pow, float beta2_pow, ttnn::DataType bias_dtype, float lr) {
        return run_tensor_with(
            inputs, make_scalar(beta1_pow, bias_dtype), make_scalar(beta2_pow, bias_dtype), lr);
    }

    static void expect_all_near(
        const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const char* what) {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_NEAR(actual[i], expected[i], tolerance) << what << " mismatch at element " << i;
        }
    }

    // Drives both overloads over identical inputs and requires them to agree on the
    // updated parameter and on both moments.
    static void compare_overloads(
        const ttnn::Shape& shape,
        ttnn::DataType param_dtype,
        ttnn::DataType bias_dtype,
        bool amsgrad,
        uint32_t step,
        float lr,
        float tolerance) {
        const float beta1_pow = std::pow(kBeta1, static_cast<float>(step));
        const float beta2_pow = std::pow(kBeta2, static_cast<float>(step));

        auto expected_inputs = make_inputs(shape, param_dtype, amsgrad);
        auto actual_inputs = make_inputs(shape, param_dtype, amsgrad);

        // The op updates the parameter in place, so snapshot it before running.
        const auto initial = ttml::core::to_vector(expected_inputs.param);

        auto expected = run_float(expected_inputs, beta1_pow, beta2_pow, lr);
        auto actual = run_tensor(actual_inputs, beta1_pow, beta2_pow, bias_dtype, lr);

        const auto expected_values = ttml::core::to_vector(expected);

        // Without this the comparison below can be vacuous: if the tolerance is as
        // large as the update itself, the test passes even when the tensor path
        // leaves the parameter untouched.
        float largest_update = 0.0F;
        for (size_t i = 0; i < expected_values.size(); ++i) {
            largest_update = std::max(largest_update, std::abs(expected_values[i] - initial[i]));
        }
        ASSERT_GT(largest_update, 4.0F * tolerance)
            << "tolerance " << tolerance << " is not small against the update " << largest_update
            << "; this comparison could not fail";

        expect_all_near(ttml::core::to_vector(actual), expected_values, tolerance, "param");

        // The moments do not involve the bias correction, so they must agree far
        // more tightly than the parameter does.
        expect_all_near(
            ttml::core::to_vector(actual_inputs.exp_avg),
            ttml::core::to_vector(expected_inputs.exp_avg),
            1e-6F,
            "exp_avg");
        expect_all_near(
            ttml::core::to_vector(actual_inputs.exp_avg_sq),
            ttml::core::to_vector(expected_inputs.exp_avg_sq),
            1e-6F,
            "exp_avg_sq");
        if (amsgrad) {
            expect_all_near(
                ttml::core::to_vector(actual_inputs.max_exp_avg_sq.value()),
                ttml::core::to_vector(expected_inputs.max_exp_avg_sq.value()),
                1e-6F,
                "max_exp_avg_sq");
        }
    }
};

}  // namespace

// The parameter tolerance is loose by construction: the tensor path scales the
// variance by 1 / bias_correction2 before the square root, the float path by
// 1 / sqrt(bias_correction2) after it.

TEST_F(AdamWBiasCorrectionTensorTest, Fp32ParamFp32Bias) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 10U,
        /* lr = */ kLr,
        /* tolerance = */ 1e-5F);
}

// Driven at lr = 1.0 rather than kLr. One step at kLr moves a parameter by ~1e-4,
// which is far below one bfloat16 ulp (~4e-3 for values near 1), so the update
// rounds away and no tolerance can tell the two overloads apart. A large lr puts
// the update well above the ulp and makes the comparison mean something.
TEST_F(AdamWBiasCorrectionTensorTest, Bf16ParamFp32Bias) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::BFLOAT16,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 10U,
        /* lr = */ 1.0F,
        /* tolerance = */ 1e-2F);
}

TEST_F(AdamWBiasCorrectionTensorTest, AmsgradFp32Bias) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ true,
        /* step = */ 10U,
        /* lr = */ kLr,
        /* tolerance = */ 1e-5F);
}

// Step 1 is the extreme case: bias_correction1 is 1 - beta1 = 0.1, so step_size is
// 10x the learning rate, and 1 / bias_correction2 is ~1000.
TEST_F(AdamWBiasCorrectionTensorTest, FirstStepLargeCorrection) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 1U,
        /* lr = */ kLr,
        /* tolerance = */ 1e-5F);
}

// Late in training bias_correction1 approaches 1 and beta2^t approaches 0.
TEST_F(AdamWBiasCorrectionTensorTest, LateStepSmallCorrection) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 5000U,
        /* lr = */ kLr,
        /* tolerance = */ 1e-5F);
}

// 7 x 23 = 161 tiles. split_work_to_cores only produces a second core group when
// the tile count does not divide evenly across the grid, so the count is chosen
// coprime to the usual grids - 161 leaves a remainder on 8x8 = 64 cores (the
// earlier 512x1024 = 512 tiles divided exactly there and left core_group_2 empty),
// 8x7 = 56, and 13x10 = 130.
TEST_F(AdamWBiasCorrectionTensorTest, MultiTileAcrossCoreGroups) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 224U, 736U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 10U,
        /* lr = */ kLr,
        /* tolerance = */ 1e-5F);
}

// Stochastic rounding needs bf16 parameters, and makes the update non-deterministic,
// so this only asserts the step runs and stays finite.
TEST_F(AdamWBiasCorrectionTensorTest, StochasticRounding) {
    const ttnn::Shape shape({1U, 1U, 64U, 64U});
    auto inputs = make_inputs(shape, ttnn::DataType::BFLOAT16, /* amsgrad = */ false);

    auto result = ttml::metal::adamw(
        inputs.param,
        inputs.grad,
        inputs.exp_avg,
        inputs.exp_avg_sq,
        inputs.max_exp_avg_sq,
        kLr,
        kBeta1,
        kBeta2,
        make_scalar(std::pow(kBeta1, 10.0F), ttnn::DataType::FLOAT32),
        make_scalar(std::pow(kBeta2, 10.0F), ttnn::DataType::FLOAT32),
        kEpsilon,
        kWeightDecay,
        ttml::metal::StochasticRounding::Enabled,
        /* stochastic_rounding_seed = */ 1234U);

    for (float value : ttml::core::to_vector(result)) {
        ASSERT_TRUE(std::isfinite(value));
    }
}

// The central claim of the overload: beta^t is read out of L1 on every dispatch,
// not folded in when the program is built. Nothing in the program hash changes
// between these two calls, so the second reuses the program the first built. If
// the kernel had captured beta^t at build time, the step-400 call would still be
// applying the step-1 correction and would disagree with the float overload.
//
// This has to be self-contained rather than relying on an earlier test having
// warmed the cache for this configuration, which would make it order-dependent.
TEST_F(AdamWBiasCorrectionTensorTest, CachedProgramPicksUpNewBetaPow) {
    const ttnn::Shape shape({1U, 1U, 32U, 32U});

    ASSERT_NO_FATAL_FAILURE(compare_overloads(
        shape,
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 1U,
        /* lr = */ kLr,
        /* tolerance = */ 1e-5F));

    ASSERT_NO_FATAL_FAILURE(compare_overloads(
        shape,
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 400U,
        /* lr = */ kLr,
        /* tolerance = */ 1e-5F));
}

// The usage pattern adamw.hpp mandates for trace: hold beta^t in two buffers and
// advance them in place, so the addresses the reader was given stay valid. Runs
// two successive steps against one pair of bias tensors, advancing them with an
// in-place device multiply between dispatches, and requires the result to match
// the float overload driven over the same two steps.
//
// Deliberately not a trace capture/replay test: tt-train opens its mesh device
// with DEFAULT_TRACE_REGION_SIZE == 0 (see ttml::core::MeshDevice), so no trace
// can be captured here without changing device setup for the whole test suite.
// What this does cover is the device-side half - the kernel reading whatever the
// buffers hold at dispatch, never a value bound earlier.
TEST_F(AdamWBiasCorrectionTensorTest, InPlaceBetaPowUpdateAcrossSteps) {
    const ttnn::Shape shape({1U, 1U, 64U, 64U});
    constexpr uint32_t kSteps = 2U;

    auto tensor_inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);
    auto float_inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);

    // One pair of buffers for the whole run, as a trace-capturing caller must do.
    auto beta1_pow = make_scalar(1.0F, ttnn::DataType::FLOAT32);
    auto beta2_pow = make_scalar(1.0F, ttnn::DataType::FLOAT32);
    const auto* beta1_pow_buffer = beta1_pow.buffer();
    const auto* beta2_pow_buffer = beta2_pow.buffer();

    for (uint32_t step = 1U; step <= kSteps; ++step) {
        // beta^t *= beta, in place - same buffers, new contents.
        ttnn::multiply_(beta1_pow, kBeta1);
        ttnn::multiply_(beta2_pow, kBeta2);
        ASSERT_EQ(beta1_pow.buffer(), beta1_pow_buffer) << "beta1_pow was reallocated at step " << step;
        ASSERT_EQ(beta2_pow.buffer(), beta2_pow_buffer) << "beta2_pow was reallocated at step " << step;

        run_tensor_with(tensor_inputs, beta1_pow, beta2_pow, kLr);
        run_float(
            float_inputs,
            std::pow(kBeta1, static_cast<float>(step)),
            std::pow(kBeta2, static_cast<float>(step)),
            kLr);
    }

    expect_all_near(
        ttml::core::to_vector(tensor_inputs.param), ttml::core::to_vector(float_inputs.param), 1e-5F, "param");
}

// Two dispatches of the same step over identical inputs must land on the same
// values. Only checks determinism - it cannot distinguish a fresh L1 read from a
// captured scalar, since both dispatches use the same beta^t. That distinction is
// CachedProgramPicksUpNewBetaPow's job.
TEST_F(AdamWBiasCorrectionTensorTest, RepeatedDispatchIsStable) {
    const ttnn::Shape shape({1U, 1U, 64U, 64U});
    const float beta1_pow = std::pow(kBeta1, 7.0F);
    const float beta2_pow = std::pow(kBeta2, 7.0F);

    auto first_inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);
    auto first = ttml::core::to_vector(run_tensor(first_inputs, beta1_pow, beta2_pow, ttnn::DataType::FLOAT32, kLr));

    auto second_inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);
    auto second = ttml::core::to_vector(run_tensor(second_inputs, beta1_pow, beta2_pow, ttnn::DataType::FLOAT32, kLr));

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
        ASSERT_FLOAT_EQ(first[i], second[i]) << "unstable at element " << i;
    }
}

// bfloat16 bias tensors are rejected: 1 - beta^t cancels, and the nearest bfloat16
// value to beta2^1 = 0.999 is exactly 1.0, so the kernel would divide by zero on
// the first step.
TEST_F(AdamWBiasCorrectionTensorTest, RejectsBf16Bias) {
    const ttnn::Shape shape({1U, 1U, 64U, 64U});
    auto inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);

    EXPECT_ANY_THROW({
        ttml::metal::adamw(
            inputs.param,
            inputs.grad,
            inputs.exp_avg,
            inputs.exp_avg_sq,
            inputs.max_exp_avg_sq,
            kLr,
            kBeta1,
            kBeta2,
            make_scalar(0.9F, ttnn::DataType::BFLOAT16),
            make_scalar(0.999F, ttnn::DataType::BFLOAT16),
            kEpsilon,
            kWeightDecay);
    });
}

// A bias tensor with more than one element would leave the kernel reading element
// (0, 0) and silently ignoring the rest.
TEST_F(AdamWBiasCorrectionTensorTest, RejectsNonScalarBias) {
    const ttnn::Shape shape({1U, 1U, 64U, 64U});
    auto inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);

    auto non_scalar = make_fp32(ttnn::Shape({1U, 1U, 1U, 4U}), 0.1F, 0.9F, /* seed = */ 9U);
    EXPECT_ANY_THROW({
        ttml::metal::adamw(
            inputs.param,
            inputs.grad,
            inputs.exp_avg,
            inputs.exp_avg_sq,
            inputs.max_exp_avg_sq,
            kLr,
            kBeta1,
            kBeta2,
            non_scalar,
            make_scalar(0.999F, ttnn::DataType::FLOAT32),
            kEpsilon,
            kWeightDecay);
    });
}
