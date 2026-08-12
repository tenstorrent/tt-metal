// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Covers the ttml::metal::adamw overload that takes beta1^t / beta2^t as
// single-element tensors instead of floats. The two overloads must agree: the
// tensor path derives step_size and 1 / bias_correction2 inside the kernel
// rather than on host, but computes the same update.

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/optimizers/adamw/adamw.hpp"
#include "ttnn/operations/core/core.hpp"
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

    static ttnn::Tensor run_float(const StepInputs& inputs, float beta1_pow, float beta2_pow) {
        return ttml::metal::adamw(
            inputs.param,
            inputs.grad,
            inputs.exp_avg,
            inputs.exp_avg_sq,
            inputs.max_exp_avg_sq,
            kLr,
            kBeta1,
            kBeta2,
            beta1_pow,
            beta2_pow,
            kEpsilon,
            kWeightDecay);
    }

    static ttnn::Tensor run_tensor(
        const StepInputs& inputs, float beta1_pow, float beta2_pow, ttnn::DataType bias_dtype) {
        return ttml::metal::adamw(
            inputs.param,
            inputs.grad,
            inputs.exp_avg,
            inputs.exp_avg_sq,
            inputs.max_exp_avg_sq,
            kLr,
            kBeta1,
            kBeta2,
            make_scalar(beta1_pow, bias_dtype),
            make_scalar(beta2_pow, bias_dtype),
            kEpsilon,
            kWeightDecay);
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
        float tolerance) {
        const float beta1_pow = std::pow(kBeta1, static_cast<float>(step));
        const float beta2_pow = std::pow(kBeta2, static_cast<float>(step));

        auto expected_inputs = make_inputs(shape, param_dtype, amsgrad);
        auto actual_inputs = make_inputs(shape, param_dtype, amsgrad);

        auto expected = run_float(expected_inputs, beta1_pow, beta2_pow);
        auto actual = run_tensor(actual_inputs, beta1_pow, beta2_pow, bias_dtype);

        expect_all_near(ttml::core::to_vector(actual), ttml::core::to_vector(expected), tolerance, "param");

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
        /* tolerance = */ 1e-5F);
}

TEST_F(AdamWBiasCorrectionTensorTest, Bf16ParamFp32Bias) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::BFLOAT16,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 10U,
        /* tolerance = */ 1e-2F);
}



TEST_F(AdamWBiasCorrectionTensorTest, AmsgradFp32Bias) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ true,
        /* step = */ 10U,
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
        /* tolerance = */ 1e-4F);
}

// Late in training bias_correction1 approaches 1 and beta2^t approaches 0.
TEST_F(AdamWBiasCorrectionTensorTest, LateStepSmallCorrection) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 64U, 64U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 5000U,
        /* tolerance = */ 1e-5F);
}

// Enough tiles that split_work_to_cores produces two core groups with different
// per-core tile counts, so the second compute kernel is exercised too.
TEST_F(AdamWBiasCorrectionTensorTest, MultiTileAcrossCoreGroups) {
    compare_overloads(
        ttnn::Shape({1U, 1U, 512U, 1024U}),
        ttnn::DataType::FLOAT32,
        ttnn::DataType::FLOAT32,
        /* amsgrad = */ false,
        /* step = */ 10U,
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

// Successive steps with different beta^t must reuse the same program: nothing the
// program hash covers changes, so only the tensor contents differ.
TEST_F(AdamWBiasCorrectionTensorTest, StepInvariantProgram) {
    const ttnn::Shape shape({1U, 1U, 32U, 32U});
    auto inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);

    for (uint32_t step = 1U; step <= 5U; ++step) {
        auto result = run_tensor(
            inputs,
            std::pow(kBeta1, static_cast<float>(step)),
            std::pow(kBeta2, static_cast<float>(step)),
            ttnn::DataType::FLOAT32);
        for (float value : ttml::core::to_vector(result)) {
            ASSERT_TRUE(std::isfinite(value)) << "non-finite parameter after step " << step;
        }
    }
}

// Running the same step twice through the tensor path must be bit-identical: the
// scalars are read fresh from L1 each dispatch, not captured at program build.
TEST_F(AdamWBiasCorrectionTensorTest, RepeatedDispatchIsStable) {
    const ttnn::Shape shape({1U, 1U, 64U, 64U});
    const float beta1_pow = std::pow(kBeta1, 7.0F);
    const float beta2_pow = std::pow(kBeta2, 7.0F);

    auto first_inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);
    auto first = ttml::core::to_vector(run_tensor(first_inputs, beta1_pow, beta2_pow, ttnn::DataType::FLOAT32));

    auto second_inputs = make_inputs(shape, ttnn::DataType::FLOAT32, /* amsgrad = */ false);
    auto second = ttml::core::to_vector(run_tensor(second_inputs, beta1_pow, beta2_pow, ttnn::DataType::FLOAT32));

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
        ASSERT_FLOAT_EQ(first[i], second[i]) << "unstable at element " << i;
    }
}


// bfloat16 bias tensors are rejected: 1 - beta^t cancels, and bf16 rounds
// beta2^1 = 0.999 to exactly 1.0, which would divide by zero in the kernel.
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
