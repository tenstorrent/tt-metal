// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Parity tests for the tensor-scalar ttml::metal::adamw overload: the variant
// taking the step-varying scalars as single-element f32 device tensors must
// produce bit-identical results to the float-scalar ttml::metal::adamw.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "autograd/auto_context.hpp"
#include "metal/optimizers/adamw/adamw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace {

class AdamWTensorScalarsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

ttnn::Tensor make_device_tensor(
    const std::vector<float>& data, const ttnn::Shape& shape, ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto spec = tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    return ttnn::Tensor::from_vector(data, spec, device);
}

ttnn::Tensor make_scalar_tensor(float value, ttnn::DataType dtype = ttnn::DataType::FLOAT32) {
    return make_device_tensor({value}, ttnn::Shape({1, 1, 1, 1}), dtype);
}

struct AdamWHyperParams {
    float lr{1e-3F};
    float beta1{0.9F};
    float beta2{0.999F};
    float epsilon{1e-8F};
    float weight_decay{0.01F};
};

void run_parity_step(
    ttnn::DataType param_dtype, bool amsgrad, uint32_t step, std::vector<ttnn::Tensor>* keep_alive = nullptr) {
    const ttnn::Shape shape({1, 1, 64, 96});
    const AdamWHyperParams hp{};

    const auto param_data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, /*seed=*/0U);
    const auto grad_data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, /*seed=*/1U);
    const auto exp_avg_data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, /*seed=*/2U);
    const auto exp_avg_sq_data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), 0.0F, 1.0F, /*seed=*/3U);
    const auto max_exp_avg_sq_data =
        ttml::test_utils::make_uniform_vector<float>(shape.volume(), 0.0F, 1.0F, /*seed=*/4U);

    const float beta1_pow = std::pow(hp.beta1, static_cast<float>(step));
    const float beta2_pow = std::pow(hp.beta2, static_cast<float>(step));

    // The op updates exp_avg / exp_avg_sq (/ max_exp_avg_sq) in place, so each path gets its own copies.
    auto keep = [&](std::initializer_list<ttnn::Tensor> tensors) {
        if (keep_alive != nullptr) {
            keep_alive->insert(keep_alive->end(), tensors.begin(), tensors.end());
        }
    };

    auto run_float_path = [&]() {
        auto param = make_device_tensor(param_data, shape, param_dtype);
        auto grad = make_device_tensor(grad_data, shape, ttnn::DataType::BFLOAT16);
        auto exp_avg = make_device_tensor(exp_avg_data, shape, param_dtype);
        auto exp_avg_sq = make_device_tensor(exp_avg_sq_data, shape, param_dtype);
        auto max_exp_avg_sq =
            amsgrad ? std::optional<ttnn::Tensor>(make_device_tensor(max_exp_avg_sq_data, shape, param_dtype))
                    : std::nullopt;
        keep({param, grad, exp_avg, exp_avg_sq});
        if (max_exp_avg_sq.has_value()) {
            keep({*max_exp_avg_sq});
        }
        auto param_out = ttml::metal::adamw(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            hp.lr,
            hp.beta1,
            hp.beta2,
            beta1_pow,
            beta2_pow,
            hp.epsilon,
            hp.weight_decay);
        auto max_exp_avg_sq_out =
            amsgrad ? max_exp_avg_sq->to_vector<float>() : std::vector<float>{};
        return std::make_tuple(
            param_out.to_vector<float>(),
            exp_avg.to_vector<float>(),
            exp_avg_sq.to_vector<float>(),
            std::move(max_exp_avg_sq_out));
    };

    auto run_tensor_path = [&]() {
        auto param = make_device_tensor(param_data, shape, param_dtype);
        auto grad = make_device_tensor(grad_data, shape, ttnn::DataType::BFLOAT16);
        auto exp_avg = make_device_tensor(exp_avg_data, shape, param_dtype);
        auto exp_avg_sq = make_device_tensor(exp_avg_sq_data, shape, param_dtype);
        auto max_exp_avg_sq =
            amsgrad ? std::optional<ttnn::Tensor>(make_device_tensor(max_exp_avg_sq_data, shape, param_dtype))
                    : std::nullopt;
        auto step_size = make_scalar_tensor(hp.lr / (1.0F - beta1_pow));
        auto inv_sqrt_bc2 = make_scalar_tensor(1.0F / std::sqrt(1.0F - beta2_pow));
        auto decay_factor = make_scalar_tensor(1.0F - hp.lr * hp.weight_decay);
        keep({param, grad, exp_avg, exp_avg_sq, step_size, inv_sqrt_bc2, decay_factor});
        if (max_exp_avg_sq.has_value()) {
            keep({*max_exp_avg_sq});
        }
        auto param_out = ttml::metal::adamw(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            step_size,
            inv_sqrt_bc2,
            decay_factor,
            hp.beta1,
            hp.beta2,
            hp.epsilon);
        auto max_exp_avg_sq_out =
            amsgrad ? max_exp_avg_sq->to_vector<float>() : std::vector<float>{};
        return std::make_tuple(
            param_out.to_vector<float>(),
            exp_avg.to_vector<float>(),
            exp_avg_sq.to_vector<float>(),
            std::move(max_exp_avg_sq_out));
    };

    const auto [param_ref, exp_avg_ref, exp_avg_sq_ref, max_exp_avg_sq_ref] = run_float_path();
    const auto [param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out] = run_tensor_path();

    // The tensor-scalar path feeds the kernel the same three immediates, so results are bit-identical.
    EXPECT_EQ(param_ref, param_out);
    EXPECT_EQ(exp_avg_ref, exp_avg_out);
    EXPECT_EQ(exp_avg_sq_ref, exp_avg_sq_out);
    EXPECT_EQ(max_exp_avg_sq_ref, max_exp_avg_sq_out);
}

void run_parity_check(ttnn::DataType param_dtype, bool amsgrad) {
    // The first step compiles and caches the program; the second hits the program
    // cache with freshly allocated scalar tensors holding different values, so it
    // exercises override_runtime_arguments updating the scalar-tensor addresses.
    // Step 10's tensors are kept alive across step 11's allocations so step 11's
    // tensors land at genuinely different addresses -- otherwise the allocator
    // would hand back the freed addresses and a broken override_runtime_arguments
    // (reading new values from old addresses) would still pass.
    std::vector<ttnn::Tensor> step10_tensors;
    run_parity_step(param_dtype, amsgrad, /*step=*/10U, &step10_tensors);
    run_parity_step(param_dtype, amsgrad, /*step=*/11U);
}

}  // namespace

TEST_F(AdamWTensorScalarsTest, MatchesFloatScalars_BF16) {
    run_parity_check(ttnn::DataType::BFLOAT16, /*amsgrad=*/false);
}

TEST_F(AdamWTensorScalarsTest, MatchesFloatScalars_BF16_AmsGrad) {
    run_parity_check(ttnn::DataType::BFLOAT16, /*amsgrad=*/true);
}

TEST_F(AdamWTensorScalarsTest, MatchesFloatScalars_FP32) {
    run_parity_check(ttnn::DataType::FLOAT32, /*amsgrad=*/false);
}

TEST_F(AdamWTensorScalarsTest, MatchesFloatScalars_FP32_AmsGrad) {
    run_parity_check(ttnn::DataType::FLOAT32, /*amsgrad=*/true);
}

TEST_F(AdamWTensorScalarsTest, RejectsMultiElementScalarTensor) {
    const ttnn::Shape shape({1, 1, 32, 32});
    const auto data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, /*seed=*/0U);
    auto param = make_device_tensor(data, shape);
    auto grad = make_device_tensor(data, shape);
    auto exp_avg = make_device_tensor(data, shape);
    auto exp_avg_sq = make_device_tensor(data, shape);

    auto bad_step_size = make_device_tensor({1.0F, 2.0F}, ttnn::Shape({1, 1, 1, 2}), ttnn::DataType::FLOAT32);
    auto inv_sqrt_bc2 = make_scalar_tensor(1.0F);
    auto decay_factor = make_scalar_tensor(1.0F);
    EXPECT_ANY_THROW(ttml::metal::adamw(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        std::nullopt,
        bad_step_size,
        inv_sqrt_bc2,
        decay_factor,
        0.9F,
        0.999F,
        1e-8F));
}

TEST_F(AdamWTensorScalarsTest, RejectsNonFloat32ScalarTensor) {
    const ttnn::Shape shape({1, 1, 32, 32});
    const auto data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, /*seed=*/0U);
    auto param = make_device_tensor(data, shape);
    auto grad = make_device_tensor(data, shape);
    auto exp_avg = make_device_tensor(data, shape);
    auto exp_avg_sq = make_device_tensor(data, shape);

    auto bad_step_size = make_scalar_tensor(1.0F, ttnn::DataType::BFLOAT16);
    auto inv_sqrt_bc2 = make_scalar_tensor(1.0F);
    auto decay_factor = make_scalar_tensor(1.0F);
    EXPECT_ANY_THROW(ttml::metal::adamw(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        std::nullopt,
        bad_step_size,
        inv_sqrt_bc2,
        decay_factor,
        0.9F,
        0.999F,
        1e-8F));
}
