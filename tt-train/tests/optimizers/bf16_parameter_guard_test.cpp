// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/muon_composite.hpp"
#include "optimizers/no_op.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

using namespace ttml;

// Every optimizer except fused AdamW only has a bf16 update. Given an fp32 parameter, they must fail at
// construction instead of silently training a bf16 copy (fused SGD) or turning the parameter into bf16 through
// set_value (the composites).
class Bf16ParameterGuardTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        autograd::ctx().close_device();
    }

protected:
    static serialization::NamedParameters parameters(ttnn::DataType dtype) {
        auto *device = &autograd::ctx().get_device();
        auto theta = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 32, 32}), device, dtype), true);
        return {{"theta", theta}};
    }

    template <typename Optimizer, typename Config>
    static void expect_rejects_fp32_and_accepts_bf16(const Config &config) {
        EXPECT_ANY_THROW({ Optimizer optimizer(parameters(ttnn::DataType::FLOAT32), config); });
        EXPECT_NO_THROW({ Optimizer optimizer(parameters(ttnn::DataType::BFLOAT16), config); });
    }
};

TEST_F(Bf16ParameterGuardTest, Sgd) {
    expect_rejects_fp32_and_accepts_bf16<optimizers::SGD>(optimizers::SGDConfig{});
}

TEST_F(Bf16ParameterGuardTest, SgdComposite) {
    expect_rejects_fp32_and_accepts_bf16<optimizers::SGDComposite>(optimizers::SGDCompositeConfig{});
}

TEST_F(Bf16ParameterGuardTest, MorehAdamW) {
    expect_rejects_fp32_and_accepts_bf16<optimizers::MorehAdamW>(optimizers::AdamWCompositeConfig{});
}

TEST_F(Bf16ParameterGuardTest, AdamWComposite) {
    expect_rejects_fp32_and_accepts_bf16<optimizers::AdamWComposite>(optimizers::AdamWCompositeConfig{});
}

TEST_F(Bf16ParameterGuardTest, MuonComposite) {
    expect_rejects_fp32_and_accepts_bf16<optimizers::MuonComposite>(optimizers::MuonConfig{});
}

TEST_F(Bf16ParameterGuardTest, FusedAdamWAcceptsFp32Parameters) {
    EXPECT_NO_THROW({ optimizers::AdamW optimizer(parameters(ttnn::DataType::FLOAT32), optimizers::AdamWConfig{}); });
}

TEST_F(Bf16ParameterGuardTest, MessageNamesTheParameterAndTheOptimizer) {
    EXPECT_THAT(
        [] { optimizers::SGD optimizer(parameters(ttnn::DataType::FLOAT32), optimizers::SGDConfig{}); },
        ::testing::ThrowsMessage<std::exception>(::testing::AllOf(
            ::testing::HasSubstr("SGD supports bf16 parameters only"), ::testing::HasSubstr("'theta'"))));
}

// With reduce_dtype float32 a parameter's gradient can reach these optimizers in fp32. Their update takes bf16
// gradients, so each rounds an fp32 gradient to bf16 once at the start of its step: the step must be the one it takes
// for the same gradient cast to bf16 beforehand, bit for bit, and the gradient left behind is that bf16 cast.
class Fp32GradientIntoBf16OptimizerTest : public test_utils::DeviceSuiteTest {
protected:
    static constexpr std::array<uint32_t, 4> kShape = {1U, 1U, 64U, 128U};

    // A bf16 parameter, the same values in every call, whose gradient is `grad`.
    static serialization::NamedParameters parameter_with_grad(const ttnn::Tensor &grad) {
        auto *device = &autograd::ctx().get_device();
        const auto values = test_utils::make_uniform_xarray<float>(kShape, -1.0F, 1.0F, 11U);
        auto theta = autograd::create_tensor(core::from_xtensor(values, device), true);
        theta->set_grad(grad);
        return {{"theta", theta}};
    }

    template <typename Optimizer, typename Config>
    static void expect_the_step_of_the_bf16_cast(const Config &config) {
        auto *device = &autograd::ctx().get_device();
        const auto values = test_utils::make_uniform_xarray<float>(kShape, -1.0F, 1.0F, 12U);
        const auto fp32_grad = core::from_xtensor<float, ttnn::DataType::FLOAT32>(values, device);
        const auto bf16_grad = ttnn::typecast(fp32_grad, ttnn::DataType::BFLOAT16);
        // Random fp32 values: the cast changes them, so a step that read the fp32 gradient would differ.
        ASSERT_GT(test_utils::count_bit_mismatches(values, core::to_xtensor(bf16_grad)), 0U);

        const auto fp32_parameters = parameter_with_grad(fp32_grad);
        const auto bf16_parameters = parameter_with_grad(bf16_grad);
        const auto before = core::to_xtensor(bf16_parameters.at("theta")->get_value());
        Optimizer fp32_optimizer(fp32_parameters, config);
        Optimizer bf16_optimizer(bf16_parameters, config);
        fp32_optimizer.step();
        bf16_optimizer.step();

        const auto &fp32_theta = fp32_parameters.at("theta");
        const auto &bf16_theta = bf16_parameters.at("theta");
        const auto expected = core::to_xtensor(bf16_theta->get_value());
        EXPECT_GT(test_utils::count_bit_mismatches(before, expected), 0U) << "the step must move the parameter";
        EXPECT_EQ(test_utils::count_bit_mismatches(expected, core::to_xtensor(fp32_theta->get_value())), 0U);
        ASSERT_EQ(fp32_theta->get_grad().dtype(), ttnn::DataType::BFLOAT16);
        EXPECT_EQ(
            test_utils::count_bit_mismatches(
                core::to_xtensor(bf16_theta->get_grad()), core::to_xtensor(fp32_theta->get_grad())),
            0U);
    }
};

TEST_F(Fp32GradientIntoBf16OptimizerTest, Sgd) {
    expect_the_step_of_the_bf16_cast<optimizers::SGD>(optimizers::SGDConfig{.lr = 0.1F, .momentum = 0.9F});
}

TEST_F(Fp32GradientIntoBf16OptimizerTest, SgdComposite) {
    expect_the_step_of_the_bf16_cast<optimizers::SGDComposite>(
        optimizers::SGDCompositeConfig{.lr = 0.1F, .momentum = 0.9F});
}

TEST_F(Fp32GradientIntoBf16OptimizerTest, MorehAdamW) {
    expect_the_step_of_the_bf16_cast<optimizers::MorehAdamW>(optimizers::AdamWCompositeConfig{.lr = 0.05F});
}

TEST_F(Fp32GradientIntoBf16OptimizerTest, AdamWComposite) {
    expect_the_step_of_the_bf16_cast<optimizers::AdamWComposite>(optimizers::AdamWCompositeConfig{.lr = 0.05F});
}

TEST_F(Fp32GradientIntoBf16OptimizerTest, MuonComposite) {
    expect_the_step_of_the_bf16_cast<optimizers::MuonComposite>(optimizers::MuonConfig{.lr = 0.05F});
}

// zero_grad keeps a zero gradient on each parameter that had one: in fp32 for a parameter that accumulates its gradient
// in fp32, so the next add needs no cast of the zeros, and in the value's dtype otherwise, as before.
TEST_F(Fp32GradientIntoBf16OptimizerTest, ZeroGradInstallsZerosInTheAccumulationDtype) {
    using Factory = std::function<std::unique_ptr<optimizers::OptimizerBase>(const serialization::NamedParameters &)>;
    const std::array<std::pair<const char *, Factory>, 4> optimizers = {{
        {"SGD", [](const auto &p) { return std::make_unique<optimizers::SGD>(p, optimizers::SGDConfig{}); }},
        {"SGDComposite",
         [](const auto &p) { return std::make_unique<optimizers::SGDComposite>(p, optimizers::SGDCompositeConfig{}); }},
        {"MuonComposite",
         [](const auto &p) { return std::make_unique<optimizers::MuonComposite>(p, optimizers::MuonConfig{}); }},
        {"NoOp", [](const auto &p) { return std::make_unique<optimizers::NoOp>(p); }},
    }};
    auto *device = &autograd::ctx().get_device();
    const auto grad = core::from_xtensor(test_utils::make_uniform_xarray<float>(kShape, -1.0F, 1.0F, 13U), device);
    for (const auto &[name, make_optimizer] : optimizers) {
        for (const bool fp32 : {true, false}) {
            SCOPED_TRACE(fmt::format("{}, fp32 accumulation {}", name, fp32));
            const auto parameters = parameter_with_grad(grad);
            parameters.at("theta")->set_accumulate_grad_in_fp32(fp32);
            make_optimizer(parameters)->zero_grad();
            const auto &zeros = parameters.at("theta")->get_grad();
            EXPECT_EQ(zeros.dtype(), fp32 ? ttnn::DataType::FLOAT32 : ttnn::DataType::BFLOAT16);
            EXPECT_EQ(core::to_vector(zeros), std::vector<float>(zeros.logical_volume(), 0.0F));
        }
    }
}
