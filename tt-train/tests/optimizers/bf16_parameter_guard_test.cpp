// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/muon_composite.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"

using namespace ttml;

// Every optimizer except fused AdamW only has a bf16 update. Given an fp32 parameter, they must fail at
// construction instead of silently training a bf16 copy (fused SGD) or turning the parameter into bf16 through
// set_value (AdamWFullPrecision and the composites).
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

TEST_F(Bf16ParameterGuardTest, AdamWFullPrecision) {
    expect_rejects_fp32_and_accepts_bf16<optimizers::AdamWFullPrecision>(optimizers::AdamWFullPrecisionConfig{});
}

TEST_F(Bf16ParameterGuardTest, FusedAdamWAcceptsFp32Parameters) {
    EXPECT_NO_THROW({ optimizers::AdamW optimizer(parameters(ttnn::DataType::FLOAT32), optimizers::AdamWConfig{}); });
}

TEST_F(Bf16ParameterGuardTest, MessageNamesTheParameterAndTheOptimizer) {
    try {
        optimizers::SGD optimizer(parameters(ttnn::DataType::FLOAT32), optimizers::SGDConfig{});
        FAIL() << "SGD accepted an fp32 parameter";
    } catch (const std::exception &e) {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("SGD supports bf16 parameters only"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("'theta'"));
    }
}
