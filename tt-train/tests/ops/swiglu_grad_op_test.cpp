// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/swiglu_grad/swiglu_grad.hpp"

class SwiGLUGradOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

float relative_l2(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    auto diff = a - b;
    float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
    float ref_l2 = std::sqrt(xt::sum(xt::square(b))());
    return diff_l2 / (ref_l2 + 1e-12f);
}

std::pair<xt::xarray<float>, xt::xarray<float>> swiglu_grad_reference(
    const xt::xarray<float>& linear1, const xt::xarray<float>& gate, const xt::xarray<float>& dL_dprod) {
    auto sigmoid = 1.0f / (1.0f + xt::exp(-linear1));
    auto swished = linear1 * sigmoid;

    xt::xarray<float> dL_dgate = swished * dL_dprod;
    auto dL_dswished = gate * dL_dprod;
    auto silu_grad = sigmoid * (1.0f + linear1 * (1.0f - sigmoid));
    xt::xarray<float> dL_dlinear1 = dL_dswished * silu_grad;

    return {dL_dlinear1, dL_dgate};
}

void CompareSwiGLUGradKernel(const std::vector<uint32_t>& shape) {
    using namespace ttml;

    auto& rng = autograd::ctx().get_generator();
    auto* device = &autograd::ctx().get_device();

    xt::xarray<float> linear1_data = xt::empty<float>(shape);
    core::parallel_generate<float>(linear1_data, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, rng());
    xt::xarray<float> gate_data = xt::empty<float>(shape);
    core::parallel_generate<float>(gate_data, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, rng());
    xt::xarray<float> dL_dprod_data = xt::empty<float>(shape);
    core::parallel_generate<float>(dL_dprod_data, []() { return std::normal_distribution<float>(0.0f, 0.1f); }, rng());

    auto [ref_dL_dlinear1, ref_dL_dgate] = swiglu_grad_reference(linear1_data, gate_data, dL_dprod_data);

    auto linear1_tt = core::from_xtensor(linear1_data, device);
    auto gate_tt = core::from_xtensor(gate_data, device);
    auto dL_dprod_tt = core::from_xtensor(dL_dprod_data, device);

    auto result = metal::swiglu_grad(linear1_tt, gate_tt, dL_dprod_tt);
    auto kernel_dL_dlinear1 = core::to_xtensor(result.dL_dlinear1);
    auto kernel_dL_dgate = core::to_xtensor(result.dL_dgate);

    const float tol = 1e-2f;
    float dl1_err = relative_l2(kernel_dL_dlinear1, ref_dL_dlinear1);
    EXPECT_LT(dl1_err, tol) << "swiglu_grad dL_dlinear1 mismatch: rel L2 = " << dl1_err;

    float dg_err = relative_l2(kernel_dL_dgate, ref_dL_dgate);
    EXPECT_LT(dg_err, tol) << "swiglu_grad dL_dgate mismatch: rel L2 = " << dg_err;
}

}  // namespace

// ============================================================================
// Section: swiglu_grad fused kernel unit tests
// ============================================================================
// Compare the fused swiglu_grad device kernel directly against the reference
// (separate xtensor ops) to verify per-element correctness.
// ============================================================================

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_Basic_1x1x32x32) {
    CompareSwiGLUGradKernel({1, 1, 32, 32});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_MultiTile_1x1x32x64) {
    CompareSwiGLUGradKernel({1, 1, 32, 64});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_MultiRow_1x1x64x32) {
    CompareSwiGLUGradKernel({1, 1, 64, 32});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_Batch_8x1x32x32) {
    CompareSwiGLUGradKernel({8, 1, 32, 32});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_Medium_2x1x32x128) {
    CompareSwiGLUGradKernel({2, 1, 32, 128});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_Large_4x1x64x256) {
    CompareSwiGLUGradKernel({4, 1, 64, 256});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_NonAligned_1x1x32x48) {
    CompareSwiGLUGradKernel({1, 1, 32, 48});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_NonAligned_2x1x32x96) {
    CompareSwiGLUGradKernel({2, 1, 32, 96});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_NonAligned_1x1x64x160) {
    CompareSwiGLUGradKernel({1, 1, 64, 160});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_NonAlignedH_1x1x48x64) {
    CompareSwiGLUGradKernel({1, 1, 48, 64});
}

TEST_F(SwiGLUGradOpTest, SwiGLUGrad_NonAlignedH_4x1x96x128) {
    CompareSwiGLUGradKernel({4, 1, 96, 128});
}

TEST_F(SwiGLUGradOpTest, NIGHTLY_SwiGLUGrad_VeryLarge_1x1x1024x1024) {
    CompareSwiGLUGradKernel({1, 1, 1024, 1024});
}
