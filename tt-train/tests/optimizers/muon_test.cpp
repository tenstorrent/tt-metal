// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/muon.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

namespace {

xt::xarray<float> newtonschulz5_cpu(const xt::xarray<float>& G, int steps, float eps) {
    constexpr float a = 3.4445f;
    constexpr float b = -4.7750f;
    constexpr float c = 2.0315f;

    auto X = G;
    auto last_2d = xt::view(X, 0, 0, xt::all(), xt::all());
    auto norm = std::sqrt(xt::sum(last_2d * last_2d)());
    X = X / (norm + eps);

    auto shape = X.shape();
    bool needs_transpose = (shape[2] > shape[3]);
    if (needs_transpose) {
        X = xt::transpose(X, {0, 1, 3, 2});
    }

    for (int iter = 0; iter < steps; ++iter) {
        auto X_2d = xt::view(X, 0, 0, xt::all(), xt::all());
        auto A = xt::linalg::dot(X_2d, xt::transpose(X_2d));
        auto B = b * A + c * xt::linalg::dot(A, A);
        xt::view(X, 0, 0, xt::all(), xt::all()) = a * X_2d + xt::linalg::dot(B, X_2d);
    }

    if (needs_transpose) {
        X = xt::transpose(X, {0, 1, 3, 2});
    }
    return X;
}

xt::xarray<float> muon_step_cpu(
    const xt::xarray<float>& param,
    const xt::xarray<float>& grad,
    xt::xarray<float>& momentum_buffer,
    float lr,
    float momentum,
    int ns_steps,
    size_t step) {
    if (step > 0 && momentum != 0.0f) {
        momentum_buffer = momentum * momentum_buffer + grad;
    } else {
        momentum_buffer = grad;
    }

    auto update = newtonschulz5_cpu(momentum_buffer, ns_steps, 1e-7f);
    return param - lr * update;
}

}  // namespace

struct MuonTestCase {
    std::array<std::size_t, 4> shape;
    float lr;
    float momentum;
    int ns_steps;
    uint32_t steps;
    std::string name;
};

void PrintTo(const MuonTestCase& tc, std::ostream* os) {
    *os << tc.name;
}

class MuonCorrectnessTest : public ::testing::TestWithParam<MuonTestCase> {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
        xt::random::seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(MuonCorrectnessTest, DeviceMatchesCPU) {
    using namespace ttml;
    const auto& tc = GetParam();

    xt::xarray<float> w0 = xt::random::randn<float>(tc.shape, 0.0f, 1.0f);
    xt::xarray<float> g0 = xt::random::randn<float>(tc.shape, 0.0f, 1.0f);

    // CPU reference
    xt::xarray<float> w_cpu = w0;
    xt::xarray<float> mom_cpu = xt::zeros<float>(tc.shape);
    for (uint32_t i = 0; i < tc.steps; ++i) {
        w_cpu = muon_step_cpu(w_cpu, g0, mom_cpu, tc.lr, tc.momentum, tc.ns_steps, i);
    }

    // Device
    auto param = autograd::create_tensor(core::from_xtensor(w0, &autograd::ctx().get_device()), true);
    param->set_grad(core::from_xtensor(g0, &autograd::ctx().get_device()));

    serialization::NamedParameters params{{"theta", param}};
    optimizers::MuonConfig config{.lr = tc.lr, .momentum = tc.momentum, .ns_steps = tc.ns_steps};
    optimizers::Muon optimizer(params, config);

    for (uint32_t i = 0; i < tc.steps; ++i) {
        optimizer.step();
    }

    auto w_device = core::to_xtensor(param->get_value());
    EXPECT_TRUE(xt::allclose(w_device, w_cpu, 1e-2f, 1e-2f));
}

static const MuonTestCase kMuonCases[] = {
    {{1, 1, 32, 128}, 1e-2f, 0.95f, 5, 1, "Wide"},
    {{1, 1, 128, 32}, 1e-2f, 0.95f, 5, 2, "Tall_2_step"},
    {{1, 1, 128, 128}, 1e-2f, 0.95f, 5, 3, "Square_3_step"},
};

INSTANTIATE_TEST_SUITE_P(MuonCorrectness, MuonCorrectnessTest, ::testing::ValuesIn(kMuonCases), [](const auto& info) {
    return info.param.name;
});
