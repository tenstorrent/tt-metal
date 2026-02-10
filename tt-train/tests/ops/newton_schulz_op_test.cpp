// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/newton_schulz_op.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

namespace {

xt::xarray<float> newtonschulz_xtensor(const xt::xarray<float>& G, int steps, float eps, float a, float b, float c) {
    auto X = G;

    auto last_2d = xt::view(X, 0, 0, xt::all(), xt::all());
    auto squares = last_2d * last_2d;
    auto sum_squares = xt::sum(squares)();
    auto norm = std::sqrt(sum_squares);

    X = X / (norm + eps);

    auto shape = X.shape();
    size_t m = shape[2];
    size_t n = shape[3];
    bool needs_transpose = (m > n);

    if (needs_transpose) {
        X = xt::transpose(X, {0, 1, 3, 2});
    }

    for (int iter = 0; iter < steps; ++iter) {
        auto X_2d = xt::view(X, 0, 0, xt::all(), xt::all());
        auto A = xt::linalg::dot(X_2d, xt::transpose(X_2d));

        auto B = b * A + c * xt::linalg::dot(A, A);

        auto a_X = a * X_2d;
        auto B_X = xt::linalg::dot(B, X_2d);
        auto X_new = a_X + B_X;

        xt::view(X, 0, 0, xt::all(), xt::all()) = X_new;
    }

    if (needs_transpose) {
        X = xt::transpose(X, {0, 1, 3, 2});
    }

    return X;
}

xt::xarray<float> newtonschulz5_xtensor(const xt::xarray<float>& G, int steps = 5, float eps = 1e-7f) {
    return newtonschulz_xtensor(G, steps, eps, 3.4445f, -4.7750f, 2.0315f);
}

}  // namespace

class NewtonSchulzOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
        xt::random::seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(NewtonSchulzOpTest, MuonCoeff) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 32, 128};
    xt::xarray<float> G_data = xt::random::randn<float>(shape, -1.0f, 1.0f);

    auto G_expected = newtonschulz5_xtensor(G_data, 10, 1e-7f);

    auto G_tensor = core::from_xtensor(G_data, &autograd::ctx().get_device());
    auto X_tensor = ops::newtonschulz5(G_tensor, 10, 1e-7f);
    auto X_result = core::to_xtensor(X_tensor);

    EXPECT_TRUE(xt::allclose(X_result, G_expected, 5e-2f, 5e-2f));
}

TEST_F(NewtonSchulzOpTest, OrthogonalityCheck) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 32, 128};
    xt::xarray<float> G_data = xt::random::randn<float>(shape, -1.0f, 1.0f);

    float a = 15.0f / 8.0f;
    float b = -5.0f / 4.0f;
    float c = 3.0f / 8.0f;

    auto G_expected = newtonschulz_xtensor(G_data, 10, 1e-7f, a, b, c);

    auto G_tensor = core::from_xtensor(G_data, &autograd::ctx().get_device());
    auto X_tensor = ops::newtonschulz(G_tensor, 10, 1e-7f, a, b, c);
    auto X_result = core::to_xtensor(X_tensor);

    EXPECT_TRUE(xt::allclose(X_result, G_expected, 5e-2f, 5e-2f));

    auto X_2d = xt::view(X_result, 0, 0, xt::all(), xt::all());
    auto XXT = xt::linalg::dot(X_2d, xt::transpose(X_2d));
    auto I = xt::eye<float>(32);
    auto ortho_error = xt::abs(XXT - I);
    EXPECT_TRUE(xt::allclose(XXT, I, 5e-2f, 5e-2f));
}
