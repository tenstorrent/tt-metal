// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/gated_rmsnorm_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/gated_rmsnorm/gated_rmsnorm.hpp"
#include "test_utils/random_data.hpp"

namespace {

constexpr float kEps = 1e-6F;
// Max error over max |expected|. bf16 rounding alone is up to 2^-9 of the peak; 1e-2 is ~5x that.
constexpr double kTol = 1e-2;

struct Shape {
    uint32_t batch;
    uint32_t seq;
    uint32_t heads;
    uint32_t group;

    [[nodiscard]] uint32_t width() const {
        return heads * group;
    }
    [[nodiscard]] uint32_t rows() const {
        return batch * seq;
    }
};

struct Inputs {
    xt::xarray<float> x;
    xt::xarray<float> gate;
    xt::xarray<float> gamma;
    xt::xarray<float> dy;
};

struct Grads {
    xt::xarray<double> dx;
    xt::xarray<double> dgate;
    xt::xarray<double> dgamma;
};

ttnn::Tensor to_device(const xt::xarray<float>& a) {
    return ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
}

xt::xarray<float> as_stored(const xt::xarray<float>& a) {
    return ttml::core::to_xtensor(to_device(a));
}

// Heads get RMS spread over 2^-2 .. 2^3, so normalizing over the wrong columns cannot pass.
Inputs make_inputs(const Shape& s, const uint32_t seed) {
    const std::vector<uint32_t> act_shape{s.batch, 1U, s.seq, s.width()};
    auto x = ttml::test_utils::make_uniform_xarray<float>(act_shape, -1.0F, 1.0F, seed);
    for (uint32_t r = 0; r < s.rows(); ++r) {
        for (uint32_t c = 0; c < s.width(); ++c) {
            x.data()[r * s.width() + c] *= std::exp2(static_cast<float>(static_cast<int>((c / s.group) % 6U) - 2));
        }
    }
    return Inputs{
        .x = as_stored(x),
        .gate = as_stored(ttml::test_utils::make_uniform_xarray<float>(act_shape, -4.0F, 4.0F, seed + 1U)),
        .gamma = as_stored(ttml::test_utils::make_uniform_xarray<float>(
            std::vector<uint32_t>{1U, 1U, 1U, s.group}, 0.5F, 1.5F, seed + 2U)),
        .dy = as_stored(ttml::test_utils::make_uniform_xarray<float>(act_shape, -1.0F, 1.0F, seed + 3U)),
    };
}

double sigmoid(const double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// inv for row r, head h.
double inv_rms(const Inputs& in, const Shape& s, const uint32_t r, const uint32_t h) {
    double sum_sq = 0.0;
    for (uint32_t c = 0; c < s.group; ++c) {
        const double v = in.x.data()[r * s.width() + h * s.group + c];
        sum_sq += v * v;
    }
    return 1.0 / std::sqrt(sum_sq / s.group + kEps);
}

xt::xarray<double> reference_forward(const Inputs& in, const Shape& s) {
    xt::xarray<double> out = xt::zeros<double>(in.x.shape());
    for (uint32_t r = 0; r < s.rows(); ++r) {
        for (uint32_t h = 0; h < s.heads; ++h) {
            const double inv = inv_rms(in, s, r, h);
            for (uint32_t c = 0; c < s.group; ++c) {
                const uint32_t i = r * s.width() + h * s.group + c;
                const double z = in.gate.data()[i];
                out.data()[i] = in.x.data()[i] * in.gamma.data()[c] * inv * z * sigmoid(z);
            }
        }
    }
    return out;
}

Grads reference_backward(const Inputs& in, const xt::xarray<float>& dy, const Shape& s) {
    Grads g{
        .dx = xt::zeros<double>(in.x.shape()),
        .dgate = xt::zeros<double>(in.x.shape()),
        .dgamma = xt::zeros<double>(in.gamma.shape()),
    };
    std::vector<double> u(s.group);
    std::vector<double> du(s.group);
    for (uint32_t r = 0; r < s.rows(); ++r) {
        for (uint32_t h = 0; h < s.heads; ++h) {
            const uint32_t base = r * s.width() + h * s.group;
            const double inv = inv_rms(in, s, r, h);
            double dot = 0.0;
            for (uint32_t c = 0; c < s.group; ++c) {
                const double z = in.gate.data()[base + c];
                u[c] = in.x.data()[base + c] * in.gamma.data()[c] * inv;
                du[c] = dy.data()[base + c] * z * sigmoid(z);
                dot += u[c] * du[c];
            }
            const double t = dot * inv / s.group;
            for (uint32_t c = 0; c < s.group; ++c) {
                const double x = in.x.data()[base + c];
                const double z = in.gate.data()[base + c];
                const double sg = sigmoid(z);
                g.dx.data()[base + c] = inv * (in.gamma.data()[c] * du[c] - t * x);
                g.dgate.data()[base + c] = dy.data()[base + c] * u[c] * sg * (1.0 + z - z * sg);
                g.dgamma.data()[c] += x * inv * du[c];
            }
        }
    }
    return g;
}

template <typename Got>
void expect_close(const Got& got, const xt::xarray<double>& expected, const std::string& label) {
    ASSERT_EQ(got.size(), expected.size()) << label;
    double peak = 0.0;
    double max_err = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        peak = std::max(peak, std::abs(expected.data()[i]));
        max_err = std::max(max_err, std::abs(static_cast<double>(got.data()[i]) - expected.data()[i]));
    }
    EXPECT_LE(max_err, kTol * peak) << label << ": max error " << max_err << " vs peak " << peak;
}

}  // namespace

class GatedRmsNormOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(GatedRmsNormOpTest, ForwardMatchesReference) {
    for (const auto& s : {Shape{1, 64, 3, 128}, Shape{2, 96, 4, 64}, Shape{1, 256, 12, 128}}) {
        const auto in = make_inputs(s, 11U);
        const auto out = ttml::metal::gated_rmsnorm_fw(to_device(in.x), to_device(in.gate), to_device(in.gamma), kEps);
        expect_close(ttml::core::to_xtensor(out), reference_forward(in, s), "fw");
    }
}

TEST_F(GatedRmsNormOpTest, BackwardMatchesReference) {
    for (const bool compute_dgamma : {true, false}) {
        const Shape s{2, 96, 4, 64};
        const auto in = make_inputs(s, 23U);
        const auto [dx, dgate, dgamma] = ttml::metal::gated_rmsnorm_bw(
            to_device(in.x), to_device(in.gate), to_device(in.gamma), to_device(in.dy), kEps, compute_dgamma);
        const auto ref = reference_backward(in, in.dy, s);
        expect_close(ttml::core::to_xtensor(dx), ref.dx, "dx");
        expect_close(ttml::core::to_xtensor(dgate), ref.dgate, "dgate");
        ASSERT_EQ(dgamma.has_value(), compute_dgamma);
        if (compute_dgamma) {
            expect_close(ttml::core::to_xtensor(dgamma.value()), ref.dgamma, "dgamma");
        }
    }
}

TEST_F(GatedRmsNormOpTest, AutogradSkipsFrozenGamma) {
    using namespace ttml;
    for (const bool train_gamma : {true, false}) {
        const Shape s{1, 64, 3, 128};
        const auto in = make_inputs(s, 37U);
        auto x = autograd::create_tensor(to_device(in.x), /* requires_grad */ true);
        auto gate = autograd::create_tensor(to_device(in.gate), /* requires_grad */ true);
        auto gamma = autograd::create_tensor(to_device(in.gamma), /* requires_grad */ train_gamma);

        auto out = ops::gated_rmsnorm(x, gate, gamma, kEps);
        expect_close(core::to_xtensor(out->get_value()), reference_forward(in, s), "autograd fw");

        out->backward();

        const xt::xarray<float> ones = xt::ones<float>(in.x.shape());  // backward() seeds dL/dout with ones
        const auto ref = reference_backward(in, ones, s);
        ASSERT_TRUE(x->is_grad_initialized());
        ASSERT_TRUE(gate->is_grad_initialized());
        expect_close(core::to_xtensor(x->get_grad()), ref.dx, "autograd dx");
        expect_close(core::to_xtensor(gate->get_grad()), ref.dgate, "autograd dgate");
        ASSERT_EQ(gamma->is_grad_initialized(), train_gamma);
        if (train_gamma) {
            expect_close(core::to_xtensor(gamma->get_grad()), ref.dgamma, "autograd dgamma");
        }
        autograd::ctx().reset_graph();
    }
}
