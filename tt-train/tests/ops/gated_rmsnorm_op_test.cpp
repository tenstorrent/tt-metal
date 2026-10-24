// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.hpp"
#include "ops/rmsnorm_op.hpp"
#include "test_utils/random_data.hpp"

namespace {

constexpr float kEps = 1e-6F;
// Max error over max |expected|. bf16 rounding alone is up to 2^-9 of the peak; 1e-2 is ~5x that.
constexpr double kTol = 1e-2;

// x and dx are head-major [B*H, 1, T, V]; gate, dL_dout, dgate and the output are head-merged [B, 1, T, H*V].
struct Shape {
    uint32_t batch;
    uint32_t seq;
    uint32_t heads;
    uint32_t head_dim;

    [[nodiscard]] std::vector<uint32_t> head_major() const {
        return {batch * heads, 1U, seq, head_dim};
    }
    [[nodiscard]] std::vector<uint32_t> head_merged() const {
        return {batch, 1U, seq, heads * head_dim};
    }
    [[nodiscard]] std::vector<uint32_t> rms() const {
        return {batch * heads, 1U, seq, 1U};
    }
    [[nodiscard]] size_t head_major_index(
        const uint32_t b, const uint32_t h, const uint32_t t, const uint32_t c) const {
        return ((static_cast<size_t>(b) * heads + h) * seq + t) * head_dim + c;
    }
    [[nodiscard]] size_t head_merged_index(
        const uint32_t b, const uint32_t h, const uint32_t t, const uint32_t c) const {
        return ((static_cast<size_t>(b) * seq + t) * heads + h) * head_dim + c;
    }
};

struct Inputs {
    xt::xarray<float> x;
    xt::xarray<float> gate;
    xt::xarray<float> gamma;
    xt::xarray<float> dy;
    ttnn::DataType x_dtype;
};

struct Reference {
    xt::xarray<double> out;
    xt::xarray<double> rms;
    xt::xarray<double> dx;
    xt::xarray<double> dgate;
    xt::xarray<double> dgamma;
};

ttnn::Tensor to_device(const xt::xarray<float>& a, const ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    auto* device = &ttml::autograd::ctx().get_device();
    return dtype == ttnn::DataType::FLOAT32 ? ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(a, device)
                                            : ttml::core::from_xtensor(a, device);
}

xt::xarray<float> as_stored(const xt::xarray<float>& a, const ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    return ttml::core::to_xtensor(to_device(a, dtype));
}

// Heads get RMS spread over 2^-2 .. 2^3, so normalizing over the wrong columns cannot pass.
Inputs make_inputs(const Shape& s, const uint32_t seed, const ttnn::DataType x_dtype = ttnn::DataType::BFLOAT16) {
    auto x = ttml::test_utils::make_uniform_xarray<float>(s.head_major(), -1.0F, 1.0F, seed);
    for (uint32_t b = 0; b < s.batch; ++b) {
        for (uint32_t h = 0; h < s.heads; ++h) {
            const float scale = std::exp2(static_cast<float>(static_cast<int>(h % 6U) - 2));
            for (uint32_t t = 0; t < s.seq; ++t) {
                for (uint32_t c = 0; c < s.head_dim; ++c) {
                    x.data()[s.head_major_index(b, h, t, c)] *= scale;
                }
            }
        }
    }
    return Inputs{
        .x = as_stored(x, x_dtype),
        .gate = as_stored(ttml::test_utils::make_uniform_xarray<float>(s.head_merged(), -4.0F, 4.0F, seed + 1U)),
        .gamma = as_stored(ttml::test_utils::make_uniform_xarray<float>(
            std::vector<uint32_t>{1U, 1U, 1U, s.head_dim}, 0.5F, 1.5F, seed + 2U)),
        .dy = as_stored(ttml::test_utils::make_uniform_xarray<float>(s.head_merged(), -1.0F, 1.0F, seed + 3U)),
        .x_dtype = x_dtype,
    };
}

double sigmoid(const double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

Reference reference(const Inputs& in, const xt::xarray<float>& dy, const Shape& s) {
    Reference ref{
        .out = xt::zeros<double>(s.head_merged()),
        .rms = xt::zeros<double>(s.rms()),
        .dx = xt::zeros<double>(s.head_major()),
        .dgate = xt::zeros<double>(s.head_merged()),
        .dgamma = xt::zeros<double>(in.gamma.shape()),
    };
    const uint32_t V = s.head_dim;
    std::vector<double> u(V);
    std::vector<double> du(V);
    for (uint32_t b = 0; b < s.batch; ++b) {
        for (uint32_t h = 0; h < s.heads; ++h) {
            for (uint32_t t = 0; t < s.seq; ++t) {
                double sum_sq = 0.0;
                for (uint32_t c = 0; c < V; ++c) {
                    const double x = in.x.data()[s.head_major_index(b, h, t, c)];
                    sum_sq += x * x;
                }
                const double rms = std::sqrt(sum_sq / V + kEps);
                const double inv = 1.0 / rms;
                ref.rms.data()[(static_cast<size_t>(b) * s.heads + h) * s.seq + t] = rms;

                double dot = 0.0;
                for (uint32_t c = 0; c < V; ++c) {
                    const size_t merged = s.head_merged_index(b, h, t, c);
                    const double z = in.gate.data()[merged];
                    u[c] = in.x.data()[s.head_major_index(b, h, t, c)] * in.gamma.data()[c] * inv;
                    du[c] = dy.data()[merged] * z * sigmoid(z);
                    dot += u[c] * du[c];
                    ref.out.data()[merged] = u[c] * z * sigmoid(z);
                }
                const double scale = dot * inv / V;
                for (uint32_t c = 0; c < V; ++c) {
                    const size_t major = s.head_major_index(b, h, t, c);
                    const size_t merged = s.head_merged_index(b, h, t, c);
                    const double x = in.x.data()[major];
                    const double z = in.gate.data()[merged];
                    const double sg = sigmoid(z);
                    ref.dx.data()[major] = inv * (in.gamma.data()[c] * du[c] - scale * x);
                    ref.dgate.data()[merged] = dy.data()[merged] * u[c] * sg * (1.0 + z - z * sg);
                    ref.dgamma.data()[c] += x * inv * du[c];
                }
            }
        }
    }
    return ref;
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

std::string label(const std::string& what, const Shape& s, const ttnn::DataType dtype) {
    return what + " " + std::to_string(s.batch) + "x" + std::to_string(s.seq) + "x" + std::to_string(s.heads) + "x" +
           std::to_string(s.head_dim) + (dtype == ttnn::DataType::FLOAT32 ? " fp32" : " bf16");
}

}  // namespace

class GatedRmsNormTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

TEST_F(GatedRmsNormTest, MatchesReference) {
    for (const auto dtype : {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}) {
        // A single head may be any width; V = 80 also covers the partial last tile.
        for (const auto& s : {Shape{1, 64, 3, 128}, Shape{2, 96, 4, 64}, Shape{1, 256, 12, 128}, Shape{2, 64, 1, 80}}) {
            const auto in = make_inputs(s, 11U, dtype);
            const auto ref = reference(in, in.dy, s);
            const auto x = to_device(in.x, dtype);
            const auto gate = to_device(in.gate);
            const auto gamma = to_device(in.gamma);

            const auto fw = ttml::metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEps, gate);
            expect_close(ttml::core::to_xtensor(fw[0].value()), ref.out, label("out", s, dtype));
            expect_close(ttml::core::to_xtensor(fw[1].value()), ref.rms, label("rms", s, dtype));

            const auto bw =
                ttml::metal::rmsnorm_bw(x, gamma, fw[1].value(), to_device(in.dy), /* compute_dgamma */ true, gate);
            expect_close(ttml::core::to_xtensor(bw[0].value()), ref.dx, label("dx", s, dtype));
            expect_close(ttml::core::to_xtensor(bw[1].value()), ref.dgamma, label("dgamma", s, dtype));
            expect_close(ttml::core::to_xtensor(bw[2].value()), ref.dgate, label("dgate", s, dtype));
        }
    }
}

TEST_F(GatedRmsNormTest, SkipsDgammaWhenNotRequested) {
    const Shape s{2, 96, 4, 64};
    const auto in = make_inputs(s, 23U);
    const auto ref = reference(in, in.dy, s);
    const auto x = to_device(in.x);
    const auto gate = to_device(in.gate);
    const auto gamma = to_device(in.gamma);

    const auto fw = ttml::metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEps, gate);
    const auto bw =
        ttml::metal::rmsnorm_bw(x, gamma, fw[1].value(), to_device(in.dy), /* compute_dgamma */ false, gate);
    ASSERT_EQ(bw.size(), 3U);
    EXPECT_FALSE(bw[1].has_value());
    expect_close(ttml::core::to_xtensor(bw[0].value()), ref.dx, "dx without dgamma");
    expect_close(ttml::core::to_xtensor(bw[2].value()), ref.dgate, "dgate without dgamma");
}

TEST_F(GatedRmsNormTest, WritesPreallocatedOutputs) {
    const Shape s{1, 64, 3, 128};
    const auto in = make_inputs(s, 31U, ttnn::DataType::FLOAT32);
    const auto ref = reference(in, in.dy, s);
    auto* device = &ttml::autograd::ctx().get_device();
    const auto x = to_device(in.x, ttnn::DataType::FLOAT32);
    const auto gate = to_device(in.gate);
    const auto gamma = to_device(in.gamma);

    const auto out = ttml::core::zeros(ttnn::Shape(s.head_merged()), device);
    const auto rms = ttml::core::zeros(ttnn::Shape(s.rms()), device);
    const auto fw = ttnn::prim::ttml_rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEps, rms, out, gate);
    EXPECT_EQ(fw[0].buffer()->address(), out.buffer()->address());
    EXPECT_EQ(fw[1].buffer()->address(), rms.buffer()->address());
    expect_close(ttml::core::to_xtensor(out), ref.out, "preallocated out");
    expect_close(ttml::core::to_xtensor(rms), ref.rms, "preallocated rms");

    const auto da = ttml::core::zeros(ttnn::Shape(s.head_major()), device);
    const auto dgamma_components = ttml::core::zeros(ttnn::Shape(s.head_major()), device);
    const auto dgate = ttml::core::zeros(ttnn::Shape(s.head_merged()), device);
    const auto bw = ttnn::prim::ttml_rmsnorm_bw(
        x, gamma, rms, to_device(in.dy), kEps, da, dgamma_components, /* compute_dgamma */ true, gate, dgate);
    EXPECT_EQ(bw[0]->buffer()->address(), da.buffer()->address());
    EXPECT_EQ(bw[1]->buffer()->address(), dgamma_components.buffer()->address());
    EXPECT_EQ(bw[2]->buffer()->address(), dgate.buffer()->address());
    expect_close(ttml::core::to_xtensor(da), ref.dx, "preallocated dx");
    expect_close(ttml::core::to_xtensor(dgate), ref.dgate, "preallocated dgate");
    const xt::xarray<double> components = xt::cast<double>(ttml::core::to_xtensor(dgamma_components));
    const xt::xarray<double> dgamma = xt::sum(components, {0, 1, 2}, xt::keep_dims);
    expect_close(dgamma, ref.dgamma, "preallocated dgamma");
}

TEST_F(GatedRmsNormTest, AutogradSkipsFrozenGamma) {
    using namespace ttml;
    for (const bool train_gamma : {true, false}) {
        const Shape s{1, 64, 3, 128};
        const auto in = make_inputs(s, 37U, ttnn::DataType::FLOAT32);
        auto x = autograd::create_tensor(to_device(in.x, ttnn::DataType::FLOAT32), /* requires_grad */ true);
        auto gate = autograd::create_tensor(to_device(in.gate), /* requires_grad */ true);
        auto gamma = autograd::create_tensor(to_device(in.gamma), /* requires_grad */ train_gamma);

        auto out = ops::gated_rmsnorm(x, gate, gamma, kEps);
        out->backward();

        const xt::xarray<float> ones = xt::ones<float>(s.head_merged());  // backward() seeds dL/dout with ones
        const auto ref = reference(in, ones, s);
        expect_close(core::to_xtensor(out->get_value()), ref.out, "autograd out");
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

TEST_F(GatedRmsNormTest, RejectsMismatchedBackwardTensors) {
    const Shape s{1, 64, 3, 128};
    const auto in = make_inputs(s, 43U);
    auto* device = &ttml::autograd::ctx().get_device();
    const auto x = to_device(in.x);
    const auto gate = to_device(in.gate);
    const auto gamma = to_device(in.gamma);
    const auto rms = ttml::metal::rmsnorm_fw(x, gamma, /* return_intermediates */ true, kEps, gate)[1].value();

    // In gated mode dL_dout follows the gate's layout, not the input's.
    EXPECT_ANY_THROW(ttml::metal::rmsnorm_bw(x, gamma, rms, x, /* compute_dgamma */ true, gate));
    // dL_dgate only exists in gated mode.
    EXPECT_ANY_THROW(ttnn::prim::ttml_rmsnorm_bw(
        x,
        gamma,
        rms,
        x,
        kEps,
        std::nullopt,
        std::nullopt,
        /* compute_dgamma */ true,
        /* gate */ std::nullopt,
        ttml::core::zeros(ttnn::Shape(s.head_merged()), device)));
    // No dL_dgamma_components are written without compute_dgamma.
    EXPECT_ANY_THROW(ttnn::prim::ttml_rmsnorm_bw(
        x,
        gamma,
        rms,
        to_device(in.dy),
        kEps,
        std::nullopt,
        ttml::core::zeros(ttnn::Shape(s.head_major()), device),
        /* compute_dgamma */ false,
        gate));
}
