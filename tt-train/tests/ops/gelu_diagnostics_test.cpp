// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops::tests {

class GeluDiagnosticsTest : public ::testing::Test {
protected:
    void SetUp() override { autograd::ctx().open_device(); }
    void TearDown() override { autograd::ctx().close_device(); }
};

// ------------------------- Reference functions ------------------------------

inline float gelu_erf_f(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + std::erff(x * inv_sqrt2));
}

inline float dgelu_erf_f(float x) {
    // Φ(x) + x * φ(x)
    // Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    // φ(x) = (1/sqrt(2π)) * exp(-x^2/2)
    constexpr float inv_sqrt2   = 0.70710678118654752440f;
    constexpr float inv_sqrt2pi = 0.39894228040143267794f;
    float Phi = 0.5f * (1.0f + std::erff(x * inv_sqrt2));
    float phi = inv_sqrt2pi * std::exp(-0.5f * x * x);
    return Phi + x * phi;
}

inline float gelu_tanh_f(float x) {
    // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
    constexpr float k = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float c = 0.044715f;
    float u = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(u));
}

inline float dgelu_tanh_f(float x) {
    // derivative of the tanh-approx gelu
    constexpr float k = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float c = 0.044715f;
    float u = k * (x + c * x * x * x);
    float t = std::tanh(u);
    float sech2 = 1.0f - t * t;                // use identity: sech^2(u) = 1 - tanh^2(u)
    float du_dx = k * (1.0f + 3.0f * c * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
}

inline float gelu_quick_f(float x) {
    // "fast_gelu" variant: x * sigmoid(1.702 * x)
    const float a = 1.702f;
    float ax = a * x;
    float s = 1.0f / (1.0f + std::exp(-ax));
    return x * s;
}

inline float dgelu_quick_f(float x) {
    const float a = 1.702f;
    float ax = a * x;
    float s = 1.0f / (1.0f + std::exp(-ax));
    return s + x * a * (s * (1.0f - s));
}

template <typename F>
inline std::vector<float> apply_ref(const std::vector<float>& xs, F f) {
    std::vector<float> ys;
    ys.reserve(xs.size());
    for (float x : xs) ys.push_back(f(x));
    return ys;
}

inline float linf(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        ADD_FAILURE() << "linf size mismatch: " << a.size() << " vs " << b.size();
        return std::numeric_limits<float>::infinity();
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

struct ClassResult {
    std::string name;   // "tanh", "erf", "quick", or "unknown"
    float e_tanh;
    float e_erf;
    float e_quick;
};

inline ClassResult classify_forward(const std::vector<float>& xs, const std::vector<float>& yf) {
    float e_erf   = linf(yf, apply_ref(xs, gelu_erf_f));
    float e_tanh  = linf(yf, apply_ref(xs, gelu_tanh_f));
    float e_quick = linf(yf, apply_ref(xs, gelu_quick_f));

    const float ABS_TOL = 3e-2f;
    const float MARGIN  = 3e-3f;

    struct Cand { const char* n; float e; };
    Cand cands[3] = {{"tanh", e_tanh}, {"erf", e_erf}, {"quick", e_quick}};
    std::sort(std::begin(cands), std::end(cands), [](const Cand& a, const Cand& b){ return a.e < b.e; });

    std::string name = "unknown";
    if (cands[0].e <= ABS_TOL && (cands[1].e - cands[0].e) >= MARGIN) {
        name = cands[0].n;
    }
    return {name, e_tanh, e_erf, e_quick};
}

inline ClassResult classify_backward(const std::vector<float>& xs, const std::vector<float>& gb) {
    float e_erf   = linf(gb, apply_ref(xs, dgelu_erf_f));
    float e_tanh  = linf(gb, apply_ref(xs, dgelu_tanh_f));
    float e_quick = linf(gb, apply_ref(xs, dgelu_quick_f));

    const float ABS_TOL = 5e-2f;  // slightly looser for derivatives in BF16
    const float MARGIN  = 5e-3f;

    struct Cand { const char* n; float e; };
    Cand cands[3] = {{"tanh", e_tanh}, {"erf", e_erf}, {"quick", e_quick}};
    std::sort(std::begin(cands), std::end(cands), [](const Cand& a, const Cand& b){ return a.e < b.e; });

    std::string name = "unknown";
    if (cands[0].e <= ABS_TOL && (cands[1].e - cands[0].e) >= MARGIN) {
        name = cands[0].n;
    }
    return {name, e_tanh, e_erf, e_quick};
}

// ------------------------- Device helpers -----------------------------------

static inline ttnn::Tensor make_tensor_1x1x1N(const std::vector<float>& v) {
    auto* device = &autograd::ctx().get_device();
    return core::from_vector(v, ttnn::Shape({1, 1, 1, static_cast<uint32_t>(v.size())}), device);
}

static inline std::vector<float> device_forward(const std::vector<float>& xs) {
    auto t = make_tensor_1x1x1N(xs);
    auto y = ttnn::gelu(t); // device forward (uses its internal default)
    return core::to_vector(y);
}

static inline float device_forward_scalar(float x) {
    std::vector<float> xs{ x };
    return device_forward(xs)[0];
}

static inline std::vector<float> autograd_backward_grads(const std::vector<float>& xs) {
    auto t_raw = make_tensor_1x1x1N(xs);
    auto t = autograd::create_tensor(t_raw);

    // Use the library op (fw/bw approximation aligned by default).
    auto y = ttml::ops::gelu(t);

    // Seed upstream gradient with ones and run backward.
    y->set_grad(core::ones_like(y->get_value()));
    y->backward();

    return core::to_vector(t->get_grad());
}

// ------------------------------ Tests ---------------------------------------

TEST_F(GeluDiagnosticsTest, ForwardFingerprintClassify) {
    // Dense sweep across [-6, 6]
    const size_t N = 1025;
    std::vector<float> xs; xs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        float u = static_cast<float>(i) / static_cast<float>(N - 1);
        xs.push_back(-6.0f + 12.0f * u);
    }

    auto yf = device_forward(xs);
    auto fc = classify_forward(xs, yf);

    std::cout << "[GELU Forward] class=" << fc.name
              << "  Linf(erf)="   << fc.e_erf
              << "  Linf(tanh)="  << fc.e_tanh
              << "  Linf(quick)=" << fc.e_quick
              << std::endl;

    float emin = std::min(fc.e_tanh, std::min(fc.e_erf, fc.e_quick));
    EXPECT_LT(emin, 5e-2f);
}

TEST_F(GeluDiagnosticsTest, ForwardMinimumWindow) {
    const float a = -1.10f, b = -0.40f;
    const int   K = 181; // ~0.0039 step
    float best_x = 0.0f;
    float best_y = 1e9f;

    for (int i = 0; i < K; ++i) {
        float u = static_cast<float>(i) / static_cast<float>(K - 1);
        float x = a + (b - a) * u;
        float y = device_forward_scalar(x);
        if (y < best_y) { best_y = y; best_x = x; }
    }

    std::cout << "[GELU Forward] minimum ~ at x=" << best_x << "  f(x)=" << best_y << std::endl;

    EXPECT_LT(best_x, -0.60f);
    EXPECT_GT(best_x, -0.95f);
    EXPECT_LT(best_y, -0.12f);
    EXPECT_GT(best_y, -0.22f);
}

TEST_F(GeluDiagnosticsTest, AsymptoticsSanity) {
    std::vector<float> xs = {-10.f, -5.f, -3.f, 3.f, 5.f, 10.f};
    auto yf = device_forward(xs);

    EXPECT_NEAR(yf[0], 0.0f, 1e-2f);
    EXPECT_NEAR(yf[1], 0.0f, 2e-2f);
    EXPECT_NEAR(yf[4], 5.0f,  5e-2f);
    EXPECT_NEAR(yf[5], 10.0f, 1e-1f);
}

TEST_F(GeluDiagnosticsTest, BackwardMatchesDeviceForwardFiniteDiff) {
    std::vector<float> xs = {
        -3.0f, -2.0f, -1.5f, -1.25f, -1.0f, -0.75f, -0.5f, -0.25f, -0.1f,
         0.0f,  0.1f,  0.25f,  0.5f,   0.75f, 1.0f,  1.25f, 1.5f,  2.0f, 3.0f
    };

    auto gb = autograd_backward_grads(xs);

    const float h = 1.0f / 512.0f;
    std::vector<float> fd; fd.reserve(xs.size());
    for (float x : xs) {
        float yp = device_forward_scalar(x + h);
        float ym = device_forward_scalar(x - h);
        fd.push_back((yp - ym) / (2.0f * h));
    }

    float e = linf(gb, fd);
    std::cout << "[GELU Backward] Linf(autograd vs FD(device-forward))=" << e << std::endl;

    EXPECT_LT(e, 8e-2f) << "Autograd backward seems inconsistent with the forward approximation.";
}

TEST_F(GeluDiagnosticsTest, BackwardClassificationAndMismatchFlag) {
    const std::vector<float> xs = {-3.f,-2.f,-1.5f,-1.0f,-0.75f,-0.5f,-0.1f,0.f,0.1f,0.5f,1.f,2.f,3.f};

    const auto yf = device_forward(xs);
    const auto gb = autograd_backward_grads(xs);

    const auto f = classify_forward(xs, yf);
    const auto g = classify_backward(xs, gb);

    std::cout << "[GELU Classes] forward=" << f.name
              << "  (|.|_∞: erf="   << f.e_erf   << ", tanh=" << f.e_tanh  << ", quick=" << f.e_quick << ")"
              << "  backward=" << g.name
              << "  (|.|_∞: erf="   << g.e_erf   << ", tanh=" << g.e_tanh  << ", quick=" << g.e_quick << ")"
              << std::endl;

    if (f.name != "unknown" && g.name != "unknown") {
        EXPECT_EQ(f.name, g.name) << "Align approx_mode in unary_ops.cpp if this fails.";
    }
}

} // namespace ttml::ops::tests
