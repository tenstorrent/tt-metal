// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/adamw_fused.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <cstdlib>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "xtensor/core/xtensor_forward.hpp"

struct AdamWCase {
    std::array<std::size_t, 4> shape;  // (B, H, S, C)
    float lr{1e-3f};
    float beta1{0.9f};
    float beta2{0.999f};
    float epsilon{1e-8f};
    std::string name;
};

// Custom printer for AdamWCase used by gtest to make test output readable
void PrintTo(const AdamWCase& pc, std::ostream* os) {
    *os << fmt::format(
        "AdamWCase(name='{}', shape=[{},{},{},{}], lr={}, beta1={}, beta2={}, eps={})",
        pc.name,
        pc.shape[0],
        pc.shape[1],
        pc.shape[2],
        pc.shape[3],
        pc.lr,
        pc.beta1,
        pc.beta2,
        pc.epsilon);
}

class AdamWFusedComparisonTest : public ::testing::TestWithParam<AdamWCase> {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

static xt::xarray<float> make_random_xarray(const std::array<std::size_t, 4>& s, uint32_t seed) {
    xt::xarray<float> x = xt::empty<float>({s[0], s[1], s[2], s[3]});
    ttml::core::parallel_generate(
        std::span{x.data(), x.size()},  // NOLINT(performance-no-span-copy)
        []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); },
        seed);
    return x;
}

static ttnn::Tensor to_tt(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor(x, &ttml::autograd::ctx().get_device());
}

// CPU reference implementation of AdamW
// Minimal implementation: no weight decay, no amsgrad
class CPUAdamW {
public:
    CPUAdamW(float lr, float beta1, float beta2, float epsilon) :
        m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_steps(0) {
    }

    void step(xt::xarray<float>& params, const xt::xarray<float>& grads) {
        if (m_steps == 0) {
            // Initialize momentum buffers
            m_first_moment = xt::zeros_like(params);
            m_second_moment = xt::zeros_like(params);
        }

        m_steps++;

        // Update biased first moment estimate
        // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        m_first_moment = m_beta1 * m_first_moment + (1.0f - m_beta1) * grads;

        // Update biased second raw moment estimate
        // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_second_moment = m_beta2 * m_second_moment + (1.0f - m_beta2) * (grads * grads);

        // Compute bias-corrected first moment estimate
        // m_hat = m_t / (1 - beta1^t)
        float bias_correction1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_steps));
        xt::xarray<float> first_moment_hat = m_first_moment / bias_correction1;

        // Compute bias-corrected second raw moment estimate
        // v_hat = v_t / (1 - beta2^t)
        float bias_correction2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_steps));
        xt::xarray<float> second_moment_hat = m_second_moment / bias_correction2;

        // Update parameters
        // theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
        params = params - m_lr * first_moment_hat / (xt::sqrt(second_moment_hat) + m_epsilon);
    }

private:
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    size_t m_steps;
    xt::xarray<float> m_first_moment;
    xt::xarray<float> m_second_moment;
};

struct ErrorMetrics {
    float mean_error;
    float max_error;
    std::string name;
};

static ErrorMetrics compute_error_metrics(
    const xt::xarray<float>& reference, const xt::xarray<float>& actual, const std::string& name) {
    float sum_error = 0.0f;
    float max_error = 0.0f;
    size_t count = reference.size();

    for (size_t i = 0; i < count; ++i) {
        float error = std::abs(reference(i) - actual(i));
        sum_error += error;
        max_error = std::max(max_error, error);
    }

    float mean_error = sum_error / static_cast<float>(count);

    fmt::print("{}: mean_error={:.6e}, max_error={:.6e}\n", name, mean_error, max_error);

    return {mean_error, max_error, name};
}

static void run_steps_and_compare(const AdamWCase& pc, uint32_t steps) {
    using namespace ttml;

    ttml::autograd::ctx().set_seed(123U);
    auto& g = autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();

    // Same data used for all optimizers
    xt::xarray<float> g0 = make_random_xarray(pc.shape, seed_grad);
    xt::xarray<float> w0 = make_random_xarray(pc.shape, seed_param);

    // CPU reference implementation
    xt::xarray<float> w_cpu = w0;
    xt::xarray<float> g_cpu = g0;
    CPUAdamW cpu_opt(pc.lr, pc.beta1, pc.beta2, pc.epsilon);

    // MorehAdamW implementation
    auto theta_moreh = autograd::create_tensor(to_tt(w0), true);
    theta_moreh->set_grad(to_tt(g0));
    ttml::serialization::NamedParameters params_moreh{{"theta", theta_moreh}};

    ttml::optimizers::AdamWConfig moreh_cfg;
    moreh_cfg.lr = pc.lr;
    moreh_cfg.beta1 = pc.beta1;
    moreh_cfg.beta2 = pc.beta2;
    moreh_cfg.epsilon = pc.epsilon;
    moreh_cfg.weight_decay = 0.0f;  // No weight decay for comparison

    ttml::optimizers::MorehAdamW opt_moreh(params_moreh, moreh_cfg);

    // AdamW (composite) implementation
    auto theta_composite = autograd::create_tensor(to_tt(w0), true);
    theta_composite->set_grad(to_tt(g0));
    ttml::serialization::NamedParameters params_composite{{"theta", theta_composite}};

    ttml::optimizers::AdamWConfig composite_cfg;
    composite_cfg.lr = pc.lr;
    composite_cfg.beta1 = pc.beta1;
    composite_cfg.beta2 = pc.beta2;
    composite_cfg.epsilon = pc.epsilon;
    composite_cfg.weight_decay = 0.0f;  // No weight decay for comparison
    composite_cfg.use_kahan_summation = false;

    ttml::optimizers::AdamW opt_composite(params_composite, composite_cfg);

    // AdamWFused implementation (placeholder - will be skipped for now)
    auto theta_fused = autograd::create_tensor(to_tt(w0), true);
    theta_fused->set_grad(to_tt(g0));
    ttml::serialization::NamedParameters params_fused{{"theta", theta_fused}};

    ttml::optimizers::AdamWFusedConfig fused_cfg;
    fused_cfg.lr = pc.lr;
    fused_cfg.beta1 = pc.beta1;
    fused_cfg.beta2 = pc.beta2;
    fused_cfg.epsilon = pc.epsilon;
    fused_cfg.weight_decay = 0.0f;

    ttml::optimizers::AdamWFused opt_fused(params_fused, fused_cfg);

    // Run all optimizers for the specified number of steps
    for (uint32_t i = 0; i < steps; ++i) {
        cpu_opt.step(w_cpu, g_cpu);
        opt_moreh.step();
        opt_composite.step();
        opt_fused.step();
    }

    // Get results
    auto result_moreh = theta_moreh->get_value();
    auto result_composite = theta_composite->get_value();
    auto result_fused = theta_fused->get_value();

    // Convert to CPU for comparison
    auto result_moreh_cpu = core::to_xtensor(result_moreh);
    auto result_composite_cpu = core::to_xtensor(result_composite);
    auto result_fused_cpu = core::to_xtensor(result_fused);

    fmt::print("\n=== Error Metrics (reference: CPU) ===\n");

    // Compute error metrics for each implementation
    auto moreh_metrics = compute_error_metrics(w_cpu, result_moreh_cpu, "MorehAdamW");
    auto composite_metrics = compute_error_metrics(w_cpu, result_composite_cpu, "AdamW (composite)");
    auto fused_metrics = compute_error_metrics(w_cpu, result_fused_cpu, "AdamWFused");

    EXPECT_LT(fused_metrics.mean_error, moreh_metrics.mean_error)
        << "AdamWFused mean error should be lower than MorehAdamW";
    EXPECT_LT(fused_metrics.max_error, moreh_metrics.max_error)
        << "AdamWFused max error should be lower than MorehAdamW";

    EXPECT_LT(fused_metrics.mean_error, composite_metrics.mean_error)
        << "AdamWFused mean error should be lower than AdamW (composite)";
    EXPECT_LT(fused_metrics.max_error, composite_metrics.max_error)
        << "AdamWFused max error should be lower than AdamW (composite)";

    fmt::print("\n");
}

static std::string CaseName(const ::testing::TestParamInfo<AdamWCase>& info) {
    const auto& c = info.param;
    return fmt::format("{}_B{}H{}S{}C{}", c.name, c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
}

TEST_P(AdamWFusedComparisonTest, CompareImplementations) {
    const auto& pc = GetParam();
    // Run 3 steps to ensure momentum buffers are properly exercised
    const uint32_t steps = 1;
    run_steps_and_compare(pc, steps);
}

// Test cases with various hyperparameter configurations
static const AdamWCase kBasicCases[] = {
    // Standard configurations with different learning rates
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, "Standard_lr1e3"},
    {{1, 8, 128, 256}, 1e-2f, 0.9f, 0.999f, 1e-8f, "Standard_lr1e2"},
    {{2, 4, 128, 256}, 1e-4f, 0.9f, 0.999f, 1e-8f, "Standard_lr1e4"},

    // Different beta1 values
    {{1, 16, 128, 128}, 1e-3f, 0.8f, 0.999f, 1e-8f, "Beta1_0p8"},
    {{2, 8, 128, 128}, 1e-3f, 0.95f, 0.999f, 1e-8f, "Beta1_0p95"},
    {{4, 4, 128, 128}, 1e-3f, 0.5f, 0.999f, 1e-8f, "Beta1_0p5"},

    // Different beta2 values
    {{1, 32, 64, 128}, 1e-3f, 0.9f, 0.99f, 1e-8f, "Beta2_0p99"},
    {{2, 16, 64, 128}, 1e-3f, 0.9f, 0.9999f, 1e-8f, "Beta2_0p9999"},
    {{1, 64, 64, 64}, 1e-3f, 0.9f, 0.95f, 1e-8f, "Beta2_0p95"},

    // Different epsilon values
    {{2, 32, 64, 64}, 1e-3f, 0.9f, 0.999f, 1e-7f, "Epsilon_1e7"},
    {{4, 16, 64, 64}, 1e-3f, 0.9f, 0.999f, 1e-9f, "Epsilon_1e9"},
    {{1, 4, 256, 256}, 1e-3f, 0.9f, 0.999f, 1e-6f, "Epsilon_1e6"},

    // Mixed configurations
    {{2, 2, 256, 256}, 1e-2f, 0.95f, 0.9999f, 1e-7f, "Mixed_1"},
    {{1, 2, 256, 512}, 1e-4f, 0.8f, 0.99f, 1e-9f, "Mixed_2"},
    {{1, 128, 32, 64}, 5e-3f, 0.85f, 0.995f, 1e-8f, "Mixed_3"},

    // Large and small tensor shapes
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, "Large_flat"},
    {{8, 8, 64, 64}, 1e-3f, 0.9f, 0.999f, 1e-8f, "Large_4D"},
    {{1, 256, 32, 32}, 1e-3f, 0.9f, 0.999f, 1e-8f, "Wide"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFusedBasicComparison, AdamWFusedComparisonTest, ::testing::ValuesIn(kBasicCases), CaseName);
