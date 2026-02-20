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
#include "xtensor/core/xtensor_forward.hpp"

struct AdamWCase {
    std::array<std::size_t, 4> shape;  // (B, H, S, C)
    float lr{1e-3f};
    float beta1{0.9f};
    float beta2{0.999f};
    float epsilon{1e-8f};
    float weight_decay{0.0f};
    bool amsgrad{false};
    std::string name;
};

// Custom printer for AdamWCase used by gtest to make test output readable
void PrintTo(const AdamWCase& pc, std::ostream* os) {
    *os << fmt::format(
        "AdamWCase(name='{}', shape=[{},{},{},{}], lr={}, beta1={}, beta2={}, eps={}, wd={}, amsgrad={})",
        pc.name,
        pc.shape[0],
        pc.shape[1],
        pc.shape[2],
        pc.shape[3],
        pc.lr,
        pc.beta1,
        pc.beta2,
        pc.epsilon,
        pc.weight_decay,
        pc.amsgrad);
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

static xt::xarray<float> make_random_xarray(
    const std::array<std::size_t, 4>& s, uint32_t seed, float min = -1.0F, float max = 1.0F) {
    xt::xarray<float> x = xt::empty<float>({s[0], s[1], s[2], s[3]});
    ttml::core::parallel_generate(
        std::span{x.data(), x.size()}, [min, max]() { return std::uniform_real_distribution<float>(min, max); }, seed);
    return x;
}

static ttnn::Tensor to_tt_bf16(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, &ttml::autograd::ctx().get_device());
}

// CPU reference implementation of AdamW
class CPUAdamW {
public:
    CPUAdamW(float lr, float beta1, float beta2, float epsilon, float weight_decay = 0.0f, bool amsgrad = false) :
        m_lr(lr),
        m_beta1(beta1),
        m_beta2(beta2),
        m_epsilon(epsilon),
        m_weight_decay(weight_decay),
        m_amsgrad(amsgrad),
        m_steps(0) {
    }

    // Set initial momentum state for testing
    void set_state(
        const xt::xarray<float>& first_moment,
        const xt::xarray<float>& second_moment,
        size_t steps,
        const xt::xarray<float>& max_second_moment = {}) {
        m_first_moment = first_moment;
        m_second_moment = second_moment;
        m_steps = steps;
        if (m_amsgrad && max_second_moment.size() > 0) {
            m_max_second_moment = max_second_moment;
        } else if (m_amsgrad) {
            m_max_second_moment = xt::zeros_like(first_moment);
        }
    }

    void step(xt::xarray<float>& params, const xt::xarray<float>& grads) {
        if (m_steps == 0 && m_first_moment.size() == 0) {
            m_first_moment = xt::zeros_like(params);
            m_second_moment = xt::zeros_like(params);
            if (m_amsgrad) {
                m_max_second_moment = xt::zeros_like(params);
            }
        }

        m_steps++;

        // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        m_first_moment = m_beta1 * m_first_moment + (1.0f - m_beta1) * grads;
        // v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_second_moment = m_beta2 * m_second_moment + (1.0f - m_beta2) * (grads * grads);

        float bias_correction1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_steps));
        // m_hat = m_t / (1 - beta1^t)
        xt::xarray<float> first_moment_hat = m_first_moment / bias_correction1;

        float bias_correction2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_steps));
        // v_hat = v_t / (1 - beta2^t)
        xt::xarray<float> second_moment_hat = m_second_moment / bias_correction2;

        // For AMSGrad: use max of past squared gradients
        xt::xarray<float> denom;
        if (m_amsgrad) {
            m_max_second_moment = xt::maximum(m_max_second_moment, m_second_moment);
            denom = xt::sqrt(m_max_second_moment / bias_correction2) + m_epsilon;
        } else {
            denom = xt::sqrt(second_moment_hat) + m_epsilon;
        }

        // params = params - lr * m_hat / denom - lr * weight_decay * params
        params = params - m_lr * first_moment_hat / denom - m_lr * m_weight_decay * params;
    }

private:
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    float m_weight_decay;
    bool m_amsgrad;
    size_t m_steps;
    xt::xarray<float> m_first_moment;
    xt::xarray<float> m_second_moment;
    xt::xarray<float> m_max_second_moment;
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
    return {mean_error, max_error, name};
}

static void run_step_and_compare(const AdamWCase& pc) {
    using namespace ttml;

    ttml::autograd::ctx().set_seed(123U);
    auto& g = autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();
    const uint32_t seed_first_moment = g();
    const uint32_t seed_second_moment = g();
    const uint32_t seed_max_second_moment = g();

    // Same data used for all optimizers
    xt::xarray<float> g0 = make_random_xarray(pc.shape, seed_grad);
    xt::xarray<float> w0 = make_random_xarray(pc.shape, seed_param);

    // Generate random momentum states
    xt::xarray<float> m0 = make_random_xarray(pc.shape, seed_first_moment);
    xt::xarray<float> v0 = make_random_xarray(pc.shape, seed_second_moment, 0.0F, 1.0F);          // must be >= 0
    xt::xarray<float> max_v0 = make_random_xarray(pc.shape, seed_max_second_moment, 0.0F, 1.0F);  // for amsgrad

    // Initial step count (non-zero to test bias correction with accumulated steps)
    const size_t initial_steps = 10;

    // CPU reference implementation
    xt::xarray<float> w_cpu = w0;
    xt::xarray<float> g_cpu = g0;
    CPUAdamW cpu_opt(pc.lr, pc.beta1, pc.beta2, pc.epsilon, pc.weight_decay, pc.amsgrad);
    cpu_opt.set_state(m0, v0, initial_steps, pc.amsgrad ? max_v0 : xt::xarray<float>{});

    // AdamWFused implementation
    auto theta_fused = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_fused->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_fused{{"theta", theta_fused}};

    ttml::optimizers::AdamWFusedConfig fused_cfg;
    fused_cfg.lr = pc.lr;
    fused_cfg.beta1 = pc.beta1;
    fused_cfg.beta2 = pc.beta2;
    fused_cfg.epsilon = pc.epsilon;
    fused_cfg.weight_decay = pc.weight_decay;
    fused_cfg.amsgrad = pc.amsgrad;

    ttml::optimizers::AdamWFused opt_fused(params_fused, fused_cfg);

    // Inject momentum state for AdamWFused
    {
        auto m0_tensor = autograd::create_tensor(to_tt_bf16(m0), false);
        auto v0_tensor = autograd::create_tensor(to_tt_bf16(v0), false);
        serialization::StateDict fused_state;
        fused_state["exp_avg"] = serialization::NamedParameters{{"theta", m0_tensor}};
        fused_state["exp_avg_sq"] = serialization::NamedParameters{{"theta", v0_tensor}};
        fused_state["steps"] = initial_steps;
        fused_state["amsgrad"] = pc.amsgrad;
        if (pc.amsgrad) {
            auto max_v0_tensor = autograd::create_tensor(to_tt_bf16(max_v0), false);
            fused_state["max_exp_avg_sq"] = serialization::NamedParameters{{"theta", max_v0_tensor}};
        }
        opt_fused.set_state_dict(fused_state);
    }

    cpu_opt.step(w_cpu, g_cpu);
    opt_fused.step();

    auto result_fused = theta_fused->get_value();
    auto result_fused_cpu = core::to_xtensor(result_fused);

    auto fused_metrics = compute_error_metrics(w_cpu, result_fused_cpu, "AdamWFused");

    const float mean_error_tolerance = 1e-3f;
    const float max_error_tolerance = 1e-2f;

    EXPECT_LE(fused_metrics.mean_error, mean_error_tolerance) << "AdamWFused mean error exceeds tolerance";
    EXPECT_LE(fused_metrics.max_error, max_error_tolerance) << "AdamWFused max error exceeds tolerance";
}

static std::string CaseName(const ::testing::TestParamInfo<AdamWCase>& info) {
    const auto& c = info.param;
    return fmt::format("{}_B{}H{}S{}C{}", c.name, c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
}

TEST_P(AdamWFusedComparisonTest, CompareImplementations) {
    const auto& pc = GetParam();
    run_step_and_compare(pc);
}

// Note: In the following test suites there are no test cases with beta2=0. When beta2=0, denom = |g_t| + eps which can
// be very small, while m_hat_t (from accumulated momentum) can be large. This may cause pathological updates that blow
// up weights to large values where distance between two consecutive bf16 numbers is poor and the max error test case
// will fail

// Test cases with various hyperparameter configurations
static const AdamWCase kBasicCases[] = {
    // Standard configurations with different learning rates
    {{1, 1, 128, 512}, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Standard_lr1e2"},
    {{1, 1, 1, 65'536}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Standard_lr1e3"},
    {{2, 4, 32, 256}, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Standard_lr1e4"},
    // Different beta1 values
    {{1, 4, 64, 128}, 1e-3f, 0.8f, 0.999f, 1e-8f, 0.0f, false, "Beta1_0p8"},
    {{2, 4, 64, 64}, 1e-3f, 0.5f, 0.999f, 1e-8f, 0.0f, false, "Beta1_0p5"},
    // Different beta2 values
    {{1, 8, 32, 128}, 1e-3f, 0.9f, 0.99f, 1e-8f, 0.0f, false, "Beta2_0p99"},
    {{1, 32, 32, 32}, 1e-3f, 0.9f, 0.95f, 1e-8f, 0.0f, false, "Beta2_0p95"},
    // Different epsilon values
    {{1, 16, 32, 64}, 1e-3f, 0.9f, 0.999f, 1e-6f, 0.0f, false, "Epsilon_1e6"},
    {{2, 8, 32, 64}, 1e-3f, 0.9f, 0.999f, 1e-9f, 0.0f, false, "Epsilon_1e9"},
    // Different tensor shapes
    {{2, 8, 64, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "NIGHTLY_Large_4D"},
    {{1, 512, 32, 32}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "NIGHTLY_Wide"},
    // Only beta2 (second moment), beta1=0
    {{1, 1, 1, 32'768}, 1e-3f, 0.0f, 0.999f, 1e-8f, 0.0f, false, "OnlyBeta2_0p999"},
    {{1, 8, 128, 64}, 1e-3f, 0.0f, 0.99f, 1e-8f, 0.0f, false, "OnlyBeta2_0p99"},
    {{1, 8, 128, 512}, 1e-3f, 0.0f, 0.999f, 1e-6f, 0.0f, false, "NIGHTLY_Beta2_eps1e6"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFusedBasicComparison, AdamWFusedComparisonTest, ::testing::ValuesIn(kBasicCases), CaseName);

// ====================================================================
// Weight Decay Tests
// Test AdamW with various weight decay configurations
// ====================================================================

// Test cases with weight decay enabled
static const AdamWCase kWeightDecayCases[] = {
    // Standard weight decay values
    {{1, 4, 32, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, false, "Standard_wd0p01"},
    {{1, 1, 1, 65'536}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f, false, "Standard_wd0p1"},
    // Weight decay with different learning rates
    {{1, 8, 64, 128}, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.01f, false, "HighLR_wd0p01"},
    {{1, 8, 64, 128}, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, false, "LowLR_wd0p01"},
    // Edge cases: very high and very low weight decay
    {{1, 4, 128, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-5f, false, "NIGHTLY_VerySmallWD_1e5"},
    {{1, 8, 64, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.5f, false, "NIGHTLY_VeryHighWD_0p5"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFusedWeightDecay, AdamWFusedComparisonTest, ::testing::ValuesIn(kWeightDecayCases), CaseName);

// ====================================================================
// AMSGrad Tests
// Test AdamW with AMSGrad variant enabled
// ====================================================================

// Test cases with AMSGrad enabled
static const AdamWCase kAMSGradCases[] = {
    // Standard AMSGrad
    {{1, 1, 1, 65'536}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "Standard"},
    // AMSGrad with weight decay
    {{1, 4, 64, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true, "WeightDecay_0p01"},
    // AMSGrad with different shape
    {{2, 8, 64, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "NIGHTLY_Large_4D"},
};

INSTANTIATE_TEST_SUITE_P(AdamWFusedAMSGrad, AdamWFusedComparisonTest, ::testing::ValuesIn(kAMSGradCases), CaseName);

// ====================================================================
// Stochastic Rounding Tests
// Test AdamW with stochastic rounding enabled
// ====================================================================

// These tests are nondeterministic but should never fail
class StochasticRoundingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// Test to verify stochastic rounding rounds in the correct direction (towards CPU result)
TEST_F(StochasticRoundingTest, RoundingDirectionCorrectness) {
    using namespace ttml;

    const std::array<std::size_t, 4> shape = {1, 4, 128, 256};
    const uint32_t steps = 100;

    // Initialize with ones - small gradients relative to parameter magnitude
    xt::xarray<float> w0 = xt::ones<float>({shape[0], shape[1], shape[2], shape[3]});
    // Very small gradients - these might get rounded away without stochastic rounding
    xt::xarray<float> g0 = xt::ones<float>({shape[0], shape[1], shape[2], shape[3]}) * 1e-3f;

    // Run with stochastic rounding
    auto theta_stoch = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_stoch->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_stoch{{"theta", theta_stoch}};

    ttml::optimizers::AdamWFusedConfig stoch_cfg;
    stoch_cfg.lr = 1e-3f;
    stoch_cfg.beta1 = 0.9f;
    stoch_cfg.beta2 = 0.999f;
    stoch_cfg.epsilon = 1e-8f;
    stoch_cfg.stochastic_rounding = true;

    ttml::optimizers::AdamWFused opt_stoch(params_stoch, stoch_cfg);

    auto theta_det = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_det->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_det{{"theta", theta_det}};

    ttml::optimizers::AdamWFusedConfig det_cfg;
    det_cfg.lr = 1e-3f;
    det_cfg.beta1 = 0.9f;
    det_cfg.beta2 = 0.999f;
    det_cfg.epsilon = 1e-8f;
    det_cfg.stochastic_rounding = false;

    ttml::optimizers::AdamWFused opt_det(params_det, det_cfg);

    xt::xarray<float> w_cpu = w0;
    CPUAdamW cpu_opt(1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false);

    for (uint32_t i = 0; i < steps; ++i) {
        opt_stoch.step();
        opt_det.step();
        cpu_opt.step(w_cpu, g0);
    }

    auto result_stoch = core::to_xtensor(theta_stoch->get_value());
    auto result_det = core::to_xtensor(theta_det->get_value());

    float error_stoch = xt::sum(xt::abs(result_stoch - w_cpu))();
    float error_det = xt::sum(xt::abs(result_det - w_cpu))();

    // Stochastic rounding should be closer to CPU result than deterministic
    EXPECT_LT(error_stoch, error_det)
        << "Stochastic rounding should produce weights closer to CPU reference than deterministic rounding";
}

// Verifies mean and max error is lower in the stochastic rounding version given enough steps
TEST_F(StochasticRoundingTest, NIGHTLY_ErrorComparisonOverMultipleSteps) {
    using namespace ttml;

    const std::array<std::size_t, 4> shape = {1, 4, 256, 512};
    const uint32_t steps = 512U;
    const uint32_t seed = 42U;

    xt::xarray<float> w0 = make_random_xarray(shape, seed);
    xt::xarray<float> g0 = make_random_xarray(shape, seed + 1, -0.1f, 0.1f);

    xt::xarray<float> w_cpu = w0;
    CPUAdamW cpu_opt(1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false);

    auto theta_stoch = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_stoch->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_stoch{{"theta", theta_stoch}};

    ttml::optimizers::AdamWFusedConfig stoch_cfg;
    stoch_cfg.lr = 1e-3f;
    stoch_cfg.beta1 = 0.9f;
    stoch_cfg.beta2 = 0.999f;
    stoch_cfg.epsilon = 1e-8f;
    stoch_cfg.stochastic_rounding = true;
    ttml::optimizers::AdamWFused opt_stoch(params_stoch, stoch_cfg);

    auto theta_det = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_det->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_det{{"theta", theta_det}};

    ttml::optimizers::AdamWFusedConfig det_cfg;
    det_cfg.lr = 1e-3f;
    det_cfg.beta1 = 0.9f;
    det_cfg.beta2 = 0.999f;
    det_cfg.epsilon = 1e-8f;
    det_cfg.stochastic_rounding = false;
    ttml::optimizers::AdamWFused opt_det(params_det, det_cfg);

    for (uint32_t i = 0; i < steps; ++i) {
        cpu_opt.step(w_cpu, g0);
        opt_stoch.step();
        opt_det.step();
    }

    auto result_stoch = core::to_xtensor(theta_stoch->get_value());
    auto result_det = core::to_xtensor(theta_det->get_value());

    auto stoch_metrics = compute_error_metrics(w_cpu, result_stoch, "Stochastic");
    auto det_metrics = compute_error_metrics(w_cpu, result_det, "Deterministic");

    EXPECT_LT(stoch_metrics.mean_error, det_metrics.mean_error);
    EXPECT_LT(stoch_metrics.max_error, det_metrics.max_error);
}
