// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/adamw.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "adamw_test_utils.hpp"
#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/muon_composite.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "xtensor/core/xtensor_forward.hpp"

using ttml::test_utils::adamw::error_metrics;

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

class AdamWComparisonTest : public ::testing::TestWithParam<AdamWCase> {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

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

    // Bias-correct the next step with these beta powers, as the kernel takes them, instead of beta^t.
    void set_beta_powers(float beta1_pow, float beta2_pow) {
        m_beta_powers = {beta1_pow, beta2_pow};
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

        float bias_correction1 =
            1.0f - (m_beta_powers ? m_beta_powers->first : std::pow(m_beta1, static_cast<float>(m_steps)));
        // m_hat = m_t / (1 - beta1^t)
        xt::xarray<float> first_moment_hat = m_first_moment / bias_correction1;

        float bias_correction2 =
            1.0f - (m_beta_powers ? m_beta_powers->second : std::pow(m_beta2, static_cast<float>(m_steps)));
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

    const xt::xarray<float>& first_moment() const {
        return m_first_moment;
    }
    const xt::xarray<float>& second_moment() const {
        return m_second_moment;
    }
    const xt::xarray<float>& max_second_moment() const {
        return m_max_second_moment;
    }

private:
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    float m_weight_decay;
    bool m_amsgrad;
    size_t m_steps;
    std::optional<std::pair<float, float>> m_beta_powers;
    xt::xarray<float> m_first_moment;
    xt::xarray<float> m_second_moment;
    xt::xarray<float> m_max_second_moment;
};

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
    xt::xarray<float> g0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_grad);
    xt::xarray<float> w0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_param);

    // Generate random momentum states
    xt::xarray<float> m0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_first_moment);
    xt::xarray<float> v0 =
        ttml::test_utils::make_uniform_xarray<float>(pc.shape, 0.0F, 1.0F, seed_second_moment);  // must be >= 0
    xt::xarray<float> max_v0 =
        ttml::test_utils::make_uniform_xarray<float>(pc.shape, 0.0F, 1.0F, seed_max_second_moment);  // for amsgrad

    // Initial step count (non-zero to test bias correction with accumulated steps)
    const size_t initial_steps = 10;

    // CPU reference implementation
    xt::xarray<float> w_cpu = w0;
    xt::xarray<float> g_cpu = g0;
    CPUAdamW cpu_opt(pc.lr, pc.beta1, pc.beta2, pc.epsilon, pc.weight_decay, pc.amsgrad);
    cpu_opt.set_state(m0, v0, initial_steps, pc.amsgrad ? max_v0 : xt::xarray<float>{});

    // AdamW implementation
    auto theta_fused = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_fused->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_fused{{"theta", theta_fused}};

    ttml::optimizers::AdamWConfig fused_cfg;
    fused_cfg.lr = pc.lr;
    fused_cfg.beta1 = pc.beta1;
    fused_cfg.beta2 = pc.beta2;
    fused_cfg.epsilon = pc.epsilon;
    fused_cfg.weight_decay = pc.weight_decay;
    fused_cfg.amsgrad = pc.amsgrad;

    ttml::optimizers::AdamW opt_fused(params_fused, fused_cfg);

    // Inject momentum state for AdamW
    {
        auto m0_tensor = autograd::create_tensor(to_tt_bf16(m0), false);
        auto v0_tensor = autograd::create_tensor(to_tt_bf16(v0), false);
        auto fused_state = opt_fused.get_state_dict();
        fused_state["exp_avg"] = serialization::NamedParameters{{"theta", m0_tensor}};
        fused_state["exp_avg_sq"] = serialization::NamedParameters{{"theta", v0_tensor}};
        fused_state["steps"] = initial_steps;
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

    auto fused_metrics = error_metrics(w_cpu, result_fused_cpu);

    const float mean_error_tolerance = 1e-3f;
    const float max_error_tolerance = 1e-2f;

    EXPECT_LE(fused_metrics.mean_error, mean_error_tolerance) << "AdamW mean error exceeds tolerance";
    EXPECT_LE(fused_metrics.max_error, max_error_tolerance) << "AdamW max error exceeds tolerance";
}

static std::string CaseName(const ::testing::TestParamInfo<AdamWCase>& info) {
    const auto& c = info.param;
    return fmt::format("{}_B{}H{}S{}C{}", c.name, c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
}

TEST_P(AdamWComparisonTest, CompareImplementations) {
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

INSTANTIATE_TEST_SUITE_P(AdamWBasicComparison, AdamWComparisonTest, ::testing::ValuesIn(kBasicCases), CaseName);

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

INSTANTIATE_TEST_SUITE_P(AdamWWeightDecay, AdamWComparisonTest, ::testing::ValuesIn(kWeightDecayCases), CaseName);

// ====================================================================
// Param group tests
// Every group is stepped with its own options; params in no group use the
// defaults. A group with weight_decay = 0 (e.g. RMSNorm gains and biases)
// must not be weight-decayed, while the default group still is.
// ====================================================================

class AdamWParamGroupsTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

TEST_F(AdamWParamGroupsTest, ZeroWeightDecayGroupSkipsDecay) {
    using namespace ttml;

    const std::array<std::size_t, 4> shape_1d = {1, 1, 1, 4096};   // RMSNorm-gain-like
    const std::array<std::size_t, 4> shape_2d = {1, 1, 128, 256};  // matmul-weight-like

    const float lr = 1e-2f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;
    // Large wd so the decay term (lr*wd*param) is well above tolerance: makes the "would-be-decayed"
    // reference clearly separable from the (correctly) undecayed 1-D result.
    const float weight_decay = 1.0f;
    const size_t initial_steps = 10;

    autograd::ctx().set_seed(123U);
    auto& gen = autograd::ctx().get_generator();
    const uint32_t seed_1d_w = gen();
    const uint32_t seed_1d_g = gen();
    const uint32_t seed_1d_m = gen();
    const uint32_t seed_1d_v = gen();
    const uint32_t seed_2d_w = gen();
    const uint32_t seed_2d_g = gen();
    const uint32_t seed_2d_m = gen();
    const uint32_t seed_2d_v = gen();

    xt::xarray<float> w1_0 = test_utils::make_uniform_xarray<float>(shape_1d, -1.0F, 1.0F, seed_1d_w);
    xt::xarray<float> g1_0 = test_utils::make_uniform_xarray<float>(shape_1d, -1.0F, 1.0F, seed_1d_g);
    xt::xarray<float> m1_0 = test_utils::make_uniform_xarray<float>(shape_1d, -1.0F, 1.0F, seed_1d_m);
    xt::xarray<float> v1_0 = test_utils::make_uniform_xarray<float>(shape_1d, 0.0F, 1.0F, seed_1d_v);

    xt::xarray<float> w2_0 = test_utils::make_uniform_xarray<float>(shape_2d, -1.0F, 1.0F, seed_2d_w);
    xt::xarray<float> g2_0 = test_utils::make_uniform_xarray<float>(shape_2d, -1.0F, 1.0F, seed_2d_g);
    xt::xarray<float> m2_0 = test_utils::make_uniform_xarray<float>(shape_2d, -1.0F, 1.0F, seed_2d_m);
    xt::xarray<float> v2_0 = test_utils::make_uniform_xarray<float>(shape_2d, 0.0F, 1.0F, seed_2d_v);

    // References: 1-D param stepped WITHOUT weight decay, 2-D param stepped WITH weight decay.
    xt::xarray<float> w1_no_wd = w1_0;
    CPUAdamW cpu_1d_no_wd(lr, beta1, beta2, epsilon, /*weight_decay=*/0.0f, /*amsgrad=*/false);
    cpu_1d_no_wd.set_state(m1_0, v1_0, initial_steps);
    cpu_1d_no_wd.step(w1_no_wd, g1_0);

    xt::xarray<float> w2_with_wd = w2_0;
    CPUAdamW cpu_2d_with_wd(lr, beta1, beta2, epsilon, weight_decay, false);
    cpu_2d_with_wd.set_state(m2_0, v2_0, initial_steps);
    cpu_2d_with_wd.step(w2_with_wd, g2_0);

    // What the 1-D param WOULD be if it were (incorrectly) decayed — used to prove the decay is
    // large enough to matter, so the "no decay" match below isn't a vacuous pass.
    xt::xarray<float> w1_with_wd = w1_0;
    CPUAdamW cpu_1d_with_wd(lr, beta1, beta2, epsilon, weight_decay, false);
    cpu_1d_with_wd.set_state(m1_0, v1_0, initial_steps);
    cpu_1d_with_wd.step(w1_with_wd, g1_0);

    // Device AdamW: both params in one optimizer, only "gain" excluded from weight decay.
    auto gain = autograd::create_tensor(to_tt_bf16(w1_0), true);
    gain->set_grad(to_tt_bf16(g1_0));
    auto weight = autograd::create_tensor(to_tt_bf16(w2_0), true);
    weight->set_grad(to_tt_bf16(g2_0));
    serialization::NamedParameters params{{"gain", gain}, {"weight", weight}};

    optimizers::AdamWConfig cfg;
    cfg.lr = lr;
    cfg.beta1 = beta1;
    cfg.beta2 = beta2;
    cfg.epsilon = epsilon;
    cfg.weight_decay = weight_decay;
    optimizers::AdamW opt(params, cfg, {{.params = {"gain"}, .options = {{"weight_decay", 0.0F}}}});

    // Inject momentum state on top of the optimizer's own state dict, which carries its param groups.
    {
        auto state = opt.get_state_dict();
        state["exp_avg"] = serialization::NamedParameters{
            {"gain", autograd::create_tensor(to_tt_bf16(m1_0), false)},
            {"weight", autograd::create_tensor(to_tt_bf16(m2_0), false)}};
        state["exp_avg_sq"] = serialization::NamedParameters{
            {"gain", autograd::create_tensor(to_tt_bf16(v1_0), false)},
            {"weight", autograd::create_tensor(to_tt_bf16(v2_0), false)}};
        state["steps"] = initial_steps;
        opt.set_state_dict(state);
    }

    opt.step();

    auto gain_result = core::to_xtensor(gain->get_value());
    auto weight_result = core::to_xtensor(weight->get_value());

    const float mean_error_tolerance = 1e-3f;
    const float max_error_tolerance = 1e-2f;

    // 1-D param must match the NO-weight-decay reference.
    auto gain_metrics = error_metrics(w1_no_wd, gain_result);
    EXPECT_LE(gain_metrics.mean_error, mean_error_tolerance) << "1-D param should not be weight-decayed";
    EXPECT_LE(gain_metrics.max_error, max_error_tolerance) << "1-D param should not be weight-decayed";

    // 2-D param must match the weight-decayed reference.
    auto weight_metrics = error_metrics(w2_with_wd, weight_result);
    EXPECT_LE(weight_metrics.mean_error, mean_error_tolerance) << "2-D param should be weight-decayed";
    EXPECT_LE(weight_metrics.max_error, max_error_tolerance) << "2-D param should be weight-decayed";

    // Guard against a vacuous pass: the undecayed 1-D result must be clearly distinct from the
    // would-be-decayed reference, i.e. the weight decay is actually large enough to observe.
    auto skip_vs_decay = error_metrics(w1_with_wd, gain_result);
    EXPECT_GT(skip_vs_decay.mean_error, mean_error_tolerance)
        << "weight decay too small to distinguish skipping from applying it; test is not meaningful";
}

TEST_F(AdamWParamGroupsTest, FillsDefaultsAndGroupsUngroupedParams) {
    using namespace ttml;

    const xt::xarray<float> zeros = xt::zeros<float>({1, 1, 32, 32});
    serialization::NamedParameters params{
        {"b", autograd::create_tensor(to_tt_bf16(zeros), true)},
        {"a", autograd::create_tensor(to_tt_bf16(zeros), true)},
        {"gamma", autograd::create_tensor(to_tt_bf16(zeros), true)}};

    optimizers::AdamWConfig defaults;
    defaults.weight_decay = 0.1F;
    optimizers::AdamW opt(params, defaults, {{.params = {"gamma"}, .options = {{"weight_decay", 0.0F}}}});

    // Ungrouped params form a leading group with the defaults; explicit groups get missing options filled.
    const auto& groups = opt.get_param_groups();
    ASSERT_EQ(groups.size(), 2U);
    EXPECT_EQ(groups[0].params, (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(groups[0].options, opt.get_defaults());
    EXPECT_EQ(groups[1].params, (std::vector<std::string>{"gamma"}));
    EXPECT_FLOAT_EQ(groups[1].options.at("weight_decay"), 0.0F);
    EXPECT_FLOAT_EQ(groups[1].options.at("lr"), defaults.lr);
    EXPECT_FLOAT_EQ(groups[1].options.at("beta1"), defaults.beta1);
}

TEST_F(AdamWParamGroupsTest, RejectsInvalidGroups) {
    using namespace ttml;

    const xt::xarray<float> zeros = xt::zeros<float>({1, 1, 32, 32});
    auto theta = autograd::create_tensor(to_tt_bf16(zeros), true);
    serialization::NamedParameters params{{"theta", theta}};
    const optimizers::AdamWConfig defaults;

    // Unknown param name.
    EXPECT_THROW(
        optimizers::AdamW(params, defaults, {{.params = {"missing"}, .options = {{"weight_decay", 0.0F}}}}),
        std::invalid_argument);

    // Same param in two groups.
    EXPECT_THROW(
        optimizers::AdamW(
            params,
            defaults,
            {{.params = {"theta"}, .options = {{"weight_decay", 0.0F}}}, {.params = {"theta"}, .options = {}}}),
        std::invalid_argument);

    // Option that is not a per-group AdamW hyperparameter.
    EXPECT_THROW(
        optimizers::AdamW(params, defaults, {{.params = {"theta"}, .options = {{"amsgrad", 1.0F}}}}),
        std::invalid_argument);
}

namespace {
// Two params: "w" in the leading default group, "gamma" in a group with its own lr and weight_decay.
ttml::serialization::NamedParameters make_grouped_params() {
    const xt::xarray<float> zeros = xt::zeros<float>({1, 1, 32, 32});
    return {
        {"w", ttml::autograd::create_tensor(to_tt_bf16(zeros), true)},
        {"gamma", ttml::autograd::create_tensor(to_tt_bf16(zeros), true)}};
}

std::unique_ptr<ttml::optimizers::AdamW> make_grouped_adamw(const ttml::serialization::NamedParameters& params) {
    ttml::optimizers::AdamWConfig defaults;
    defaults.lr = 1e-2F;
    defaults.weight_decay = 0.1F;
    return std::make_unique<ttml::optimizers::AdamW>(
        params,
        defaults,
        std::vector<ttml::optimizers::ParamGroup>{
            {.params = {"gamma"}, .options = {{"lr", 2e-2F}, {"weight_decay", 0.0F}}}});
}
}  // namespace

TEST_F(AdamWParamGroupsTest, StateDictRestoresEveryGroupsOptions) {
    using namespace ttml;
    auto params = make_grouped_params();
    auto src = make_grouped_adamw(params);
    (void)src->get_initial_group_lrs();           // a scheduler attaching records the initial LRs
    src->set_param_group_option(0, "lr", 5e-3F);  // ...and the LRs have moved since
    src->set_param_group_option(1, "lr", 7e-3F);
    src->set_param_group_option(1, "beta2", 0.95F);
    src->set_param_group_option(0, "weight_decay", 0.05F);
    const auto state = src->get_state_dict();

    auto dst = make_grouped_adamw(params);
    dst->set_state_dict(state);

    ASSERT_EQ(dst->get_param_groups().size(), src->get_param_groups().size());
    for (size_t g = 0; g < src->get_param_groups().size(); ++g) {
        EXPECT_EQ(dst->get_param_groups()[g].options, src->get_param_groups()[g].options) << "group " << g;
    }
    EXPECT_EQ(dst->get_initial_group_lrs(), (std::vector<float>{1e-2F, 2e-2F}));
}

TEST_F(AdamWParamGroupsTest, StateDictRejectsMismatchedGroups) {
    using namespace ttml;
    auto params = make_grouped_params();
    const auto state = make_grouped_adamw(params)->get_state_dict();

    // One group instead of two.
    optimizers::AdamW one_group(params, optimizers::AdamWConfig{});
    EXPECT_THROW(one_group.set_state_dict(state), std::runtime_error);

    // Two groups, but with different sizes: {0, 2} params vs the checkpoint's {1, 1}.
    optimizers::AdamW other_sizes(params, optimizers::AdamWConfig{}, {{.params = {"w", "gamma"}, .options = {}}});
    EXPECT_THROW(other_sizes.set_state_dict(state), std::runtime_error);

    // Same sizes {1, 1}, but the members swapped ({gamma}, {w} instead of {w}, {gamma}): the saved options would
    // land on the wrong params, so this is rejected too.
    optimizers::AdamW swapped(params, optimizers::AdamWConfig{}, {{.params = {"w"}, .options = {}}});
    ASSERT_EQ(swapped.get_param_groups()[0].params, std::vector<std::string>{"gamma"});
    EXPECT_THROW(swapped.set_state_dict(state), std::runtime_error);
}

TEST_F(AdamWParamGroupsTest, StateDictRequiresGroupEntries) {
    using namespace ttml;
    auto params = make_grouped_params();
    auto src = make_grouped_adamw(params);
    src->set_param_group_option(0, "lr", 5e-3F);
    src->set_param_group_option(0, "beta1", 0.5F);
    auto state = src->get_state_dict();
    state.erase("param_groups.1.initial_lr");

    auto dst = make_grouped_adamw(params);
    const auto groups_before = dst->get_param_groups();
    EXPECT_THROW(dst->set_state_dict(state), std::runtime_error);
    // Group 1 is invalid, so group 0's (valid, different) options must not have been applied either.
    for (size_t g = 0; g < groups_before.size(); ++g) {
        EXPECT_EQ(dst->get_param_groups()[g].options, groups_before[g].options) << "group " << g;
    }
}

// ====================================================================
// State-dict restore tests
// A checkpoint's betas may differ from the constructor config; both the
// moment updates and the bias correction (beta powers) must follow the
// restored betas.
// ====================================================================

class AdamWStateDictTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

// Builds an optimizer whose constructor betas differ from the effective ones, applies the
// effective betas either through the state dict or through set_group_options, then verifies one
// step against a CPU reference driven purely by the effective betas.
static void run_effective_betas_step_and_compare(bool use_beta_setters) {
    using namespace ttml;

    const float lr = 1e-2f;
    const float epsilon = 1e-8f;
    const size_t initial_steps = 10;
    const float constructor_beta1 = 0.9f;
    const float constructor_beta2 = 0.999f;
    const float effective_beta1 = 0.5f;
    const float effective_beta2 = 0.9f;
    const std::array<std::size_t, 4> shape = {1, 1, 128, 256};

    autograd::ctx().set_seed(123U);
    auto& gen = autograd::ctx().get_generator();
    const uint32_t seed_param = gen();
    const uint32_t seed_grad = gen();
    const uint32_t seed_first_moment = gen();
    const uint32_t seed_second_moment = gen();

    xt::xarray<float> w0 = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed_param);
    xt::xarray<float> g0 = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed_grad);
    xt::xarray<float> m0 = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed_first_moment);
    xt::xarray<float> v0 = test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, seed_second_moment);

    // CPU reference driven purely by the effective betas: the behavior a resumed run must match.
    xt::xarray<float> w_cpu = w0;
    CPUAdamW cpu_opt(lr, effective_beta1, effective_beta2, epsilon, /*weight_decay=*/0.0f, /*amsgrad=*/false);
    cpu_opt.set_state(m0, v0, initial_steps);
    cpu_opt.step(w_cpu, g0);

    auto theta = autograd::create_tensor(to_tt_bf16(w0), true);
    theta->set_grad(to_tt_bf16(g0));
    serialization::NamedParameters params{{"theta", theta}};

    optimizers::AdamWConfig cfg;
    cfg.lr = lr;
    cfg.beta1 = constructor_beta1;
    cfg.beta2 = constructor_beta2;
    cfg.epsilon = epsilon;
    cfg.weight_decay = 0.0f;
    optimizers::AdamW opt(params, cfg);

    // When exercising set_group_options, the state dict carries the constructor betas so that only
    // it introduces the effective ones.
    const float state_beta1 = use_beta_setters ? constructor_beta1 : effective_beta1;
    const float state_beta2 = use_beta_setters ? constructor_beta2 : effective_beta2;
    {
        auto state = opt.get_state_dict();
        state["exp_avg"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_bf16(m0), false)}};
        state["exp_avg_sq"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_bf16(v0), false)}};
        state["steps"] = initial_steps;
        state["param_groups.0.beta1"] = state_beta1;
        state["param_groups.0.beta2"] = state_beta2;
        opt.set_state_dict(state);
    }

    if (use_beta_setters) {
        opt.set_group_options("beta1", {effective_beta1});
        opt.set_group_options("beta2", {effective_beta2});
    }

    EXPECT_EQ(opt.get_steps(), initial_steps);
    EXPECT_FLOAT_EQ(opt.get_param_groups()[0].options.at("beta1"), effective_beta1);
    EXPECT_FLOAT_EQ(opt.get_param_groups()[0].options.at("beta2"), effective_beta2);

    opt.step();

    auto result = core::to_xtensor(theta->get_value());
    auto metrics = error_metrics(w_cpu, result);

    const float mean_error_tolerance = 1e-3f;
    const float max_error_tolerance = 1e-2f;
    EXPECT_LE(metrics.mean_error, mean_error_tolerance) << "bias correction must follow the effective betas";
    EXPECT_LE(metrics.max_error, max_error_tolerance) << "bias correction must follow the effective betas";

    // Guard against a vacuous pass: a step whose moments follow the effective betas but whose
    // bias correction still carries the constructor betas' powers must be clearly distinguishable
    // from the reference.
    xt::xarray<float> m1 = effective_beta1 * m0 + (1.0f - effective_beta1) * g0;
    xt::xarray<float> v1 = effective_beta2 * v0 + (1.0f - effective_beta2) * (g0 * g0);
    const float stale_bias_correction1 =
        1.0f - std::pow(constructor_beta1, static_cast<float>(initial_steps)) * effective_beta1;
    const float stale_bias_correction2 =
        1.0f - std::pow(constructor_beta2, static_cast<float>(initial_steps)) * effective_beta2;
    xt::xarray<float> w_stale =
        w0 - lr * (m1 / stale_bias_correction1) / (xt::sqrt(v1 / stale_bias_correction2) + epsilon);
    auto stale_metrics = error_metrics(w_cpu, w_stale);
    EXPECT_GT(stale_metrics.mean_error, mean_error_tolerance)
        << "constructor and effective betas too close to distinguish; test is not meaningful";
}

TEST_F(AdamWStateDictTest, RestoredBetasDriveBiasCorrection) {
    run_effective_betas_step_and_compare(/*use_beta_setters=*/false);
}

TEST_F(AdamWStateDictTest, BetaSettersRecomputeBiasCorrection) {
    run_effective_betas_step_and_compare(/*use_beta_setters=*/true);
}

// ====================================================================
// Validation tests
// ====================================================================

using AdamWValidationTest = AdamWStateDictTest;

TEST_F(AdamWValidationTest, RejectsLogicalShapeMismatchWithEqualPadding) {
    using namespace ttml;

    // Logical 31x32 and 32x32 both round up to one 32x32 tile, so only logical-shape
    // validation can tell them apart.
    const std::array<std::size_t, 4> param_shape = {1, 1, 32, 32};
    const std::array<std::size_t, 4> grad_shape = {1, 1, 31, 32};
    auto param = to_tt_bf16(test_utils::make_uniform_xarray<float>(param_shape, -1.0F, 1.0F, 123U));
    auto grad = to_tt_bf16(test_utils::make_uniform_xarray<float>(grad_shape, -1.0F, 1.0F, 124U));
    auto exp_avg = to_tt_bf16(test_utils::make_uniform_xarray<float>(param_shape, -1.0F, 1.0F, 125U));
    auto exp_avg_sq = to_tt_bf16(test_utils::make_uniform_xarray<float>(param_shape, 0.0F, 1.0F, 126U));

    EXPECT_ANY_THROW(ttml::metal::adamw(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        /* max_exp_avg_sq */ std::nullopt,
        /* lr */ 1e-3f,
        /* beta1 */ 0.9f,
        /* beta2 */ 0.999f,
        /* beta1_pow */ 0.9f,
        /* beta2_pow */ 0.999f,
        /* epsilon */ 1e-8f,
        /* weight_decay */ 0.0f));
}

// ====================================================================
// AMSGrad Tests
// Test AdamW with AMSGrad variant enabled
// ====================================================================

// Test cases with AMSGrad enabled
static const AdamWCase kAMSGradCases[] = {
    // Standard AMSGrad
    {{1, 1, 1, 65'536}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "Standard"},
    // Disabled: non-deterministic accuracy failures — https://github.com/tenstorrent/tt-metal/issues/46121
    // {{1, 4, 64, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true, "WeightDecay_0p01"},
    // AMSGrad with different shape
    {{2, 8, 64, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "NIGHTLY_Large_4D"},
};

INSTANTIATE_TEST_SUITE_P(AdamWAMSGrad, AdamWComparisonTest, ::testing::ValuesIn(kAMSGradCases), CaseName);

// ====================================================================
// Stochastic Rounding Tests
// Test AdamW with stochastic rounding enabled
// ====================================================================

static constexpr uint32_t kSrSeed = 0x12345678U;
static constexpr uint32_t kSrRepeats = 5U;
static constexpr std::array<size_t, 4> kSrShape = {1, 1, 256, 1024};

static std::vector<uint32_t> to_bits(const ttnn::Tensor& tensor) {
    auto values = ttml::core::to_xtensor(tensor);
    std::vector<uint32_t> bits;
    bits.reserve(values.size());
    for (float v : values) {
        bits.push_back(std::bit_cast<uint32_t>(v));
    }
    return bits;
}

static size_t count_mismatches(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += static_cast<size_t>(a[i] != b[i]);
    }
    return n;
}

static xt::xarray<float> make_sine_pattern(float base, float amplitude, float freq) {
    xt::xarray<float> x = xt::zeros<float>(kSrShape);
    for (size_t i = 0; i < x.size(); ++i) {
        x.flat(i) = base + amplitude * std::sin(freq * static_cast<float>(i));
    }
    return x;
}

static std::vector<uint32_t> run_adamw_once(ttml::metal::StochasticRounding sr, uint32_t seed = kSrSeed) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto param = ttml::core::from_xtensor(make_sine_pattern(1.0F, 0.25F, 0.37F), device);
    auto grad = ttml::core::from_xtensor(make_sine_pattern(0.0F, 1.0F, 0.11F), device);
    auto exp_avg = ttml::core::zeros_like(param);
    auto exp_avg_sq = ttml::core::zeros_like(param);
    auto out = ttml::metal::adamw(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        std::nullopt,
        /*lr=*/1e-3F,
        /*beta1=*/0.9F,
        /*beta2=*/0.999F,
        /*beta1_pow=*/0.9F,
        /*beta2_pow=*/0.999F,
        /*epsilon=*/1e-8F,
        /*weight_decay=*/0.0F,
        sr,
        sr == ttml::metal::StochasticRounding::Enabled ? std::optional<uint32_t>{seed} : std::nullopt);
    return to_bits(out);
}

static void expect_repeats_identical(std::string_view name, const std::function<std::vector<uint32_t>()>& run) {
    const auto first = run();
    for (uint32_t r = 1; r < kSrRepeats; ++r) {
        const auto again = run();
        const size_t mismatches = count_mismatches(first, again);
        EXPECT_EQ(mismatches, 0U) << name << ": repeat " << r << " differs from repeat 0";
    }
}

class StochasticRoundingTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
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

    ttml::optimizers::AdamWConfig stoch_cfg;
    stoch_cfg.lr = 1e-3f;
    stoch_cfg.beta1 = 0.9f;
    stoch_cfg.beta2 = 0.999f;
    stoch_cfg.epsilon = 1e-8f;
    stoch_cfg.stochastic_rounding = true;

    ttml::optimizers::AdamW opt_stoch(params_stoch, stoch_cfg);

    auto theta_det = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_det->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_det{{"theta", theta_det}};

    ttml::optimizers::AdamWConfig det_cfg;
    det_cfg.lr = 1e-3f;
    det_cfg.beta1 = 0.9f;
    det_cfg.beta2 = 0.999f;
    det_cfg.epsilon = 1e-8f;
    det_cfg.stochastic_rounding = false;

    ttml::optimizers::AdamW opt_det(params_det, det_cfg);

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

    xt::xarray<float> w0 = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed);
    xt::xarray<float> g0 = ttml::test_utils::make_uniform_xarray<float>(shape, -0.1F, 0.1F, seed + 1);

    xt::xarray<float> w_cpu = w0;
    CPUAdamW cpu_opt(1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false);

    auto theta_stoch = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_stoch->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_stoch{{"theta", theta_stoch}};

    ttml::optimizers::AdamWConfig stoch_cfg;
    stoch_cfg.lr = 1e-3f;
    stoch_cfg.beta1 = 0.9f;
    stoch_cfg.beta2 = 0.999f;
    stoch_cfg.epsilon = 1e-8f;
    stoch_cfg.stochastic_rounding = true;
    ttml::optimizers::AdamW opt_stoch(params_stoch, stoch_cfg);

    auto theta_det = autograd::create_tensor(to_tt_bf16(w0), true);
    theta_det->set_grad(to_tt_bf16(g0));
    ttml::serialization::NamedParameters params_det{{"theta", theta_det}};

    ttml::optimizers::AdamWConfig det_cfg;
    det_cfg.lr = 1e-3f;
    det_cfg.beta1 = 0.9f;
    det_cfg.beta2 = 0.999f;
    det_cfg.epsilon = 1e-8f;
    det_cfg.stochastic_rounding = false;
    ttml::optimizers::AdamW opt_det(params_det, det_cfg);

    for (uint32_t i = 0; i < steps; ++i) {
        cpu_opt.step(w_cpu, g0);
        opt_stoch.step();
        opt_det.step();
    }

    auto result_stoch = core::to_xtensor(theta_stoch->get_value());
    auto result_det = core::to_xtensor(theta_det->get_value());

    auto stoch_metrics = error_metrics(w_cpu, result_stoch);
    auto det_metrics = error_metrics(w_cpu, result_det);

    EXPECT_LT(stoch_metrics.mean_error, det_metrics.mean_error);
    EXPECT_LT(stoch_metrics.max_error, det_metrics.max_error);
}

TEST_F(StochasticRoundingTest, DisabledIsBitIdentical) {
    expect_repeats_identical("sr_off", [] { return run_adamw_once(ttml::metal::StochasticRounding::Disabled); });
}

TEST_F(StochasticRoundingTest, SameSeedIsBitIdentical) {
    expect_repeats_identical("sr_on", [] { return run_adamw_once(ttml::metal::StochasticRounding::Enabled); });
}

TEST_F(StochasticRoundingTest, DifferentSeedDiffers) {
    const auto a = run_adamw_once(ttml::metal::StochasticRounding::Enabled, kSrSeed);
    const auto b = run_adamw_once(ttml::metal::StochasticRounding::Enabled, kSrSeed + 1U);
    const auto off = run_adamw_once(ttml::metal::StochasticRounding::Disabled);
    const size_t seed_mismatches = count_mismatches(a, b);
    const size_t off_mismatches = count_mismatches(a, off);
    SCOPED_TRACE(fmt::format("mismatches seed vs seed+1={} seed vs off={}", seed_mismatches, off_mismatches));
    EXPECT_GT(seed_mismatches, 0U) << "a different seed must change the rounding";
    EXPECT_GT(off_mismatches, 0U) << "stochastic rounding must differ from round-to-nearest";
}

// ====================================================================
// param_group_defaults: a config's float fields are its per-group options
// ====================================================================

TEST(ParamGroupDefaultsTest, FloatFieldsArePerGroupOptions) {
    using namespace ttml::optimizers;
    auto keys = [](const ParamGroupOptions& options) {
        std::vector<std::string> names;
        for (const auto& [name, value] : options) {
            names.push_back(name);
        }
        std::ranges::sort(names);
        return names;
    };
    const std::vector<std::string> adamw_keys = {"beta1", "beta2", "epsilon", "lr", "weight_decay"};
    EXPECT_EQ(keys(param_group_defaults(AdamWConfig{})), adamw_keys);  // not amsgrad / stochastic_rounding
    EXPECT_EQ(keys(param_group_defaults(AdamWFullPrecisionConfig{})), adamw_keys);
    EXPECT_EQ(keys(param_group_defaults(AdamWCompositeConfig{})), adamw_keys);  // not kahan_summation
    const std::vector<std::string> sgd_keys = {"dampening", "lr", "momentum", "weight_decay"};
    EXPECT_EQ(keys(param_group_defaults(SGDConfig{})), sgd_keys);  // not nesterov
    EXPECT_EQ(keys(param_group_defaults(SGDCompositeConfig{})), sgd_keys);
    EXPECT_EQ(keys(param_group_defaults(MuonConfig{})), (std::vector<std::string>{"lr", "momentum"}));  // not ns_steps

    AdamWConfig config;
    config.lr = 0.123F;
    config.weight_decay = 0.5F;
    const auto defaults = param_group_defaults(config);
    EXPECT_FLOAT_EQ(defaults.at("lr"), 0.123F);
    EXPECT_FLOAT_EQ(defaults.at("weight_decay"), 0.5F);
}

// ====================================================================
// Program-cache hits
// lr, the betas, the beta powers, epsilon, weight decay and the stochastic-rounding seed are not part of the program
// hash, so a hit must re-apply them with the new buffer addresses. Each test starts from an empty program cache and
// keeps the tensors of its earlier launches alive, so a hit cannot land on a freed address.
// ====================================================================

class AdamWCacheHitTest : public ttml::test_utils::ProgramCacheSuiteTest {};

namespace {

using ttml::test_utils::CacheExpectation;
using ttml::test_utils::count_bit_mismatches;
using ttml::test_utils::expect_cache;
using ttml::test_utils::same_bits;
using namespace ttml::test_utils::adamw;

// Without weight decay the kernel skips the decay multiply. Launch B changes every scalar of launch A, each by enough
// that a hit keeping launch A's value of it alone misses the bf16 tolerance; hence the large lr, epsilon and decay, and
// a beta2 far enough from launch A's to show through AMSGrad's max with max_exp_avg_sq.
const AdamWStepScalars kScalarsA = step_scalars(1e-3F, 0.9F, 0.999F, 10U, 1e-8F, 0.0F);
const AdamWStepScalars kScalarsB = step_scalars(0.1F, 0.5F, 0.5F, 3U, 0.1F, 0.2F);
// AMSGrad keeps weight decay at 0, as kAMSGradCases does (#46121).
const AdamWStepScalars kAmsgradScalarsB = step_scalars(0.1F, 0.5F, 0.5F, 3U, 0.1F, 0.0F);

// run_step_and_compare's tolerances.
constexpr float kMeanErrorTolerance = 1e-3f;
constexpr float kMaxErrorTolerance = 1e-2f;

constexpr size_t kTileElements = 32U * 32U;

// The kernels read the host values rounded to bf16.
StoredTensor store_bf16(const xt::xarray<float>& values) {
    return store(to_tt_bf16(values));
}

// The tensors of one launch, drawn as run_step_and_compare draws them.
AdamWLaunchTensors make_adamw_launch_tensors(const std::vector<uint32_t>& shape, bool amsgrad, uint32_t seed) {
    using ttml::test_utils::make_uniform_xarray;
    AdamWLaunchTensors tensors{
        store_bf16(make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed)),
        store_bf16(make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed + 1U)),
        store_bf16(make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed + 2U)),
        store_bf16(make_uniform_xarray<float>(shape, 0.0F, 1.0F, seed + 3U)),
        std::nullopt};
    if (amsgrad) {
        tensors.max_exp_avg_sq = store_bf16(make_uniform_xarray<float>(shape, 0.0F, 1.0F, seed + 4U));
    }
    return tensors;
}

// CPUAdamW's step from the values the launch read.
AdamWValues adamw_reference(const AdamWLaunchTensors& tensors, const AdamWStepScalars& scalars) {
    const bool amsgrad = tensors.max_exp_avg_sq.has_value();
    CPUAdamW cpu_opt(scalars.lr, scalars.beta1, scalars.beta2, scalars.epsilon, scalars.weight_decay, amsgrad);
    // The step count only feeds the bias corrections, which take the launch's beta powers instead.
    if (amsgrad) {
        cpu_opt.set_state(tensors.exp_avg.before, tensors.exp_avg_sq.before, 0U, tensors.max_exp_avg_sq->before);
    } else {
        cpu_opt.set_state(tensors.exp_avg.before, tensors.exp_avg_sq.before, 0U);
    }
    cpu_opt.set_beta_powers(scalars.beta1_pow, scalars.beta2_pow);
    xt::xarray<float> param = tensors.param.before;
    cpu_opt.step(param, tensors.grad.before);
    AdamWValues values{
        std::move(param), tensors.grad.before, cpu_opt.first_moment(), cpu_opt.second_moment(), std::nullopt};
    if (amsgrad) {
        values.max_exp_avg_sq = cpu_opt.max_second_moment();
    }
    return values;
}

// Elements of two bf16 tensors that are neither equal nor neighbouring bf16 values.
size_t count_non_neighbouring_bf16(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto a_bf16 = static_cast<int32_t>(std::bit_cast<uint32_t>(a.flat(i)) >> 16U);
        const auto b_bf16 = static_cast<int32_t>(std::bit_cast<uint32_t>(b.flat(i)) >> 16U);
        count += static_cast<size_t>(std::abs(a_bf16 - b_bf16) > 1);
    }
    return count;
}

// Each core updates a run of consecutive tiles, in the order of the core walk: one tile more on each of the first
// `tiles % cores` cores (core group 1) than on the others (core group 2). In a parameter one tile wide, tile t holds
// elements [1024 t, 1024 t + 1024).
void expect_every_core_differs(const xt::xarray<float>& a, const xt::xarray<float>& b, const std::string& label) {
    ASSERT_EQ(a.shape(), b.shape()) << label;
    ASSERT_EQ(a.shape().back(), 32U) << label << ": the parameter must be one tile wide";
    const size_t tiles = a.size() / kTileElements;
    ASSERT_GT(tiles, 0U) << label;
    const size_t cores = std::min<size_t>(tiles, ttml::test_utils::num_worker_cores());
    const size_t group_1_cores = tiles % cores;
    size_t tile = 0;
    for (size_t core = 0; core < cores; ++core) {
        const size_t core_tiles = tiles / cores + (core < group_1_cores ? 1U : 0U);
        size_t mismatches = 0;
        for (size_t i = tile * kTileElements; i < (tile + core_tiles) * kTileElements; ++i) {
            mismatches += static_cast<size_t>(!same_bits(a.flat(i), b.flat(i)));
        }
        EXPECT_GT(mismatches, 0U) << label << ": core " << core << " (core group "
                                  << (group_1_cores == 0U || core < group_1_cores ? 1 : 2) << ")";
        tile += core_tiles;
    }
}

// The tolerances of run_step_and_compare, for every tensor the launch updates.
void expect_close_to_reference(const AdamWValues& expected, const AdamWValues& actual, const std::string& label) {
    auto expect_close = [&](const xt::xarray<float>& reference, const xt::xarray<float>& result, const char* name) {
        ASSERT_EQ(reference.shape(), result.shape()) << label << ": " << name;
        const auto metrics = error_metrics(reference, result);
        EXPECT_LE(metrics.mean_error, kMeanErrorTolerance) << label << ": " << name;
        EXPECT_LE(metrics.max_error, kMaxErrorTolerance) << label << ": " << name;
    };
    expect_close(expected.param, actual.param, "param");
    expect_close(expected.exp_avg, actual.exp_avg, "exp_avg");
    expect_close(expected.exp_avg_sq, actual.exp_avg_sq, "exp_avg_sq");
    ASSERT_EQ(expected.max_exp_avg_sq.has_value(), actual.max_exp_avg_sq.has_value()) << label;
    if (expected.max_exp_avg_sq.has_value()) {
        expect_close(*expected.max_exp_avg_sq, *actual.max_exp_avg_sq, "max_exp_avg_sq");
    }
    EXPECT_EQ(count_bit_mismatches(expected.grad, actual.grad), 0U) << label << ": the gradient must not change";
}

// Guards against a vacuous pass: a hit that kept any one of the cached program's scalars must miss the reference by
// more than the mean tolerance.
void expect_each_stale_scalar_detectable(
    const AdamWLaunchTensors& tensors, const AdamWStepScalars& scalars, const AdamWStepScalars& cached) {
    const auto expected = adamw_reference(tensors, scalars);
    for (const auto& [name, stale] : stale_scalar_variants(scalars, cached)) {
        const auto metrics = error_metrics(expected.param, adamw_reference(tensors, stale).param);
        EXPECT_GT(metrics.mean_error, kMeanErrorTolerance) << "a stale " << name << " is too close to tell apart";
    }
}

}  // namespace

// Launch A compiles the program. Launch B updates fresh states with every hash-excluded scalar changed. Launch C
// updates fresh copies of launch A's states with launch A's scalars and must reproduce launch A bit for bit.
TEST_F(AdamWCacheHitTest, NewScalarsOnFreshStates_TwoCoreGroups) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto shape = ttml::test_utils::two_core_group_rows_shape(32U);
    constexpr uint32_t kSeedA = 1000U;

    const auto a = make_adamw_launch_tensors(shape, /*amsgrad=*/false, kSeedA);
    const auto entries_before_a = device.num_program_cache_entries();
    const auto a_out = launch_adamw(a, kScalarsA);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_a) << "launch A must compile a program";
    EXPECT_EQ(a_out.buffer()->address(), a.param.tensor.buffer()->address()) << "AdamW updates the parameter in place";
    const auto a_after = read_back(a);
    expect_close_to_reference(adamw_reference(a, kScalarsA), a_after, "launch A");

    const auto b = make_adamw_launch_tensors(shape, /*amsgrad=*/false, 2000U);
    const auto entries_before_b = device.num_program_cache_entries();
    launch_adamw(b, kScalarsB);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_b) << "launch B must reuse launch A's program";
    expect_distinct_buffers(a, b);
    const auto b_after = read_back(b);
    expect_close_to_reference(adamw_reference(b, kScalarsB), b_after, "launch B (cache hit, new scalars)");
    expect_each_stale_scalar_detectable(b, kScalarsB, kScalarsA);

    const auto c = make_adamw_launch_tensors(shape, /*amsgrad=*/false, kSeedA);
    const auto entries_before_c = device.num_program_cache_entries();
    launch_adamw(c, kScalarsA);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    expect_distinct_buffers(a, c);
    expect_distinct_buffers(b, c);
    expect_bitwise_equal(a_after, read_back(c), "launch C (launch A again, cache hit)");

    expect_bitwise_equal(a_after, read_back(a), "launch A's buffers after the later launches");
    expect_bitwise_equal(b_after, read_back(b), "launch B's buffers after launch C");
}

// AMSGrad reads and writes a fifth tensor, max_exp_avg_sq, and has a program of its own. The launches alternate
// between the two cached programs on fresh states, with new scalars on every hit.
TEST_F(AdamWCacheHitTest, AmsgradAndPlainProgramsAlternate_TwoCoreGroups) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto shape = ttml::test_utils::two_core_group_rows_shape(32U);

    struct Launch {
        bool amsgrad;
        AdamWStepScalars scalars;
        bool cache_hit;
        const char* name;
    };
    const std::array<Launch, 4> launches = {{
        {false, kScalarsA, false, "launch A (plain, miss)"},
        {true, kScalarsA, false, "launch B (AMSGrad, miss)"},
        {false, kScalarsB, true, "launch C (plain, cache hit)"},
        {true, kAmsgradScalarsB, true, "launch D (AMSGrad, cache hit)"},
    }};

    std::vector<AdamWLaunchTensors> tensors;
    std::vector<AdamWValues> after;
    tensors.reserve(launches.size());
    after.reserve(launches.size());
    for (uint32_t i = 0; i < launches.size(); ++i) {
        const auto& launch = launches[i];
        auto launch_tensors = make_adamw_launch_tensors(shape, launch.amsgrad, 3000U + 10U * i);
        for (const auto& earlier : tensors) {
            expect_distinct_buffers(earlier, launch_tensors);
        }
        const auto entries_before = device.num_program_cache_entries();
        launch_adamw(launch_tensors, launch.scalars);
        expect_cache(entries_before, launch.cache_hit ? CacheExpectation::Hit : CacheExpectation::Miss, launch.name);
        after.push_back(read_back(launch_tensors));
        expect_close_to_reference(adamw_reference(launch_tensors, launch.scalars), after.back(), launch.name);
        if (launch.cache_hit) {
            // Both programs were compiled with kScalarsA.
            expect_each_stale_scalar_detectable(launch_tensors, launch.scalars, kScalarsA);
        }
        tensors.push_back(std::move(launch_tensors));
    }
    for (uint32_t i = 0; i < launches.size(); ++i) {
        expect_bitwise_equal(
            after[i], read_back(tensors[i]), std::string(launches[i].name) + ": buffers after the later launches");
    }
}

// Every core seeds its stochastic rounding from the call's seed. Launches A, C and D update fresh copies of the same
// states with the same scalars and seeds kSrSeed, kSrSeed + 1 and kSrSeed. The moments do not depend on the seed and a
// seed can only round the parameter to one of the two bf16 neighbours of the fp32 update, so launch C must differ from
// launch A only in that choice, on the tiles of every core, and launch D must reproduce launch A bit for bit. Launch B
// runs without stochastic rounding, which the program hash keys, so it must compile a program of its own.
TEST_F(AdamWCacheHitTest, StochasticRoundingSeedOnCacheHit_TwoCoreGroups) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto shape = ttml::test_utils::two_core_group_rows_shape(32U);
    constexpr uint32_t kDataSeed = 5000U;
    constexpr auto kEnabled = ttml::metal::StochasticRounding::Enabled;

    const auto a = make_adamw_launch_tensors(shape, /*amsgrad=*/false, kDataSeed);
    const auto entries_before_a = device.num_program_cache_entries();
    launch_adamw(a, kScalarsA, kEnabled, kSrSeed);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_a) << "launch A must compile a program";
    const auto a_after = read_back(a);
    expect_close_to_reference(adamw_reference(a, kScalarsA), a_after, "launch A (stochastic rounding)");

    const auto b = make_adamw_launch_tensors(shape, /*amsgrad=*/false, kDataSeed);
    const auto entries_before_b = device.num_program_cache_entries();
    launch_adamw(b, kScalarsA);
    EXPECT_GT(device.num_program_cache_entries(), entries_before_b)
        << "launch B (no stochastic rounding) must compile a program of its own";
    expect_distinct_buffers(a, b);
    const auto b_after = read_back(b);
    expect_close_to_reference(adamw_reference(b, kScalarsA), b_after, "launch B (round to nearest)");

    const auto c = make_adamw_launch_tensors(shape, /*amsgrad=*/false, kDataSeed);
    const auto entries_before_c = device.num_program_cache_entries();
    launch_adamw(c, kScalarsA, kEnabled, kSrSeed + 1U);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    expect_distinct_buffers(a, c);
    expect_distinct_buffers(b, c);
    const auto c_after = read_back(c);
    expect_close_to_reference(adamw_reference(c, kScalarsA), c_after, "launch C (stochastic rounding, cache hit)");
    EXPECT_EQ(count_bit_mismatches(a_after.exp_avg, c_after.exp_avg), 0U) << "launch C: exp_avg";
    EXPECT_EQ(count_bit_mismatches(a_after.exp_avg_sq, c_after.exp_avg_sq), 0U) << "launch C: exp_avg_sq";
    expect_every_core_differs(a_after.param, c_after.param, "launch C: a new seed must change the rounding on a hit");
    EXPECT_EQ(count_non_neighbouring_bf16(a_after.param, c_after.param), 0U)
        << "launch C: both seeds must round the same fp32 update";

    const auto d = make_adamw_launch_tensors(shape, /*amsgrad=*/false, kDataSeed);
    const auto entries_before_d = device.num_program_cache_entries();
    launch_adamw(d, kScalarsA, kEnabled, kSrSeed);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_d) << "launch D must reuse launch A's program";
    expect_distinct_buffers(a, d);
    expect_distinct_buffers(b, d);
    expect_distinct_buffers(c, d);
    expect_bitwise_equal(a_after, read_back(d), "launch D (seed of launch A, cache hit)");

    expect_bitwise_equal(a_after, read_back(a), "launch A's buffers after the later launches");
    expect_bitwise_equal(b_after, read_back(b), "launch B's buffers after the later launches");
    expect_bitwise_equal(c_after, read_back(c), "launch C's buffers after launch D");
}
