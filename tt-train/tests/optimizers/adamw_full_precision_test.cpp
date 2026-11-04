// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/adamw_full_precision.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "adamw_test_utils.hpp"
#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include "xtensor/core/xtensor_forward.hpp"

using ttml::test_utils::adamw::error_metrics;

struct AdamWFullPrecisionCase {
    std::array<std::size_t, 4> shape;  // (B, H, S, C)
    float lr{1e-3f};
    float beta1{0.9f};
    float beta2{0.999f};
    float epsilon{1e-8f};
    float weight_decay{0.0f};
    bool amsgrad{false};
    std::string name;
};

// Custom printer for AdamWFullPrecisionCase used by gtest to make test output readable
void PrintTo(const AdamWFullPrecisionCase& pc, std::ostream* os) {
    *os << fmt::format(
        "AdamWFullPrecisionCase(name='{}', shape=[{},{},{},{}], lr={}, beta1={}, beta2={}, eps={}, wd={}, amsgrad={})",
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

class AdamWFullPrecisionComparisonTest : public ::testing::TestWithParam<AdamWFullPrecisionCase> {
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

static ttnn::Tensor to_tt_bf16(const xt::xarray<bfloat16>& x) {
    return ttml::core::from_xtensor<bfloat16, ttnn::DataType::BFLOAT16>(x, &ttml::autograd::ctx().get_device());
}

// NOTE: ttml::core::from_xtensor() performs fp32 -> bf16 conversion, that's why this function exists
static ttnn::Tensor to_tt_fp32(const xt::xarray<float>& x) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttnn::experimental::xtensor::get_shape_from_xarray(x);
    auto buffer_span = ttml::core::xtensor_to_span(x);

    ttnn::MemoryConfig output_mem_config{};
    const auto tensor_layout = tt::tt_metal::TensorLayout(
        ttnn::DataType::FLOAT32, ttnn::PageConfig(ttnn::Layout::ROW_MAJOR), tt::tt_metal::MemoryConfig{});
    auto output = ttnn::Tensor::from_span<float>(buffer_span, tt::tt_metal::TensorSpec(shape, tensor_layout));

    output = ttnn::to_layout(output, ttnn::Layout::TILE, std::nullopt, output_mem_config);
    output = ttnn::to_device(output, device, output_mem_config);
    return output;
}

// CPU reference implementation of AdamW with fp32 master weights
class CPUAdamWFullPrecision {
public:
    CPUAdamWFullPrecision(
        float lr, float beta1, float beta2, float epsilon, float weight_decay = 0.0f, bool amsgrad = false) :
        m_lr(lr),
        m_beta1(beta1),
        m_beta2(beta2),
        m_epsilon(epsilon),
        m_weight_decay(weight_decay),
        m_amsgrad(amsgrad),
        m_steps(0) {
    }

    // Set initial optimizer state for testing
    void set_state(
        const xt::xarray<float>& master_weights,
        const xt::xarray<float>& first_moment,
        const xt::xarray<float>& second_moment,
        size_t steps,
        const xt::xarray<float>& max_second_moment = {}) {
        m_master_weights = master_weights;
        m_first_moment = first_moment;
        m_second_moment = second_moment;
        m_steps = steps;
        if (m_amsgrad && max_second_moment.size() > 0) {
            m_max_second_moment = max_second_moment;
        } else if (m_amsgrad) {
            m_max_second_moment = xt::zeros_like(master_weights);
        }
    }

    // Bias-correct the next step with these beta powers, as the kernel takes them, instead of beta^t.
    void set_beta_powers(float beta1_pow, float beta2_pow) {
        m_beta_powers = {beta1_pow, beta2_pow};
    }

    // Perform one optimization step
    // grads should be bf16 precision (truncated)
    xt::xarray<float> step(const xt::xarray<float>& grads) {
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

        // For AMSGrad: use max of past squared gradients
        xt::xarray<float> denom;
        if (m_amsgrad) {
            m_max_second_moment = xt::maximum(m_max_second_moment, m_second_moment);
            denom = xt::sqrt(m_max_second_moment / bias_correction2) + m_epsilon;
        } else {
            denom = xt::sqrt(m_second_moment / bias_correction2) + m_epsilon;
        }

        // Update master weights (fp32): params = params - lr * m_hat / denom - lr * weight_decay * params
        m_master_weights =
            m_master_weights - m_lr * first_moment_hat / denom - m_lr * m_weight_decay * m_master_weights;

        return m_master_weights;
    }

    const xt::xarray<float>& get_master_weights() const {
        return m_master_weights;
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
    xt::xarray<float> m_master_weights;
    xt::xarray<float> m_first_moment;
    xt::xarray<float> m_second_moment;
    xt::xarray<float> m_max_second_moment;  // amsgrad
};

static void run_steps_and_compare(const AdamWFullPrecisionCase& pc, uint32_t steps) {
    using namespace ttml;

    ttml::autograd::ctx().set_seed(123U);
    auto& g = autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();
    const uint32_t seed_first_moment = g();
    const uint32_t seed_second_moment = g();
    const uint32_t seed_max_second_moment = g();

    // Generate random data
    xt::xarray<float> w0_fp32 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_param);
    xt::xarray<float> m0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_first_moment);
    xt::xarray<float> v0 =
        ttml::test_utils::make_uniform_xarray<float>(pc.shape, 0.0F, 1.0F, seed_second_moment);  // must be >= 0
    xt::xarray<float> max_v0 =
        ttml::test_utils::make_uniform_xarray<float>(pc.shape, 0.0F, 1.0F, seed_max_second_moment);

    // Generate gradient directly as bfloat16
    xt::xarray<bfloat16> g0_bf16 =
        ttml::test_utils::make_uniform_xarray<bfloat16>(pc.shape, bfloat16{-1.0F}, bfloat16{1.0F}, seed_grad);
    auto g0_bf16_tt = to_tt_bf16(g0_bf16);

    // Initial step count (non-zero to test bias correction with accumulated steps)
    const size_t initial_steps = 10;

    // CPU reference implementation with fp32 master weights
    CPUAdamWFullPrecision cpu_opt(pc.lr, pc.beta1, pc.beta2, pc.epsilon, pc.weight_decay, pc.amsgrad);
    cpu_opt.set_state(w0_fp32, m0, v0, initial_steps, pc.amsgrad ? max_v0 : xt::xarray<float>{});

    // Create theta parameter with bf16 weights
    auto theta = autograd::create_tensor(ttml::core::from_xtensor(w0_fp32, &ttml::autograd::ctx().get_device()), true);
    ttml::serialization::NamedParameters params{{"theta", theta}};

    ttml::optimizers::AdamWFullPrecisionConfig cfg;
    cfg.lr = pc.lr;
    cfg.beta1 = pc.beta1;
    cfg.beta2 = pc.beta2;
    cfg.epsilon = pc.epsilon;
    cfg.weight_decay = pc.weight_decay;
    cfg.amsgrad = pc.amsgrad;

    ttml::optimizers::AdamWFullPrecision opt(params, cfg);

    // Inject optimizer state on top of the optimizer's own state dict, which carries its param groups.
    auto state = opt.get_state_dict();
    state["steps"] = initial_steps;
    state["master_weights"] =
        serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(w0_fp32), false)}};
    state["exp_avg"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(m0), false)}};
    state["exp_avg_sq"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(v0), false)}};
    if (pc.amsgrad) {
        state["max_exp_avg_sq"] =
            serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(max_v0), false)}};
    }
    opt.set_state_dict(state);

    // Run both optimizers for the specified number of steps
    for (uint32_t i = 0; i < steps; ++i) {
        theta->set_grad(g0_bf16_tt);
        cpu_opt.step(xt::cast<float>(g0_bf16));
        opt.step();
    }

    // Compare master weights
    const auto& cpu_master_weights = cpu_opt.get_master_weights();
    const auto& device_master_weights = opt.get_master_weights();
    auto master_weights_tensor = device_master_weights.at("theta")->get_value(autograd::PreferredPrecision::FULL);
    auto device_master_weights_cpu = core::to_xtensor(master_weights_tensor);

    float sum_error = 0.0f;
    float max_error = 0.0f;
    for (size_t i = 0; i < cpu_master_weights.size(); ++i) {
        float error = std::abs(cpu_master_weights.flat(i) - device_master_weights_cpu.flat(i));
        sum_error += error;
        max_error = std::max(max_error, error);
    }
    float mean_error = sum_error / static_cast<float>(cpu_master_weights.size());

    const float mean_error_tolerance = 1e-7f;
    const float max_error_tolerance = 1e-6f;

    EXPECT_LT(mean_error, mean_error_tolerance);
    EXPECT_LT(max_error, max_error_tolerance);
}

static std::string CaseName(const ::testing::TestParamInfo<AdamWFullPrecisionCase>& info) {
    const auto& c = info.param;
    return fmt::format("{}_B{}H{}S{}C{}", c.name, c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
}

TEST_P(AdamWFullPrecisionComparisonTest, CompareWithCPU) {
    const auto& pc = GetParam();
    // Single step with pre-initialized momentum states for rigorous testing
    const uint32_t steps = 1;
    run_steps_and_compare(pc, steps);
}

// Test cases with various hyperparameter configurations
static const AdamWFullPrecisionCase kBasicCases[] = {
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
    AdamWFullPrecisionBasicComparison, AdamWFullPrecisionComparisonTest, ::testing::ValuesIn(kBasicCases), CaseName);

// ====================================================================
// Weight Decay Tests
// Test AdamWFullPrecision with various weight decay configurations
// ====================================================================

static const AdamWFullPrecisionCase kWeightDecayCases[] = {
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
    AdamWFullPrecisionWeightDecay, AdamWFullPrecisionComparisonTest, ::testing::ValuesIn(kWeightDecayCases), CaseName);

// ====================================================================
// AMSGrad Tests
// Test AdamWFullPrecision with AMSGrad variant enabled
// ====================================================================

static const AdamWFullPrecisionCase kAMSGradCases[] = {
    // Standard AMSGrad
    {{1, 1, 1, 65'536}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "Standard"},
    // Disabled: non-deterministic accuracy failures — https://github.com/tenstorrent/tt-metal/issues/46121
    // {{1, 4, 64, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true, "WeightDecay_0p01"},
    // AMSGrad with different shape
    {{2, 8, 64, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "NIGHTLY_Large_4D"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFullPrecisionAMSGrad, AdamWFullPrecisionComparisonTest, ::testing::ValuesIn(kAMSGradCases), CaseName);

// ====================================================================
// State-dict restore tests
// A checkpoint's betas may differ from the constructor config; both the
// moment updates and the bias correction (beta powers) must follow the
// restored betas.
// ====================================================================

class AdamWFullPrecisionStateDictTest : public ::testing::Test {
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
    xt::xarray<bfloat16> g0_bf16 =
        test_utils::make_uniform_xarray<bfloat16>(shape, bfloat16{-1.0F}, bfloat16{1.0F}, seed_grad);
    xt::xarray<float> g0 = xt::cast<float>(g0_bf16);
    xt::xarray<float> m0 = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed_first_moment);
    xt::xarray<float> v0 = test_utils::make_uniform_xarray<float>(shape, 0.0F, 1.0F, seed_second_moment);

    // CPU reference driven purely by the effective betas: the behavior a resumed run must match.
    CPUAdamWFullPrecision cpu_opt(lr, effective_beta1, effective_beta2, epsilon, /*weight_decay=*/0.0f);
    cpu_opt.set_state(w0, m0, v0, initial_steps);
    xt::xarray<float> w_cpu = cpu_opt.step(g0);

    auto theta = autograd::create_tensor(core::from_xtensor(w0, &autograd::ctx().get_device()), true);
    serialization::NamedParameters params{{"theta", theta}};

    optimizers::AdamWFullPrecisionConfig cfg;
    cfg.lr = lr;
    cfg.beta1 = constructor_beta1;
    cfg.beta2 = constructor_beta2;
    cfg.epsilon = epsilon;
    cfg.weight_decay = 0.0f;
    optimizers::AdamWFullPrecision opt(params, cfg);

    // When exercising set_group_options, the state dict carries the constructor betas so that only
    // it introduces the effective ones.
    const float state_beta1 = use_beta_setters ? constructor_beta1 : effective_beta1;
    const float state_beta2 = use_beta_setters ? constructor_beta2 : effective_beta2;
    {
        auto state = opt.get_state_dict();
        state["steps"] = initial_steps;
        state["param_groups.0.beta1"] = state_beta1;
        state["param_groups.0.beta2"] = state_beta2;
        state["master_weights"] =
            serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(w0), false)}};
        state["exp_avg"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(m0), false)}};
        state["exp_avg_sq"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(v0), false)}};
        opt.set_state_dict(state);
    }

    if (use_beta_setters) {
        opt.set_group_options("beta1", {effective_beta1});
        opt.set_group_options("beta2", {effective_beta2});
    }

    EXPECT_EQ(opt.get_steps(), initial_steps);
    EXPECT_FLOAT_EQ(opt.get_param_groups()[0].options.at("beta1"), effective_beta1);
    EXPECT_FLOAT_EQ(opt.get_param_groups()[0].options.at("beta2"), effective_beta2);

    theta->set_grad(to_tt_bf16(g0_bf16));
    opt.step();

    const auto& device_master_weights = opt.get_master_weights();
    auto master_weights_tensor = device_master_weights.at("theta")->get_value(autograd::PreferredPrecision::FULL);
    auto actual = core::to_xtensor(master_weights_tensor);

    const float mean_error_tolerance = 1e-7f;
    const float max_error_tolerance = 1e-6f;
    auto [mean_error, max_error] = error_metrics(w_cpu, actual);
    EXPECT_LE(mean_error, mean_error_tolerance) << "bias correction must follow the effective betas";
    EXPECT_LE(max_error, max_error_tolerance) << "bias correction must follow the effective betas";

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
    auto [stale_mean_error, stale_max_error] = error_metrics(w_cpu, w_stale);
    EXPECT_GT(stale_mean_error, mean_error_tolerance)
        << "constructor and effective betas too close to distinguish; test is not meaningful";
}

TEST_F(AdamWFullPrecisionStateDictTest, RestoredBetasDriveBiasCorrection) {
    run_effective_betas_step_and_compare(/*use_beta_setters=*/false);
}

TEST_F(AdamWFullPrecisionStateDictTest, BetaSettersRecomputeBiasCorrection) {
    run_effective_betas_step_and_compare(/*use_beta_setters=*/true);
}

// ====================================================================
// Program-cache hits
// lr, the betas, the beta powers, epsilon and weight decay are not part of the program hash, so a hit must re-apply
// them with the new buffer addresses; at fp32 precision each of them moves the update far past the tolerance. Each
// test starts from an empty program cache and keeps the tensors of its earlier launches alive.
// ====================================================================

class AdamWFullPrecisionCacheHitTest : public ttml::test_utils::ProgramCacheSuiteTest {};

namespace {

using ttml::test_utils::count_bit_mismatches;
using ttml::test_utils::two_core_group_rows_shape;
using namespace ttml::test_utils::adamw;

// Without weight decay the kernel skips the decay multiply; launch B changes every scalar of launch A.
const AdamWStepScalars kScalarsA = step_scalars(1e-3F, 0.9F, 0.999F, 10U, 1e-8F, 0.0F);
const AdamWStepScalars kScalarsB = step_scalars(1e-2F, 0.5F, 0.9F, 3U, 1e-3F, 0.1F);
// AMSGrad keeps weight decay at 0, as kAMSGradCases does (#46121).
const AdamWStepScalars kAmsgradScalarsB = step_scalars(1e-2F, 0.5F, 0.9F, 3U, 1e-3F, 0.0F);

// The parameter's mean and max error tolerances; the moments take the max one.
struct Tolerances {
    float mean_error;
    float max_error;
};
// run_steps_and_compare's tolerances, and the bf16 ones of AdamWComparisonTest.
constexpr Tolerances kFp32Tolerances{1e-7f, 1e-6f};
constexpr Tolerances kBf16Tolerances{1e-3f, 1e-2f};

// The tensors of one launch as AdamWFullPrecision passes them: fp32 parameter and states and a bf16 gradient. With
// bf16 parameter and states the same call runs the op's bf16 program.
AdamWLaunchTensors make_launch_tensors(
    const std::vector<uint32_t>& shape,
    bool amsgrad,
    uint32_t seed,
    ttnn::DataType state_dtype = ttnn::DataType::FLOAT32) {
    using ttml::test_utils::make_uniform_xarray;
    auto upload = [state_dtype](const xt::xarray<float>& values) {
        return store(
            state_dtype == ttnn::DataType::FLOAT32
                ? to_tt_fp32(values)
                : ttml::core::from_xtensor(values, &ttml::autograd::ctx().get_device()));
    };
    AdamWLaunchTensors tensors{
        upload(make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed)),
        store(to_tt_bf16(make_uniform_xarray<bfloat16>(shape, bfloat16{-1.0F}, bfloat16{1.0F}, seed + 1U))),
        upload(make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed + 2U)),
        upload(make_uniform_xarray<float>(shape, 0.0F, 1.0F, seed + 3U)),
        std::nullopt};
    if (amsgrad) {
        tensors.max_exp_avg_sq = upload(make_uniform_xarray<float>(shape, 0.0F, 1.0F, seed + 4U));
    }
    return tensors;
}

// CPUAdamWFullPrecision's step from the values the launch read.
AdamWValues adamw_reference(const AdamWLaunchTensors& tensors, const AdamWStepScalars& scalars) {
    const bool amsgrad = tensors.max_exp_avg_sq.has_value();
    CPUAdamWFullPrecision cpu_opt(
        scalars.lr, scalars.beta1, scalars.beta2, scalars.epsilon, scalars.weight_decay, amsgrad);
    // The step count only feeds the bias corrections, which take the launch's beta powers instead.
    if (amsgrad) {
        cpu_opt.set_state(
            tensors.param.before,
            tensors.exp_avg.before,
            tensors.exp_avg_sq.before,
            0U,
            tensors.max_exp_avg_sq->before);
    } else {
        cpu_opt.set_state(tensors.param.before, tensors.exp_avg.before, tensors.exp_avg_sq.before, 0U);
    }
    cpu_opt.set_beta_powers(scalars.beta1_pow, scalars.beta2_pow);
    xt::xarray<float> param = cpu_opt.step(tensors.grad.before);
    AdamWValues values{
        std::move(param), tensors.grad.before, cpu_opt.first_moment(), cpu_opt.second_moment(), std::nullopt};
    if (amsgrad) {
        values.max_exp_avg_sq = cpu_opt.max_second_moment();
    }
    return values;
}

// The moments are computed outright rather than as a small update of a larger value, so they carry the kernel's
// rounding at full scale and take only the max tolerance; a stale address or a stale beta moves them far more.
void expect_close_to_reference(
    const AdamWValues& expected, const AdamWValues& actual, const Tolerances& tolerances, const std::string& label) {
    ASSERT_EQ(expected.param.shape(), actual.param.shape()) << label;
    const auto [param_mean_error, param_max_error] = error_metrics(expected.param, actual.param);
    EXPECT_LT(param_mean_error, tolerances.mean_error) << label << ": param";
    EXPECT_LT(param_max_error, tolerances.max_error) << label << ": param";
    auto expect_close = [&](const xt::xarray<float>& reference, const xt::xarray<float>& result, const char* name) {
        ASSERT_EQ(reference.shape(), result.shape()) << label << ": " << name;
        EXPECT_LT(error_metrics(reference, result).max_error, tolerances.max_error) << label << ": " << name;
    };
    expect_close(expected.exp_avg, actual.exp_avg, "exp_avg");
    expect_close(expected.exp_avg_sq, actual.exp_avg_sq, "exp_avg_sq");
    ASSERT_EQ(expected.max_exp_avg_sq.has_value(), actual.max_exp_avg_sq.has_value()) << label;
    if (expected.max_exp_avg_sq.has_value()) {
        expect_close(*expected.max_exp_avg_sq, *actual.max_exp_avg_sq, "max_exp_avg_sq");
    }
    EXPECT_EQ(count_bit_mismatches(expected.grad, actual.grad), 0U) << label << ": the gradient must not change";
}

// Guards against a vacuous pass: a hit that kept any one of the cached program's scalars must miss the reference.
void expect_each_stale_scalar_detectable(
    const AdamWLaunchTensors& tensors, const AdamWStepScalars& scalars, const AdamWStepScalars& cached) {
    const auto expected = adamw_reference(tensors, scalars);
    for (const auto& [name, stale] : stale_scalar_variants(scalars, cached)) {
        const float mean_error = error_metrics(expected.param, adamw_reference(tensors, stale).param).mean_error;
        EXPECT_GT(mean_error, kFp32Tolerances.mean_error) << "a stale " << name << " is too close to tell apart";
    }
}

// Launch A compiles the program. Launch B updates fresh states with the hash-excluded scalars changed. Launch C
// updates fresh copies of launch A's states with launch A's scalars and must reproduce launch A bit for bit.
void run_new_scalars_on_fresh_states(
    bool amsgrad, const AdamWStepScalars& scalars_b, const std::vector<uint32_t>& shape) {
    auto& device = ttml::autograd::ctx().get_device();
    constexpr uint32_t kSeedA = 1000U;

    const auto a = make_launch_tensors(shape, amsgrad, kSeedA);
    const auto entries_before_a = device.num_program_cache_entries();
    const auto a_out = launch_adamw(a, kScalarsA);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_a) << "launch A must compile a program of its own";
    EXPECT_EQ(a_out.buffer()->address(), a.param.tensor.buffer()->address()) << "AdamW updates the parameter in place";
    const auto a_after = read_back(a);
    expect_close_to_reference(adamw_reference(a, kScalarsA), a_after, kFp32Tolerances, "launch A");

    const auto b = make_launch_tensors(shape, amsgrad, 2000U);
    const auto entries_before_b = device.num_program_cache_entries();
    launch_adamw(b, scalars_b);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_b) << "launch B must reuse launch A's program";
    expect_distinct_buffers(a, b);
    const auto b_after = read_back(b);
    expect_close_to_reference(
        adamw_reference(b, scalars_b), b_after, kFp32Tolerances, "launch B (cache hit, new scalars)");
    expect_each_stale_scalar_detectable(b, scalars_b, kScalarsA);

    const auto c = make_launch_tensors(shape, amsgrad, kSeedA);
    const auto entries_before_c = device.num_program_cache_entries();
    launch_adamw(c, kScalarsA);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    expect_distinct_buffers(a, c);
    expect_distinct_buffers(b, c);
    expect_bitwise_equal(a_after, read_back(c), "launch C (launch A again, cache hit)");

    expect_bitwise_equal(a_after, read_back(a), "launch A's buffers after the later launches");
    expect_bitwise_equal(b_after, read_back(b), "launch B's buffers after launch C");
}

}  // namespace

TEST_F(AdamWFullPrecisionCacheHitTest, NewScalarsOnFreshStates_TwoCoreGroups) {
    run_new_scalars_on_fresh_states(/*amsgrad=*/false, kScalarsB, two_core_group_rows_shape(32U));
}

// AMSGrad reads and writes max_exp_avg_sq as well and has a program of its own.
TEST_F(AdamWFullPrecisionCacheHitTest, AmsgradNewScalarsOnFreshStates_TwoCoreGroups) {
    run_new_scalars_on_fresh_states(/*amsgrad=*/true, kAmsgradScalarsB, two_core_group_rows_shape(32U));
}

// One tile: one core, and no group-2 compute kernel to patch. The bf16 program of the same shape is cached first; the
// program hash keys the dtype, so the fp32 launch A must still compile a program of its own.
TEST_F(AdamWFullPrecisionCacheHitTest, NewScalarsOnFreshStates_OneTile) {
    auto& device = ttml::autograd::ctx().get_device();
    const std::vector<uint32_t> shape = {1U, 1U, 32U, 32U};

    const auto bf16 = make_launch_tensors(shape, /*amsgrad=*/false, 900U, ttnn::DataType::BFLOAT16);
    const auto entries_before = device.num_program_cache_entries();
    launch_adamw(bf16, kScalarsA);
    ASSERT_GT(device.num_program_cache_entries(), entries_before) << "the bf16 launch must compile a program";
    const auto bf16_after = read_back(bf16);
    expect_close_to_reference(adamw_reference(bf16, kScalarsA), bf16_after, kBf16Tolerances, "bf16 launch");

    run_new_scalars_on_fresh_states(/*amsgrad=*/false, kScalarsB, shape);
    expect_bitwise_equal(bf16_after, read_back(bf16), "the bf16 launch's buffers after the fp32 launches");
}
