// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/adamw_full_precision.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <cstdlib>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "xtensor/core/xtensor_forward.hpp"

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

// Convert float to bfloat16 and back to simulate bf16 precision loss
static float to_bfloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    // Zero out the lower 16 bits (mantissa truncation for bfloat16)
    bits &= 0xFFFF0000U;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// Convert entire xarray to bfloat16 precision
static xt::xarray<float> to_bfloat16_precision(const xt::xarray<float>& x) {
    xt::xarray<float> result = xt::empty_like(x);
    for (size_t i = 0; i < x.size(); ++i) {
        result.flat(i) = to_bfloat16(x.flat(i));
    }
    return result;
}

[[maybe_unused]] static ttnn::Tensor to_tt(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor(x, &ttml::autograd::ctx().get_device());
}

static ttnn::Tensor to_tt_bf16(const xt::xarray<float>& x) {
    auto tensor = ttml::core::from_xtensor(x, &ttml::autograd::ctx().get_device());
    return ttnn::typecast(tensor, tt::tt_metal::DataType::BFLOAT16);
}

// CPU reference implementation of AdamW with fp32 master weights
// This simulates the full precision optimizer behavior:
// - Receives bf16 gradients (simulated by truncating to bf16 precision)
// - Stores master weights in fp32
// - Stores momentum buffers in fp32
// - Outputs bf16 visible weights (simulated by truncating to bf16 precision)
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

    // Initialize master weights from bf16 weights (converted to fp32)
    void init(const xt::xarray<float>& bf16_weights) {
        // Master weights are fp32 copy of bf16 weights
        m_master_weights = bf16_weights;  // Already in fp32, but values are bf16-truncated
        m_first_moment = xt::zeros_like(m_master_weights);
        m_second_moment = xt::zeros_like(m_master_weights);
        if (m_amsgrad) {
            m_max_second_moment = xt::zeros_like(m_master_weights);
        }
    }

    // Perform one optimization step
    // grads should be bf16 precision (truncated)
    xt::xarray<float> step(const xt::xarray<float>& grads) {
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
            m_max_second_moment = xt::maximum(m_max_second_moment, second_moment_hat);
            denom = xt::sqrt(m_max_second_moment) + m_epsilon;
        } else {
            denom = xt::sqrt(second_moment_hat) + m_epsilon;
        }

        // Update master weights (fp32): params = params - lr * m_hat / denom - lr * weight_decay * params
        m_master_weights =
            m_master_weights - m_lr * first_moment_hat / denom - m_lr * m_weight_decay * m_master_weights;

        // Return bf16-truncated visible weights
        return to_bfloat16_precision(m_master_weights);
    }

    const xt::xarray<float>& get_master_weights() const {
        return m_master_weights;
    }

private:
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    float m_weight_decay;
    bool m_amsgrad;
    size_t m_steps;
    xt::xarray<float> m_master_weights;     // fp32
    xt::xarray<float> m_first_moment;       // fp32
    xt::xarray<float> m_second_moment;      // fp32
    xt::xarray<float> m_max_second_moment;  // fp32, for amsgrad
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

static void run_steps_and_compare(const AdamWFullPrecisionCase& pc, uint32_t steps) {
    using namespace ttml;

    ttml::autograd::ctx().set_seed(123U);
    auto& g = autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();

    // Generate random data in fp32
    xt::xarray<float> g0_fp32 = make_random_xarray(pc.shape, seed_grad);
    xt::xarray<float> w0_fp32 = make_random_xarray(pc.shape, seed_param);

    // Convert to bf16 precision (simulate bf16 input)
    xt::xarray<float> g0_bf16 = to_bfloat16_precision(g0_fp32);
    xt::xarray<float> w0_bf16 = to_bfloat16_precision(w0_fp32);

    // CPU reference implementation with fp32 master weights
    CPUAdamWFullPrecision cpu_opt(pc.lr, pc.beta1, pc.beta2, pc.epsilon, pc.weight_decay, pc.amsgrad);
    cpu_opt.init(w0_bf16);  // Initialize with bf16-truncated weights

    // AdamWFullPrecision implementation
    auto theta_full_precision = autograd::create_tensor(to_tt_bf16(w0_bf16), true);
    theta_full_precision->set_grad(to_tt_bf16(g0_bf16));
    ttml::serialization::NamedParameters params_full_precision{{"theta", theta_full_precision}};

    ttml::optimizers::AdamWFullPrecisionConfig full_precision_cfg;
    full_precision_cfg.lr = pc.lr;
    full_precision_cfg.beta1 = pc.beta1;
    full_precision_cfg.beta2 = pc.beta2;
    full_precision_cfg.epsilon = pc.epsilon;
    full_precision_cfg.weight_decay = pc.weight_decay;
    full_precision_cfg.amsgrad = pc.amsgrad;

    ttml::optimizers::AdamWFullPrecision opt_full_precision(params_full_precision, full_precision_cfg);

    // Run both optimizers for the specified number of steps
    for (uint32_t i = 0; i < steps; ++i) {
        cpu_opt.step(g0_bf16);
        opt_full_precision.step();
    }

    // Get CPU master weights (fp32)
    const auto& cpu_master_weights = cpu_opt.get_master_weights();

    // Get device master weights (fp32)
    const auto& device_master_weights = opt_full_precision.get_master_weights();
    auto master_weights_tensor = device_master_weights.at("theta")->get_value(autograd::PreferredPrecision::FULL);

    // Convert to CPU for comparison
    auto device_master_weights_cpu = core::to_xtensor(master_weights_tensor);

    fmt::print("\n=== Error Metrics (comparing fp32 master weights) ===\n");

    // Compute error metrics for master weights
    auto master_weights_metrics =
        compute_error_metrics(cpu_master_weights, device_master_weights_cpu, "AdamWFullPrecision Master Weights");

    // The error should be very small since both use fp32 master weights
    // Allow for small numerical differences due to hardware
    EXPECT_LT(master_weights_metrics.mean_error, 1e-3f)
        << "AdamWFullPrecision master weights mean error should be small compared to CPU reference";
    EXPECT_LT(master_weights_metrics.max_error, 1e-2f)
        << "AdamWFullPrecision master weights max error should be small compared to CPU reference";

    fmt::print("\n");
}

static std::string CaseName(const ::testing::TestParamInfo<AdamWFullPrecisionCase>& info) {
    const auto& c = info.param;
    return fmt::format("{}_B{}H{}S{}C{}", c.name, c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
}

TEST_P(AdamWFullPrecisionComparisonTest, CompareWithCPU) {
    const auto& pc = GetParam();
    // Run 2 steps to ensure momentum buffers are properly exercised
    const uint32_t steps = 2;
    run_steps_and_compare(pc, steps);
}

// Test cases with various hyperparameter configurations
static const AdamWFullPrecisionCase kBasicCases[] = {
    // Standard configurations with different learning rates
    {{1, 2, 128, 256}, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, false, "PureLR_1"},

    {{4, 8, 256, 1024}, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Standard_lr1e2"},
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Standard_lr1e3"},
    {{2, 4, 128, 256}, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Standard_lr1e4"},

    // Different beta1 values
    {{1, 16, 128, 128}, 1e-3f, 0.8f, 0.999f, 1e-8f, 0.0f, false, "Beta1_0p8"},
    {{2, 8, 128, 128}, 1e-3f, 0.95f, 0.999f, 1e-8f, 0.0f, false, "Beta1_0p95"},
    {{4, 4, 128, 128}, 1e-3f, 0.5f, 0.999f, 1e-8f, 0.0f, false, "Beta1_0p5"},

    // Different beta2 values
    {{1, 32, 64, 128}, 1e-3f, 0.9f, 0.99f, 1e-8f, 0.0f, false, "Beta2_0p99"},
    {{2, 16, 64, 128}, 1e-3f, 0.9f, 0.9999f, 1e-8f, 0.0f, false, "Beta2_0p9999"},
    {{1, 64, 64, 64}, 1e-3f, 0.9f, 0.95f, 1e-8f, 0.0f, false, "Beta2_0p95"},

    // Different epsilon values
    {{2, 32, 64, 64}, 1e-3f, 0.9f, 0.999f, 1e-7f, 0.0f, false, "Epsilon_1e7"},
    {{4, 16, 64, 64}, 1e-3f, 0.9f, 0.999f, 1e-9f, 0.0f, false, "Epsilon_1e9"},
    {{1, 4, 256, 256}, 1e-3f, 0.9f, 0.999f, 1e-6f, 0.0f, false, "Epsilon_1e6"},

    // Mixed configurations
    {{1, 32, 128, 512}, 1e-2f, 0.95f, 0.9999f, 1e-7f, 0.0f, false, "Mixed_1"},
    {{1, 32, 128, 512}, 1e-4f, 0.8f, 0.99f, 1e-9f, 0.0f, false, "Mixed_2"},
    {{1, 32, 128, 512}, 5e-3f, 0.85f, 0.995f, 1e-8f, 0.0f, false, "Mixed_3"},

    // Large and small tensor shapes
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Large_flat"},
    {{8, 8, 64, 64}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Large_4D"},
    {{1, 256, 32, 32}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, false, "Wide"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFullPrecisionBasicComparison, AdamWFullPrecisionComparisonTest, ::testing::ValuesIn(kBasicCases), CaseName);

// ====================================================================
// Weight Decay Tests
// Test AdamWFullPrecision with various weight decay configurations
// ====================================================================

static const AdamWFullPrecisionCase kWeightDecayCases[] = {
    // Standard configurations with different weight decay values
    {{2, 4, 128, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, false, "Standard_wd0p01"},
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.001f, false, "Standard_wd0p001"},
    {{2, 4, 128, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f, false, "Standard_wd0p1"},
    {{2, 8, 128, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0001f, false, "Standard_wd0p0001"},

    // Weight decay with different learning rates
    {{1, 16, 128, 128}, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.01f, false, "HighLR_wd0p01"},
    {{2, 8, 128, 128}, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, false, "LowLR_wd0p01"},
    {{4, 4, 128, 128}, 5e-3f, 0.9f, 0.999f, 1e-8f, 0.05f, false, "MidLR_wd0p05"},

    // Weight decay with different beta1 values
    {{1, 32, 64, 128}, 1e-3f, 0.8f, 0.999f, 1e-8f, 0.01f, false, "Beta1_0p8_wd0p01"},
    {{2, 16, 64, 128}, 1e-3f, 0.95f, 0.999f, 1e-8f, 0.01f, false, "Beta1_0p95_wd0p01"},
    {{1, 64, 64, 64}, 1e-3f, 0.5f, 0.999f, 1e-8f, 0.01f, false, "Beta1_0p5_wd0p01"},

    // High weight decay (aggressive regularization)
    {{1, 8, 128, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.5f, false, "HighWD_0p5"},
    {{2, 4, 128, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.2f, false, "HighWD_0p2"},
    {{1, 16, 64, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.3f, false, "HighWD_0p3"},

    // Small weight decay (light regularization)
    {{1, 8, 128, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f, false, "SmallWD_1e4"},
    {{2, 4, 128, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-5f, false, "SmallWD_1e5"},
    {{1, 16, 64, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-3f, false, "SmallWD_1e3"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFullPrecisionWeightDecay, AdamWFullPrecisionComparisonTest, ::testing::ValuesIn(kWeightDecayCases), CaseName);

// ====================================================================
// AMSGrad Tests
// Test AdamWFullPrecision with AMSGrad variant enabled
// ====================================================================

static const AdamWFullPrecisionCase kAMSGradCases[] = {
    // Standard configurations with AMSGrad
    {{4, 8, 256, 1024}, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "Standard_lr1e2"},
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "Standard_lr1e3"},
    {{2, 4, 128, 256}, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "Standard_lr1e4"},

    // AMSGrad with different beta1 values
    {{1, 16, 128, 128}, 1e-3f, 0.8f, 0.999f, 1e-8f, 0.0f, true, "Beta1_0p8"},
    {{2, 8, 128, 128}, 1e-3f, 0.95f, 0.999f, 1e-8f, 0.0f, true, "Beta1_0p95"},
    {{4, 4, 128, 128}, 1e-3f, 0.5f, 0.999f, 1e-8f, 0.0f, true, "Beta1_0p5"},

    // AMSGrad with different beta2 values
    {{1, 32, 64, 128}, 1e-3f, 0.9f, 0.99f, 1e-8f, 0.0f, true, "Beta2_0p99"},
    {{2, 16, 64, 128}, 1e-3f, 0.9f, 0.9999f, 1e-8f, 0.0f, true, "Beta2_0p9999"},
    {{1, 64, 64, 64}, 1e-3f, 0.9f, 0.95f, 1e-8f, 0.0f, true, "Beta2_0p95"},

    // AMSGrad with weight decay
    {{2, 4, 128, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true, "WeightDecay_0p01"},
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.001f, true, "WeightDecay_0p001"},
    {{2, 4, 128, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f, true, "WeightDecay_0p1"},

    // AMSGrad with high weight decay
    {{1, 8, 128, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.5f, true, "HighWD_0p5"},
    {{2, 4, 128, 128}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.2f, true, "HighWD_0p2"},

    // Mixed configurations with AMSGrad
    {{1, 32, 128, 512}, 1e-2f, 0.95f, 0.9999f, 1e-7f, 0.02f, true, "Mixed_1"},
    {{1, 32, 128, 512}, 1e-4f, 0.8f, 0.99f, 1e-9f, 0.001f, true, "Mixed_2"},
    {{1, 32, 128, 512}, 5e-3f, 0.85f, 0.995f, 1e-8f, 0.015f, true, "Mixed_3"},

    // Large and small tensor shapes with AMSGrad
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true, "Large_flat"},
    {{8, 8, 64, 64}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true, "Large_4D"},
    {{1, 256, 32, 32}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.005f, true, "Wide"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFullPrecisionAMSGrad, AdamWFullPrecisionComparisonTest, ::testing::ValuesIn(kAMSGradCases), CaseName);
