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

static xt::xarray<float> make_random_xarray(
    const std::array<std::size_t, 4>& s, uint32_t seed, float min = -1.0F, float max = 1.0F) {
    xt::xarray<float> x = xt::empty<float>({s[0], s[1], s[2], s[3]});
    ttml::core::parallel_generate(
        std::span{x.data(), x.size()}, [min, max]() { return std::uniform_real_distribution<float>(min, max); }, seed);
    return x;
}

static xt::xarray<bfloat16> make_random_bf16_xarray(
    const std::array<std::size_t, 4>& s, uint32_t seed, float min = -1.0F, float max = 1.0F) {
    xt::xarray<bfloat16> x = xt::empty<bfloat16>({s[0], s[1], s[2], s[3]});
    ttml::core::parallel_generate(
        std::span{x.data(), x.size()}, [min, max]() { return std::uniform_real_distribution<float>(min, max); }, seed);
    return x;
}

static ttnn::Tensor to_tt_bf16(const xt::xarray<bfloat16>& x) {
    return ttml::core::from_xtensor<bfloat16, ttnn::DataType::BFLOAT16>(x, &ttml::autograd::ctx().get_device());
}

// NOTE: ttml::core::from_xtensor() performs fp32 -> bf16 conversion, that's why this function exists
static ttnn::Tensor to_tt_fp32(const xt::xarray<float>& x) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = tt::tt_metal::experimental::xtensor::get_shape_from_xarray(x);
    auto buffer_span = ttml::core::xtensor_to_span(x);

    ttnn::MemoryConfig output_mem_config{};
    const auto tensor_layout = ttnn::TensorLayout(
        ttnn::DataType::FLOAT32, ttnn::PageConfig(ttnn::Layout::ROW_MAJOR), tt::tt_metal::MemoryConfig{});
    auto output = ttnn::Tensor::from_span<float>(buffer_span, ttnn::TensorSpec(shape, tensor_layout));

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

private:
    float m_lr;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    float m_weight_decay;
    bool m_amsgrad;
    size_t m_steps;
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
    xt::xarray<float> w0_fp32 = make_random_xarray(pc.shape, seed_param);
    xt::xarray<float> m0 = make_random_xarray(pc.shape, seed_first_moment);
    xt::xarray<float> v0 = make_random_xarray(pc.shape, seed_second_moment, 0.0F, 1.0F);  // must be >= 0
    xt::xarray<float> max_v0 = make_random_xarray(pc.shape, seed_max_second_moment, 0.0F, 1.0F);

    // Generate gradient directly as bfloat16
    xt::xarray<bfloat16> g0_bf16 = make_random_bf16_xarray(pc.shape, seed_grad);
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

    // Inject optimizer state
    serialization::StateDict state;
    state["steps"] = initial_steps;
    state["master_weights"] =
        serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(w0_fp32), false)}};
    state["exp_avg"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(m0), false)}};
    state["exp_avg_sq"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_fp32(v0), false)}};
    state["amsgrad"] = pc.amsgrad;
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
    // AMSGrad with weight decay
    {{1, 4, 64, 256}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true, "WeightDecay_0p01"},
    // AMSGrad with different shape
    {{2, 8, 64, 512}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f, true, "NIGHTLY_Large_4D"},
};

INSTANTIATE_TEST_SUITE_P(
    AdamWFullPrecisionAMSGrad, AdamWFullPrecisionComparisonTest, ::testing::ValuesIn(kAMSGradCases), CaseName);
