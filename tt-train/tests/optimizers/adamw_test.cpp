// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/adamw.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/optimizers/adamw/adamw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
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

// How the step-varying scalars reach the kernel: as float runtime args (lr, beta1_pow,
// beta2_pow, weight_decay), or pre-combined into single-element f32 device tensors
// (step_size, inv_sqrt_bc2, decay_factor). Every case below is run both ways.
enum class ScalarSource : std::uint8_t { Float, Tensor };

static std::string_view scalar_source_name(ScalarSource source) {
    return source == ScalarSource::Float ? "FloatScalars" : "TensorScalars";
}

// Custom printer for ScalarSource used by gtest to make test output readable
void PrintTo(const ScalarSource& source, std::ostream* os) {
    *os << scalar_source_name(source);
}

class AdamWComparisonTest : public ::testing::TestWithParam<std::tuple<AdamWCase, ScalarSource>> {
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

// from_xtensor() always converts to bf16, so the op-level paths (which also need fp32 params)
// build their tensors from a flat vector with an explicit spec.
static ttnn::Tensor make_device_tensor(const std::vector<float>& data, const ttnn::Shape& shape, ttnn::DataType dtype) {
    const auto spec = tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    return ttnn::Tensor::from_vector(data, spec, &ttml::autograd::ctx().get_device());
}

static ttnn::Tensor to_scalar_tensor(float value) {
    return make_device_tensor({value}, ttnn::Shape({1, 1, 1, 1}), ttnn::DataType::FLOAT32);
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

static ErrorMetrics compute_error_metrics(
    const xt::xarray<float>& reference, const std::vector<float>& actual, const std::string& name) {
    float sum_error = 0.0f;
    float max_error = 0.0f;
    const size_t count = reference.size();

    for (size_t i = 0; i < count; ++i) {
        const float error = std::abs(reference.data()[i] - actual[i]);
        sum_error += error;
        max_error = std::max(max_error, error);
    }

    return {sum_error / static_cast<float>(count), max_error, name};
}

struct DeviceStepResult {
    std::vector<float> param;
    std::vector<float> exp_avg;
    std::vector<float> exp_avg_sq;
    std::vector<float> max_exp_avg_sq;  // empty unless amsgrad is enabled
};

// One ttml::metal::adamw step at `step` on freshly allocated device tensors, with the
// step-varying scalars delivered as floats or as single-element f32 tensors. `keep_alive`,
// when non-null, retains the tensors so a following call is forced to allocate at different
// addresses (which is what exercises override_runtime_arguments on a program cache hit).
static DeviceStepResult run_device_step(
    const AdamWCase& pc,
    size_t step,
    ScalarSource scalar_source,
    ttnn::DataType param_dtype,
    const xt::xarray<float>& w0,
    const xt::xarray<float>& g0,
    const xt::xarray<float>& m0,
    const xt::xarray<float>& v0,
    const xt::xarray<float>& max_v0,
    std::vector<ttnn::Tensor>* keep_alive = nullptr) {
    const ttnn::Shape shape(
        {static_cast<uint32_t>(pc.shape[0]),
         static_cast<uint32_t>(pc.shape[1]),
         static_cast<uint32_t>(pc.shape[2]),
         static_cast<uint32_t>(pc.shape[3])});
    const auto flatten = [](const xt::xarray<float>& x) { return std::vector<float>(x.begin(), x.end()); };

    auto param = make_device_tensor(flatten(w0), shape, param_dtype);
    // Gradients are always bf16.
    auto grad = make_device_tensor(flatten(g0), shape, ttnn::DataType::BFLOAT16);
    auto exp_avg = make_device_tensor(flatten(m0), shape, param_dtype);
    auto exp_avg_sq = make_device_tensor(flatten(v0), shape, param_dtype);
    std::optional<ttnn::Tensor> max_exp_avg_sq;
    if (pc.amsgrad) {
        max_exp_avg_sq = make_device_tensor(flatten(max_v0), shape, param_dtype);
    }

    const float beta1_pow = std::pow(pc.beta1, static_cast<float>(step));
    const float beta2_pow = std::pow(pc.beta2, static_cast<float>(step));

    std::optional<ttnn::Tensor> step_size;
    std::optional<ttnn::Tensor> inv_sqrt_bc2;
    std::optional<ttnn::Tensor> decay_factor;
    ttnn::Tensor param_out;
    if (scalar_source == ScalarSource::Tensor) {
        step_size = to_scalar_tensor(pc.lr / (1.0F - beta1_pow));
        inv_sqrt_bc2 = to_scalar_tensor(1.0F / std::sqrt(1.0F - beta2_pow));
        decay_factor = to_scalar_tensor(1.0F - pc.lr * pc.weight_decay);
        param_out = ttml::metal::adamw(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            *step_size,
            *inv_sqrt_bc2,
            *decay_factor,
            pc.beta1,
            pc.beta2,
            pc.epsilon);
    } else {
        param_out = ttml::metal::adamw(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            pc.lr,
            pc.beta1,
            pc.beta2,
            beta1_pow,
            beta2_pow,
            pc.epsilon,
            pc.weight_decay);
    }

    // exp_avg / exp_avg_sq (/ max_exp_avg_sq) are updated in place.
    DeviceStepResult result;
    result.param = param_out.to_vector<float>();
    result.exp_avg = exp_avg.to_vector<float>();
    result.exp_avg_sq = exp_avg_sq.to_vector<float>();
    if (max_exp_avg_sq.has_value()) {
        result.max_exp_avg_sq = max_exp_avg_sq->to_vector<float>();
    }

    if (keep_alive != nullptr) {
        keep_alive->insert(keep_alive->end(), {param, grad, exp_avg, exp_avg_sq});
        if (max_exp_avg_sq.has_value()) {
            keep_alive->push_back(*max_exp_avg_sq);
        }
        if (step_size.has_value()) {
            keep_alive->insert(keep_alive->end(), {*step_size, *inv_sqrt_bc2, *decay_factor});
        }
    }
    return result;
}

static void run_step_and_compare(const AdamWCase& pc, ScalarSource scalar_source) {
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

    cpu_opt.step(w_cpu, g_cpu);

    ErrorMetrics fused_metrics{};
    if (scalar_source == ScalarSource::Float) {
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
            serialization::StateDict fused_state;
            fused_state["exp_avg"] = serialization::NamedParameters{{"theta", m0_tensor}};
            fused_state["exp_avg_sq"] = serialization::NamedParameters{{"theta", v0_tensor}};
            fused_state["steps"] = initial_steps;
            fused_state["lr"] = pc.lr;
            fused_state["beta1"] = pc.beta1;
            fused_state["beta2"] = pc.beta2;
            fused_state["epsilon"] = pc.epsilon;
            fused_state["weight_decay"] = pc.weight_decay;
            fused_state["amsgrad"] = pc.amsgrad;
            fused_state["stochastic_rounding"] = false;
            if (pc.amsgrad) {
                auto max_v0_tensor = autograd::create_tensor(to_tt_bf16(max_v0), false);
                fused_state["max_exp_avg_sq"] = serialization::NamedParameters{{"theta", max_v0_tensor}};
            }
            opt_fused.set_state_dict(fused_state);
        }

        opt_fused.step();

        auto result_fused = theta_fused->get_value();
        auto result_fused_cpu = core::to_xtensor(result_fused);

        fused_metrics = compute_error_metrics(w_cpu, result_fused_cpu, "AdamW");
    } else {
        // The optimizer only drives the float-scalar overload, so the tensor-scalar variant is
        // exercised at the op level with the scalars the optimizer would have produced at this step.
        const size_t step = initial_steps + 1;
        const auto float_scalars =
            run_device_step(pc, step, ScalarSource::Float, ttnn::DataType::BFLOAT16, w0, g0, m0, v0, max_v0);
        const auto tensor_scalars =
            run_device_step(pc, step, ScalarSource::Tensor, ttnn::DataType::BFLOAT16, w0, g0, m0, v0, max_v0);

        // Both overloads feed the kernel the same three immediates, so results are bit-identical.
        EXPECT_EQ(float_scalars.param, tensor_scalars.param);
        EXPECT_EQ(float_scalars.exp_avg, tensor_scalars.exp_avg);
        EXPECT_EQ(float_scalars.exp_avg_sq, tensor_scalars.exp_avg_sq);
        EXPECT_EQ(float_scalars.max_exp_avg_sq, tensor_scalars.max_exp_avg_sq);

        fused_metrics = compute_error_metrics(w_cpu, tensor_scalars.param, "AdamW");
    }

    const float mean_error_tolerance = 1e-3f;
    const float max_error_tolerance = 1e-2f;

    EXPECT_LE(fused_metrics.mean_error, mean_error_tolerance) << "AdamW mean error exceeds tolerance";
    EXPECT_LE(fused_metrics.max_error, max_error_tolerance) << "AdamW max error exceeds tolerance";
}

static std::string CaseName(const ::testing::TestParamInfo<std::tuple<AdamWCase, ScalarSource>>& info) {
    const auto& [c, scalar_source] = info.param;
    return fmt::format(
        "{}_B{}H{}S{}C{}_{}",
        c.name,
        c.shape[0],
        c.shape[1],
        c.shape[2],
        c.shape[3],
        scalar_source_name(scalar_source));
}

// Both scalar sources are covered for every case, so each shape / hyperparameter / amsgrad
// combination is checked against the CPU reference with float scalars and with tensor scalars.
static auto scalar_sources() {
    return ::testing::Values(ScalarSource::Float, ScalarSource::Tensor);
}

TEST_P(AdamWComparisonTest, CompareImplementations) {
    const auto& [pc, scalar_source] = GetParam();
    run_step_and_compare(pc, scalar_source);
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
    AdamWBasicComparison,
    AdamWComparisonTest,
    ::testing::Combine(::testing::ValuesIn(kBasicCases), scalar_sources()),
    CaseName);

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
    AdamWWeightDecay,
    AdamWComparisonTest,
    ::testing::Combine(::testing::ValuesIn(kWeightDecayCases), scalar_sources()),
    CaseName);

// ====================================================================
// weight_decay_skip_1d Tests
// With the flag enabled, 1-D params (RMSNorm gains/biases, shape {1,1,1,N})
// must not be weight-decayed, while 2-D params still are.
// ====================================================================

class AdamWWeightDecaySkip1DTest : public ::testing::Test {
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

TEST_F(AdamWWeightDecaySkip1DTest, SkipsDecayOn1DParamsOnly) {
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

    // Device AdamW: both params in one optimizer, weight_decay_skip_1d enabled.
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
    cfg.weight_decay_skip_1d = true;
    optimizers::AdamW opt(params, cfg);

    // Inject momentum state (weight_decay_skip_1d is config-only, not serialized, so it is preserved).
    {
        serialization::StateDict state;
        state["exp_avg"] = serialization::NamedParameters{
            {"gain", autograd::create_tensor(to_tt_bf16(m1_0), false)},
            {"weight", autograd::create_tensor(to_tt_bf16(m2_0), false)}};
        state["exp_avg_sq"] = serialization::NamedParameters{
            {"gain", autograd::create_tensor(to_tt_bf16(v1_0), false)},
            {"weight", autograd::create_tensor(to_tt_bf16(v2_0), false)}};
        state["steps"] = initial_steps;
        state["lr"] = lr;
        state["beta1"] = beta1;
        state["beta2"] = beta2;
        state["epsilon"] = epsilon;
        state["weight_decay"] = weight_decay;
        state["amsgrad"] = false;
        state["stochastic_rounding"] = false;
        opt.set_state_dict(state);
    }

    opt.step();

    auto gain_result = core::to_xtensor(gain->get_value());
    auto weight_result = core::to_xtensor(weight->get_value());

    const float mean_error_tolerance = 1e-3f;
    const float max_error_tolerance = 1e-2f;

    // 1-D param must match the NO-weight-decay reference.
    auto gain_metrics = compute_error_metrics(w1_no_wd, gain_result, "gain_1d");
    EXPECT_LE(gain_metrics.mean_error, mean_error_tolerance) << "1-D param should not be weight-decayed";
    EXPECT_LE(gain_metrics.max_error, max_error_tolerance) << "1-D param should not be weight-decayed";

    // 2-D param must match the weight-decayed reference.
    auto weight_metrics = compute_error_metrics(w2_with_wd, weight_result, "weight_2d");
    EXPECT_LE(weight_metrics.mean_error, mean_error_tolerance) << "2-D param should be weight-decayed";
    EXPECT_LE(weight_metrics.max_error, max_error_tolerance) << "2-D param should be weight-decayed";

    // Guard against a vacuous pass: the undecayed 1-D result must be clearly distinct from the
    // would-be-decayed reference, i.e. the weight decay is actually large enough to observe.
    auto skip_vs_decay = compute_error_metrics(w1_with_wd, gain_result, "gain_skip_vs_decay");
    EXPECT_GT(skip_vs_decay.mean_error, mean_error_tolerance)
        << "weight decay too small to distinguish skipping from applying it; test is not meaningful";
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
// effective betas either through the state dict or through the setters, then verifies one
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

    // When exercising the setters, the state dict carries the constructor betas so that only
    // the setters introduce the effective ones.
    const float state_beta1 = use_beta_setters ? constructor_beta1 : effective_beta1;
    const float state_beta2 = use_beta_setters ? constructor_beta2 : effective_beta2;
    {
        serialization::StateDict state;
        state["exp_avg"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_bf16(m0), false)}};
        state["exp_avg_sq"] = serialization::NamedParameters{{"theta", autograd::create_tensor(to_tt_bf16(v0), false)}};
        state["steps"] = initial_steps;
        state["lr"] = lr;
        state["beta1"] = state_beta1;
        state["beta2"] = state_beta2;
        state["epsilon"] = epsilon;
        state["weight_decay"] = 0.0f;
        state["amsgrad"] = false;
        state["stochastic_rounding"] = false;
        opt.set_state_dict(state);
    }

    if (use_beta_setters) {
        opt.set_beta1(effective_beta1);
        opt.set_beta2(effective_beta2);
    }

    EXPECT_EQ(opt.get_steps(), initial_steps);
    EXPECT_FLOAT_EQ(opt.get_beta1(), effective_beta1);
    EXPECT_FLOAT_EQ(opt.get_beta2(), effective_beta2);

    opt.step();

    auto result = core::to_xtensor(theta->get_value());
    auto metrics = compute_error_metrics(w_cpu, result, "effective_betas");

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
    auto stale_metrics = compute_error_metrics(w_cpu, w_stale, "stale_bias_correction");
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

INSTANTIATE_TEST_SUITE_P(
    AdamWAMSGrad,
    AdamWComparisonTest,
    ::testing::Combine(::testing::ValuesIn(kAMSGradCases), scalar_sources()),
    CaseName);

// ====================================================================
// Stochastic Rounding Tests
// Test AdamW with stochastic rounding enabled
// ====================================================================

// These tests are nondeterministic but should never fail
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

    auto stoch_metrics = compute_error_metrics(w_cpu, result_stoch, "Stochastic");
    auto det_metrics = compute_error_metrics(w_cpu, result_det, "Deterministic");

    EXPECT_LT(stoch_metrics.mean_error, det_metrics.mean_error);
    EXPECT_LT(stoch_metrics.max_error, det_metrics.max_error);
}

// ====================================================================
// Tensor-scalar overload
// Coverage the parameterized suite above cannot express: fp32 params (that harness is
// bf16-only), program-cache reuse across steps, and scalar-tensor validation.
// ====================================================================

class AdamWTensorScalarsTest : public ::testing::Test {
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

static void expect_scalar_source_parity(const AdamWCase& pc, ttnn::DataType param_dtype) {
    ttml::autograd::ctx().set_seed(123U);
    auto& g = ttml::autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();
    const uint32_t seed_first_moment = g();
    const uint32_t seed_second_moment = g();
    const uint32_t seed_max_second_moment = g();

    xt::xarray<float> w0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_param);
    xt::xarray<float> g0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_grad);
    xt::xarray<float> m0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_first_moment);
    xt::xarray<float> v0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, 0.0F, 1.0F, seed_second_moment);
    xt::xarray<float> max_v0 =
        ttml::test_utils::make_uniform_xarray<float>(pc.shape, 0.0F, 1.0F, seed_max_second_moment);

    // Step 10 compiles and caches the program; step 11 hits the program cache with freshly
    // allocated scalar tensors holding different values, so it exercises
    // override_runtime_arguments updating the scalar-tensor addresses. Every tensor is kept
    // alive so later runs land at genuinely different addresses -- otherwise the allocator
    // would hand back the freed addresses and a broken override_runtime_arguments (reading
    // new values from old addresses) would still pass.
    std::vector<ttnn::Tensor> keep_alive;
    for (const size_t step : {size_t{10}, size_t{11}}) {
        const auto float_scalars =
            run_device_step(pc, step, ScalarSource::Float, param_dtype, w0, g0, m0, v0, max_v0, &keep_alive);
        const auto tensor_scalars =
            run_device_step(pc, step, ScalarSource::Tensor, param_dtype, w0, g0, m0, v0, max_v0, &keep_alive);

        EXPECT_EQ(float_scalars.param, tensor_scalars.param) << "step " << step;
        EXPECT_EQ(float_scalars.exp_avg, tensor_scalars.exp_avg) << "step " << step;
        EXPECT_EQ(float_scalars.exp_avg_sq, tensor_scalars.exp_avg_sq) << "step " << step;
        EXPECT_EQ(float_scalars.max_exp_avg_sq, tensor_scalars.max_exp_avg_sq) << "step " << step;
    }
}

TEST_F(AdamWTensorScalarsTest, MatchesFloatScalars_FP32) {
    expect_scalar_source_parity(
        {{1, 1, 64, 96}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, /*amsgrad=*/false, "FP32"},
        ttnn::DataType::FLOAT32);
}

TEST_F(AdamWTensorScalarsTest, MatchesFloatScalars_FP32_AmsGrad) {
    expect_scalar_source_parity(
        {{1, 1, 64, 96}, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, /*amsgrad=*/true, "FP32_AmsGrad"},
        ttnn::DataType::FLOAT32);
}

TEST_F(AdamWTensorScalarsTest, RejectsMultiElementScalarTensor) {
    const ttnn::Shape shape({1, 1, 32, 32});
    const auto data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, /*seed=*/0U);
    auto param = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);
    auto grad = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);
    auto exp_avg = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);
    auto exp_avg_sq = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);

    auto bad_step_size = make_device_tensor({1.0F, 2.0F}, ttnn::Shape({1, 1, 1, 2}), ttnn::DataType::FLOAT32);
    auto inv_sqrt_bc2 = to_scalar_tensor(1.0F);
    auto decay_factor = to_scalar_tensor(1.0F);
    EXPECT_ANY_THROW(ttml::metal::adamw(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        std::nullopt,
        bad_step_size,
        inv_sqrt_bc2,
        decay_factor,
        0.9F,
        0.999F,
        1e-8F));
}

TEST_F(AdamWTensorScalarsTest, RejectsNonFloat32ScalarTensor) {
    const ttnn::Shape shape({1, 1, 32, 32});
    const auto data = ttml::test_utils::make_uniform_vector<float>(shape.volume(), -1.0F, 1.0F, /*seed=*/0U);
    auto param = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);
    auto grad = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);
    auto exp_avg = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);
    auto exp_avg_sq = make_device_tensor(data, shape, ttnn::DataType::BFLOAT16);

    auto bad_step_size = make_device_tensor({1.0F}, ttnn::Shape({1, 1, 1, 1}), ttnn::DataType::BFLOAT16);
    auto inv_sqrt_bc2 = to_scalar_tensor(1.0F);
    auto decay_factor = to_scalar_tensor(1.0F);
    EXPECT_ANY_THROW(ttml::metal::adamw(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        std::nullopt,
        bad_step_size,
        inv_sqrt_bc2,
        decay_factor,
        0.9F,
        0.999F,
        1e-8F));
}
