// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/sgd_fused.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <bit>
#include <core/ttnn_all_includes.hpp>
#include <functional>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "xtensor/core/xtensor_forward.hpp"

class SGDFusedFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(SGDFusedFullTest, SGDFusedTest) {
    using namespace ttml::ops;
    ttml::autograd::ctx().set_seed(42U);
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32U;
    const size_t num_features = 64U;
    std::vector<float> features;
    features.reserve(batch_size * num_features);

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    std::vector<float> targets;
    for (size_t i = 0; i < batch_size; ++i) {
        targets.push_back(static_cast<float>(i) * 0.1F);
    }

    auto data_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1U, 1U, num_features}), device));

    auto targets_tensor =
        ttml::autograd::create_tensor(ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1U, 1U, 1U}), device));

    auto model = ttml::modules::LinearLayer(num_features, 1U);
    auto initial_weight = model.get_weight();
    auto initial_weight_values = ttml::core::to_vector(initial_weight->get_value());

    auto sgd_fused_config = ttml::optimizers::SGDFusedConfig();
    sgd_fused_config.lr = 1e-4f;
    auto optimizer = ttml::optimizers::SGDFused(model.parameters(), sgd_fused_config);

    const size_t steps = 10U;
    std::vector<float> losses;
    losses.reserve(steps);
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data_tensor);
        auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
        auto loss_value = ttml::core::to_vector(loss->get_value())[0];
        losses.emplace_back(loss_value);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }

    auto current_weight_values = ttml::core::to_vector(model.get_weight()->get_value());

    EXPECT_LT(losses.back(), losses.front());
}

void performTraining(
    xt::xarray<float>& losses,
    size_t steps,
    ttml::optimizers::OptimizerBase& optimizer,
    ttml::modules::LinearLayer& model,
    ttml::autograd::TensorPtr& data_tensor,
    ttml::autograd::TensorPtr& targets_tensor,
    size_t num_features) {
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data_tensor);
        auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
        auto loss_value = ttml::core::to_vector(loss->get_value())[0];
        losses(step) = loss_value;
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }
}

TEST_F(SGDFusedFullTest, SGDCpuVsFusedVsCompositeTest) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32U;
    const size_t num_features = 64U;
    const size_t one = 1U;

    xt::xarray<float> features = xt::zeros<float>({batch_size, one, one, num_features});
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features(i, 0, 0, j) = static_cast<float>(i) * 0.1F;
        }
    }

    xt::xarray<float> targets = xt::zeros<float>({batch_size, one, one, one});
    for (size_t i = 0; i < batch_size; ++i) {
        targets(i, 0, 0, 0) = static_cast<float>(i) * 0.1F;
    }

    auto data_tensor =
        ttml::autograd::create_tensor(ttml::core::from_xtensor(features, &ttml::autograd::ctx().get_device()));
    auto targets_tensor =
        ttml::autograd::create_tensor(ttml::core::from_xtensor(targets, &ttml::autograd::ctx().get_device()));

    constexpr float learning_rate = 1e-3f;
    const size_t steps = 100U;

    xt::xarray<float> cpuLosses = xt::zeros<float>({steps});
    ttml::autograd::ctx().set_seed(42U);
    {
        auto model = ttml::modules::LinearLayer(num_features, 1U);
        auto initial_weight = model.get_weight();
        auto initial_weight_values = ttml::core::to_vector(initial_weight->get_value());

        for (size_t step = 0; step < steps; ++step) {
            for (auto& [name, tensor_ptr] : model.parameters()) {
                if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
                    tensor_ptr->set_grad(ttnn::Tensor());
                }
            }
            auto prediction = model(data_tensor);
            auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
            auto loss_value = ttml::core::to_vector(loss->get_value())[0];
            cpuLosses(step) = loss_value;
            loss->backward();
            for (auto& [name, tensor_ptr] : model.parameters()) {
                auto gradients = tensor_ptr->get_grad();
                auto param = ttml::core::to_vector(tensor_ptr->get_value());
                auto grad_vec = ttml::core::to_xtensor(gradients);
                for (size_t i = 0; i < param.size(); ++i) {
                    param[i] -= learning_rate * grad_vec(i);
                }
                tensor_ptr->set_value(ttml::core::from_vector(param, tensor_ptr->get_shape(), device));
            }
            ttml::autograd::ctx().reset_graph();
        }
    }

    xt::xarray<float> compLosses = xt::zeros<float>({steps});
    ttml::autograd::ctx().set_seed(42U);
    {
        auto model = ttml::modules::LinearLayer(num_features, 1U);
        auto initial_weight = model.get_weight();
        auto initial_weight_values = ttml::core::to_vector(initial_weight->get_value());

        auto config = ttml::optimizers::SGDConfig();
        config.lr = learning_rate;

        auto optimizer = ttml::optimizers::SGD(model.parameters(), config);
        performTraining(compLosses, steps, optimizer, model, data_tensor, targets_tensor, num_features);
    }

    xt::xarray<float> fusedLosses = xt::zeros<float>({steps});
    ttml::autograd::ctx().set_seed(42U);
    {
        auto model = ttml::modules::LinearLayer(num_features, 1U);
        auto initial_weight = model.get_weight();
        auto initial_weight_values = ttml::core::to_vector(initial_weight->get_value());

        auto config = ttml::optimizers::SGDFusedConfig();
        config.lr = learning_rate;

        auto optimizer = ttml::optimizers::SGDFused(model.parameters(), config);
        performTraining(fusedLosses, steps, optimizer, model, data_tensor, targets_tensor, num_features);
    }

    fmt::print("Cpu vs Composite vs Fused losses (step, cpu, comp, fused, abs_diff_comp_fused, abs_diff_cpu_comp):\n");
    for (size_t i = 0; i < steps; ++i) {
        const float cpu = cpuLosses(i);
        const float comp = compLosses(i);
        const float fused = fusedLosses(i);
        fmt::print(
            "{:3d}: {: .8f}  {: .8f}  {: .8f}  |Δcomp-fused|={: .8f}  |Δcpu-fused|={: .8f}\n",
            static_cast<int>(i),
            cpu,
            comp,
            fused,
            std::fabs(comp - fused),
            std::fabs(cpu - fused));
    }

    constexpr float atol = 1e-3f;
    EXPECT_NEAR(fusedLosses.back(), cpuLosses.back(), atol);
    EXPECT_NEAR(fusedLosses.back(), compLosses.back(), atol);
}

struct ParityCase {
    std::array<std::size_t, 4> shape;  // (B, H, S, C)
    float lr{1e-3f};
    float momentum{0.0f};
    float dampening{0.0f};
    float weight_decay{0.0f};
    bool nesterov{false};
    std::string name;
};

// Custom printer for ParityCase to make test output readable
void PrintTo(const ParityCase& pc, std::ostream* os) {
    *os << fmt::format(
        "ParityCase(name='{}', shape=[{},{},{},{}], lr={}, momentum={}, dampening={}, weight_decay={}, nesterov={})",
        pc.name,
        pc.shape[0],
        pc.shape[1],
        pc.shape[2],
        pc.shape[3],
        pc.lr,
        pc.momentum,
        pc.dampening,
        pc.weight_decay,
        pc.nesterov);
}

class SGDFusedParityTest : public ::testing::TestWithParam<ParityCase> {
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

static inline uint16_t fp32_to_bf16_rne(float v) {
    uint32_t x = std::bit_cast<uint32_t>(v);
    uint32_t lsb = (x >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;  // ties-to-even
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

static inline float bf16_to_fp32(uint16_t bf16_bits) {
    uint32_t bits = static_cast<uint32_t>(bf16_bits) << 16;
    return std::bit_cast<float>(bits);
}

static void cpu_sgd_step(
    xt::xarray<float>& param,
    xt::xarray<float>& grad,
    xt::xarray<float>& momentum_buffer,
    float lr,
    float momentum_val,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool has_momentum) {
    for (size_t i = 0; i < param.size(); ++i) {
        // Inputs are BF16-quantized already
        float theta = param(i);
        float g_orig = grad(i);
        float m_prev = (has_momentum ? momentum_buffer(i) : 0.0f);

        // g_t includes weight decay and dampening (used to build momentum)
        float g_t = g_orig;
        if (weight_decay != 0.0f)
            g_t += weight_decay * theta;
        if (dampening != 0.0f)
            g_t *= (1.0f - dampening);

        // momentum update
        float m_t = (has_momentum && momentum_val != 0.0f) ? (g_t + momentum_val * m_prev) : g_t;

        float update = nesterov ? (g_t + momentum_val * m_t) : m_t;

        float theta_new = theta - lr * update;

        param(i) = theta_new;
        if (has_momentum && momentum_val != 0.0f) {
            momentum_buffer(i) = m_t;
        }
    }
}

static void run_one_step_and_compare(const ParityCase& pc) {
    using namespace ttml;

    auto& g = autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();
    const uint32_t seed_mom = g();

    // Same data used for both optimizers
    xt::xarray<float> w0 = make_random_xarray(pc.shape, seed_param);
    xt::xarray<float> g0 = make_random_xarray(pc.shape, seed_grad);

    // Convert to BF16 precision
    xt::xarray<float> w0_bf16 = w0;
    xt::xarray<float> g0_bf16 = g0;
    for (size_t i = 0; i < w0_bf16.size(); ++i) {
        w0_bf16(i) = bf16_to_fp32(fp32_to_bf16_rne(w0_bf16(i)));
        g0_bf16(i) = bf16_to_fp32(fp32_to_bf16_rne(g0_bf16(i)));
    }
    xt::xarray<float> w_cpu = w0_bf16;
    xt::xarray<float> g_cpu = g0_bf16;
    xt::xarray<float> mom_cpu = xt::zeros_like(w0_bf16);
    if (pc.momentum > 0.0f) {
        mom_cpu = make_random_xarray(pc.shape, seed_mom);
        for (size_t i = 0; i < mom_cpu.size(); ++i) {
            mom_cpu(i) = bf16_to_fp32(fp32_to_bf16_rne(mom_cpu(i)));
        }
    }

    // Build autograd tensor for fused optimizer
    auto theta_fused = autograd::create_tensor(to_tt(w0_bf16), true);
    theta_fused->set_grad(to_tt(g0_bf16));

    // Named parameters map for fused optimizer
    ttml::serialization::NamedParameters params_fused{{"theta", theta_fused}};

    // Build fused config
    ttml::optimizers::SGDFusedConfig fused_cfg;
    fused_cfg.lr = pc.lr;
    fused_cfg.momentum = pc.momentum;
    fused_cfg.dampening = pc.dampening;
    fused_cfg.weight_decay = pc.weight_decay;
    fused_cfg.nesterov = pc.nesterov;

    // Create fused optimizer
    ttml::optimizers::SGDFused opt_fused(params_fused, fused_cfg);

    // If momentum is used, initialize the momentum buffer with the same values
    if (pc.momentum > 0.0f) {
        auto sd_fused = opt_fused.get_state_dict();
        auto mom_fused = std::get<ttml::serialization::NamedParameters>(sd_fused.at("momentum"));
        mom_fused["theta"]->set_value(to_tt(mom_cpu));
        opt_fused.set_state_dict(sd_fused);
    }

    // fused optimizer step
    opt_fused.step();

    // CPU optimizer step
    cpu_sgd_step(
        w_cpu, g_cpu, mom_cpu, pc.lr, pc.momentum, pc.dampening, pc.weight_decay, pc.nesterov, pc.momentum > 0.0f);

    // Pull fused params
    const auto w_fused = ttml::core::to_xtensor(theta_fused->get_value());
    ASSERT_EQ(w_fused.shape(), w_cpu.shape());

    // 1) Round CPU FP32 result once to BF16 (RNE), then back to FP32 for compare
    xt::xarray<float> w_cpu_bf = w_cpu;
    for (size_t i = 0; i < w_cpu_bf.size(); ++i) {
        w_cpu_bf(i) = bf16_to_fp32(fp32_to_bf16_rne(w_cpu_bf(i)));
    }

    const float atol = 1e-3f;  // absolute tolerance
    const float rtol = 1e-2f;  // relative tolerance (1%)

    bool params_match = xt::allclose(w_fused, w_cpu_bf, atol, rtol);
    if (!params_match) {
        auto abs_err = xt::abs(w_fused - w_cpu_bf);
        float max_abs = xt::amax(abs_err)();
        auto denom = xt::maximum(xt::abs(w_cpu_bf), 1e-12f);
        float max_rel = xt::amax(abs_err / denom)();
        fmt::print(
            "allclose failed: max_abs = {:.6e}, max_rel = {:.6e} (atol={}, rtol={})\n", max_abs, max_rel, atol, rtol);

        size_t printed = 0;
        for (size_t i = 0; i < w_fused.size() && printed < 8; ++i) {
            float a = w_fused(i);
            float b = w_cpu_bf(i);
            float e = std::fabs(a - b);
            float er = e / std::max(std::fabs(b), 1e-12f);
            if (!(e <= atol + rtol * std::fabs(b))) {
                fmt::print(
                    "idx {:4d} | fused={: .8f} | cpu_bf={: .8f} | abs={:.3e} rel={:.3e}\n",
                    static_cast<int>(i),
                    a,
                    b,
                    e,
                    er);
                ++printed;
            }
        }
    }

    EXPECT_TRUE(params_match) << "Param mismatch after one step in case '" << pc.name << "' "
                              << "shape=(" << pc.shape[0] << "," << pc.shape[1] << "," << pc.shape[2] << ","
                              << pc.shape[3] << ")";

    if (pc.momentum > 0.0f) {
        xt::xarray<float> mom_cpu_bf = mom_cpu;
        for (size_t i = 0; i < mom_cpu_bf.size(); ++i) {
            mom_cpu_bf(i) = bf16_to_fp32(fp32_to_bf16_rne(mom_cpu_bf(i)));
        }
        auto sd_fused = opt_fused.get_state_dict();
        auto mom_fused = std::get<ttml::serialization::NamedParameters>(sd_fused.at("momentum"));
        ASSERT_TRUE(mom_fused.count("theta"));

        const auto m_fused = ttml::core::to_xtensor(mom_fused.at("theta")->get_value());

        bool momentum_match = xt::allclose(m_fused, mom_cpu_bf, atol, rtol);
        if (!momentum_match) {
            float max_diff = 0.0f;
            size_t mismatch_count = 0;
            for (size_t i = 0; i < m_fused.size(); ++i) {
                float diff = std::fabs(m_fused(i) - mom_cpu_bf(i));
                if (diff > atol) {
                    max_diff = std::max(max_diff, diff);
                    mismatch_count++;
                }
            }
            fmt::print(
                "Momentum mismatch: {} / {} values differ, max diff = {}\n", mismatch_count, m_fused.size(), max_diff);
        }

        EXPECT_TRUE(momentum_match) << "Momentum buffer mismatch in case '" << pc.name << "'";
    }
}
static std::string CaseName(const ::testing::TestParamInfo<ParityCase>& info) {
    const auto& c = info.param;
    return fmt::format("{}_B{}H{}S{}C{}", c.name, c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
}

TEST_P(SGDFusedParityTest, UpdateParityOneStep) {
    const auto& pc = GetParam();
    run_one_step_and_compare(pc);
}

static const ParityCase kCases[] = {
    {{1, 1, 1, 32}, 1e-3f, 0.0f, 0.0f, 0.0f, false, "Vec32_SGD"},
    {{1, 1, 1, 64}, 1e-3f, 0.0f, 0.0f, 1e-2f, false, "Vec64_L2"},
    {{1, 1, 32, 64}, 1e-3f, 0.9f, 0.1f, 0.0f, false, "Seq32x64_Mom_Damp"},
    {{2, 1, 32, 128}, 1e-3f, 0.9f, 0.0f, 0.0f, true, "Batch2_Seq32x128_Nesterov"},
    {{1, 4, 32, 32}, 5e-4f, 0.9f, 0.0f, 1e-3f, true, "Heads4_Seq32x32_Nest_L2"},
};

INSTANTIATE_TEST_SUITE_P(SGDFusedVsCPUParity, SGDFusedParityTest, ::testing::ValuesIn(kCases), CaseName);

static const ParityCase kBigCases[] = {
    // A "fatter" single vector (~262k params)
    {{1, 1, 1, 262'144}, 1e-3f, 0.0f, 0.0f, 0.0f, false, "Vec262k_SGD"},

    // Transformer-like blocks in the low millions of elements
    // 4 * 8 * 1024 * 128 = 4,194,304
    {{4, 8, 1024, 128}, 1e-3f, 0.9f, 0.0f, 1e-4f, true, "B4H8_S1024_C128_Nesterov_L2"},

    // 2 * 16 * 2048 * 128 = 8,388,608
    {{2, 16, 2048, 128}, 5e-4f, 0.9f, 0.1f, 0.0f, false, "B2H16_S2048_C128_Mom_Damp"},
};

INSTANTIATE_TEST_SUITE_P(SGDFusedVsCPUParity_Big, SGDFusedParityTest, ::testing::ValuesIn(kBigCases), CaseName);

// --- HUGE test cases ----------------------------------------------
static const ParityCase kHugeCases[] = {
    // ~1.0M params vector — fast and stresses contiguous path
    {{1, 1, 1, 1'048'576}, 1e-3f, 0.0f, 0.0f, 0.0f, false, "Vec1M_SGD"},

    // A bigger transformer-like block (~16.7M elems):
    // 4 * 16 * 2048 * 128 = 16,777,216
    {{4, 16, 2048, 128}, 1e-3f, 0.9f, 0.0f, 1e-4f, true, "B4H16_S2048_C128_Nesterov_L2"},

    // Wide MLP-style layer (~67.1M elems):
    // 8 * 16 * 2048 * 256 = 67,108,864
    {{8, 16, 2048, 256}, 5e-4f, 0.9f, 0.0f, 1e-4f, true, "B8H16_S2048_C256_Nesterov_L2"},

    // Long-sequence stress (~134.2M elems):
    // 4 * 32 * 4096 * 256 = 134,217,728
    {{4, 32, 4096, 256}, 1e-2f, 0.9f, 0.0f, 1e-3f, true, "B4H32_S4096_C256_Nesterov_L2"},
};

INSTANTIATE_TEST_SUITE_P(SGDFusedVsCPUParity_Huge, SGDFusedParityTest, ::testing::ValuesIn(kHugeCases), CaseName);
