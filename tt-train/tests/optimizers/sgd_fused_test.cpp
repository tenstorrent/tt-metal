// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/sgd_fused.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <bit>
#include <core/ttnn_all_includes.hpp>
#include <cstdlib>
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

[[maybe_unused]] static xt::xarray<float> to_bf16(const xt::xarray<float>& x) {
    xt::xarray<float> result = xt::empty_like(x);
    for (size_t i = 0; i < x.size(); ++i) {
        uint32_t bits = std::bit_cast<uint32_t>(x(i));
        // BF16: keep sign (1 bit) + exponent (8 bits) + top 7 bits of mantissa, zero out lower 16 bits
        uint32_t bf16_bits = bits & 0xFFFF0000u;
        result(i) = std::bit_cast<float>(bf16_bits);
    }
    return result;
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
    bool has_momentum,
    size_t step) {
    // Debug: Print first few elements before update
    fmt::print("[CPU Reference] step={}\n", step);
    fmt::print("  param[0:3]: [{:.8f}, {:.8f}, {:.8f}]\n", param(0), param(1), param(2));
    fmt::print("  grad[0:3]:  [{:.8f}, {:.8f}, {:.8f}]\n", grad(0), grad(1), grad(2));
    if (has_momentum) {
        fmt::print(
            "  mom_before[0:3]: [{:.8f}, {:.8f}, {:.8f}]\n",
            momentum_buffer(0),
            momentum_buffer(1),
            momentum_buffer(2));
    }

    for (size_t i = 0; i < param.size(); ++i) {
        float theta = param(i);
        float g_orig = grad(i);
        float m_prev = (has_momentum ? momentum_buffer(i) : 0.0f);

        // Apply weight decay to gradient
        float g_t = g_orig;
        if (weight_decay != 0.0f)
            g_t += weight_decay * theta;

        // Momentum buffer update (step-conditional)
        float m_t = g_t;
        if (has_momentum && momentum_val != 0.0f) {
            if (step == 0) {
                // First step: momentum buffer = gradient (no dampening)
                m_t = g_t;
            } else {
                // Subsequent steps: apply momentum and dampening
                m_t = momentum_val * m_prev;
                if (dampening != 0.0f) {
                    m_t += (1.0f - dampening) * g_t;
                } else {
                    m_t += g_t;
                }
            }
        }

        // Compute update (Nesterov vs standard)
        float update = nesterov ? (g_t + momentum_val * m_t) : m_t;

        // Update parameter
        float theta_new = theta - lr * update;

        param(i) = theta_new;
        if (has_momentum && momentum_val != 0.0f) {
            momentum_buffer(i) = m_t;
        }
    }

    // Debug: Print updated values
    fmt::print("  param_after[0:3]: [{:.8f}, {:.8f}, {:.8f}]\n", param(0), param(1), param(2));
    if (has_momentum) {
        fmt::print(
            "  mom_after[0:3]:  [{:.8f}, {:.8f}, {:.8f}]\n",
            momentum_buffer(0),
            momentum_buffer(1),
            momentum_buffer(2));
    }
    fmt::print("\n");
}

static void run_one_step_and_compare(const ParityCase& pc) {
    using namespace ttml;

    ttml::autograd::ctx().set_seed(0U);
    auto& g = autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();
    const uint32_t seed_mom = g();

    // Same data used for all optimizers
    xt::xarray<float> w0 = make_random_xarray(pc.shape, seed_param);
    xt::xarray<float> g0 = make_random_xarray(pc.shape, seed_grad);

    xt::xarray<float> w_cpu = w0;
    xt::xarray<float> g_cpu = g0;
    xt::xarray<float> mom_cpu = xt::zeros_like(w0);
    if (pc.momentum > 0.0f) {
        mom_cpu = make_random_xarray(pc.shape, seed_mom);
    }

    // Build autograd tensor for fused optimizer
    auto theta_fused = autograd::create_tensor(to_tt(w0), true);
    theta_fused->set_grad(to_tt(g0));

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
    opt_fused.set_steps(1);

    // If momentum is used, initialize the momentum buffer with the same values
    if (pc.momentum > 0.0f) {
        auto sd_fused = opt_fused.get_state_dict();
        auto mom_fused = std::get<ttml::serialization::NamedParameters>(sd_fused.at("momentum"));
        mom_fused["theta"]->set_value(to_tt(mom_cpu));
        opt_fused.set_state_dict(sd_fused);
    }

    // Build autograd tensor for composite optimizer
    auto theta_comp = autograd::create_tensor(to_tt(w0), true);
    theta_comp->set_grad(to_tt(g0));

    // Named parameters map for composite optimizer
    ttml::serialization::NamedParameters params_comp{{"theta", theta_comp}};

    // Build composite config
    ttml::optimizers::SGDConfig comp_cfg;
    comp_cfg.lr = pc.lr;
    comp_cfg.momentum = pc.momentum;
    comp_cfg.dampening = pc.dampening;
    comp_cfg.weight_decay = pc.weight_decay;
    comp_cfg.nesterov = pc.nesterov;

    // Create composite optimizer
    ttml::optimizers::SGD opt_comp(params_comp, comp_cfg);
    opt_comp.set_steps(1);

    // If momentum is used, initialize the momentum buffer with the same values
    if (pc.momentum > 0.0f) {
        auto sd_comp = opt_comp.get_state_dict();
        auto mom_comp = std::get<ttml::serialization::NamedParameters>(sd_comp.at("theta"));
        mom_comp["theta"]->set_value(to_tt(mom_cpu));
        opt_comp.set_state_dict(sd_comp);
    }

    // Run optimizer steps
    opt_fused.step();
    opt_comp.step();

    // CPU optimizer step (note: both fused and comp optimizers have m_steps=1 when step() is called,
    // so they execute the non-first-step path. For proper parity, we pass step=1 to CPU reference)
    cpu_sgd_step(
        w_cpu, g_cpu, mom_cpu, pc.lr, pc.momentum, pc.dampening, pc.weight_decay, pc.nesterov, pc.momentum > 0.0f, 1);

    // Pull fused and composite params
    const auto w_fused = ttml::core::to_xtensor(theta_fused->get_value());
    const auto w_comp = ttml::core::to_xtensor(theta_comp->get_value());
    ASSERT_EQ(w_fused.shape(), w_cpu.shape());
    ASSERT_EQ(w_comp.shape(), w_cpu.shape());

    // Calculate errors relative to CPU (ground truth)
    auto err_fused = xt::abs(w_fused - w_cpu);
    auto err_comp = xt::abs(w_comp - w_cpu);
    float max_err_fused = xt::amax(err_fused)();
    float max_err_comp = xt::amax(err_comp)();
    float mean_err_fused = xt::mean(err_fused)();
    float mean_err_comp = xt::mean(err_comp)();

    // Check for plot mode environment variable
    const char* plot_mode_env = std::getenv("PLOT_MODE");
    bool plot_mode = (plot_mode_env != nullptr && std::string(plot_mode_env) == "1");

    if (plot_mode) {
        // Mark start of plot data block
        fmt::print("===PLOT_DATA_START===\n");

        // Print summary statistics
        fmt::print(
            "# Case: {} | max_err_fused={:.6e} | max_err_comp={:.6e} | mean_err_fused={:.6e} | mean_err_comp={:.6e}\n",
            pc.name,
            max_err_fused,
            max_err_comp,
            mean_err_fused,
            mean_err_comp);

        // Print header
        fmt::print("# idx\terr_fused\terr_comp\tparam\tgrad\tfused_val\tcomp_val\tcpu_val\n");

        // Print all data points in tab-separated format
        for (size_t i = 0; i < w_fused.size(); ++i) {
            float ef = std::fabs(w_fused(i) - w_cpu(i));
            float ec = std::fabs(w_comp(i) - w_cpu(i));
            fmt::print(
                "{}\t{:.10e}\t{:.10e}\t{:.10e}\t{:.10e}\t{:.10e}\t{:.10e}\t{:.10e}\n",
                i,
                ef,
                ec,
                w0(i),
                g0(i),
                w_fused(i),
                w_comp(i),
                w_cpu(i));
        }

        // Mark end of plot data block
        fmt::print("===PLOT_DATA_END===\n");
    } else {
        fmt::print(
            "Case '{}': Fused max_err={:.6e}, mean_err={:.6e} | Comp max_err={:.6e}, mean_err={:.6e}\n",
            pc.name,
            max_err_fused,
            mean_err_fused,
            max_err_comp,
            mean_err_comp);
    }

    // Print diagnostic info if fused performs worse than composite
    if (!plot_mode && (max_err_fused > max_err_comp || mean_err_fused > mean_err_comp)) {
        fmt::print("Fused performance is worse than composite - printing diagnostics:\n");

        // Collect all cases where fused is worse than composite and sort by error
        struct ErrorCase {
            size_t idx;
            float param;
            float grad;
            float fused_val;
            float comp_val;
            float cpu_val;
            float err_fused;
            float err_comp;
        };
        std::vector<ErrorCase> error_cases;
        for (size_t i = 0; i < w_fused.size(); ++i) {
            float err_fused = std::fabs(w_fused(i) - w_cpu(i));
            float err_comp = std::fabs(w_comp(i) - w_cpu(i));
            if (err_fused > err_comp) {
                error_cases.push_back({i, w0(i), g0(i), w_fused(i), w_comp(i), w_cpu(i), err_fused, err_comp});
            }
        }

        // Sort by fused error in descending order
        std::sort(error_cases.begin(), error_cases.end(), [](const ErrorCase& a, const ErrorCase& b) {
            return a.err_fused > b.err_fused;
        });

        fmt::print(
            "Elements where fused error > composite error: {} / {} ({:.2f}%)\n",
            error_cases.size(),
            w_fused.size(),
            (100.0 * error_cases.size()) / w_fused.size());

        // Print top 8 highest errors
        fmt::print("8 highest errors where fused error > composite error:\n");
        size_t to_print = std::min(size_t(8), error_cases.size());
        for (size_t i = 0; i < to_print; ++i) {
            const auto& ec = error_cases[i];
            fmt::print(
                "idx {:6d} | param={: 12.8f} | grad={: 12.8f} | fused={: 12.8f} | comp={: 12.8f} | cpu={: 12.8f} | "
                "err_fused={:10.3e} | err_comp={:10.3e}\n",
                static_cast<int>(ec.idx),
                ec.param,
                ec.grad,
                ec.fused_val,
                ec.comp_val,
                ec.cpu_val,
                ec.err_fused,
                ec.err_comp);
        }
    }

    // Ensure fused error is always better than (or equal to) composite error

    EXPECT_LE(mean_err_fused, mean_err_comp)
        << "Fused mean error should be <= composite mean error (case '" << pc.name << "'). "
        << "Fused: " << mean_err_fused << ", Composite: " << mean_err_comp;

    /*
    if (pc.momentum > 0.0f) {
        auto sd_fused = opt_fused.get_state_dict();
        auto mom_fused = std::get<ttml::serialization::NamedParameters>(sd_fused.at("momentum"));
        ASSERT_TRUE(mom_fused.count("theta"));

        const auto m_fused = ttml::core::to_xtensor(mom_fused.at("theta")->get_value());
    }
    */
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
    {{1, 1, 1, 32}, 1e-3f, 0.0f, 0.0f, 0.0f, false, "Vanilla_Vec32"},
    {{1, 1, 1, 64}, 1e-3f, 0.0f, 0.0f, 0.0f, false, "Vanilla_Vec64"},
    {{1, 1, 1, 64}, 1e-3f, 0.0f, 0.0f, 1e-2f, false, "L2_Vec64"},
    {{1, 1, 32, 64}, 1e-3f, 0.9f, 0.0f, 0.0f, false, "Mom_Seq32x64"},
    {{1, 1, 32, 64}, 1e-3f, 0.9f, 0.1f, 0.0f, false, "Mom_Damp_Seq32x64"},
    {{2, 1, 32, 128}, 1e-3f, 0.9f, 0.0f, 0.0f, true, "Nesterov_B2_Seq32x128"},
    {{1, 4, 32, 32}, 5e-4f, 0.9f, 0.0f, 1e-3f, true, "Nesterov_L2_H4_Seq32x32"},
};

INSTANTIATE_TEST_SUITE_P(SGDFusedVsCPUParity, SGDFusedParityTest, ::testing::ValuesIn(kCases), CaseName);

static const ParityCase kBigCases[] = {
    // A "fatter" single vector (~262k params)
    {{1, 1, 1, 262'144}, 1e-2f, 0.0f, 0.0f, 0.0f, false, "Vanilla_Vec262k"},

    {{4, 8, 1024, 128}, 1e-3f, 0.0f, 0.0f, 1e-4f, false, "Vanilla_L2_B4H8_S1024_C128"},
    // Transformer-like blocks in the low millions of elements
    // 4 * 8 * 1024 * 128 = 4,194,304
    //{{4, 8, 1024, 128}, 1e-1f, 0.99f, 0.0f, 0.0f, true, "Nesterov_B4H8_S1024_C128"},
    {{4, 8, 1024, 128}, 1e-3f, 0.9f, 0.0f, 1e-4f, true, "Nesterov_L2_B4H8_S1024_C128"},

    {{4, 8, 1024, 128}, 1e-3f, 0.9f, 0.0f, 0.0f, false, "Mom_B4H8_S1024_C128"},

    // 2 * 16 * 2048 * 128 = 8,388,608
    {{2, 16, 2048, 128}, 5e-4f, 0.9f, 0.1f, 0.0f, false, "Mom_Damp_B2H16_S2048_C128"},

    {{2, 16, 2048, 128}, 5e-4f, 0.9f, 0.1f, 1e-4f, false, "Mom_Damp_L2_B2H16_S2048_C128"},
};

INSTANTIATE_TEST_SUITE_P(SGDFusedVsCPUParity_Big, SGDFusedParityTest, ::testing::ValuesIn(kBigCases), CaseName);

// --- HUGE test cases ----------------------------------------------
static const ParityCase kHugeCases[] = {
    // ~1.0M params vector — fast and stresses contiguous path
    {{1, 1, 1, 1'048'576}, 1e-2f, 0.0f, 0.0f, 0.0f, false, "Vanilla_Vec1M"},

    // A bigger transformer-like block (~16.7M elems):
    // 4 * 16 * 2048 * 128 = 16,777,216
    {{4, 16, 2048, 128}, 1e-1f, 0.99f, 0.0f, 1e-3f, true, "Nesterov_L2_B4H16_S2048_C128_1"},
    {{4, 16, 2048, 128}, 1e-2f, 0.99f, 0.0f, 1e-3f, true, "Nesterov_L2_B4H16_S2048_C128_2"},
    {{4, 16, 2048, 128}, 1e-3f, 0.99f, 0.0f, 1e-3f, true, "Nesterov_L2_B4H16_S2048_C128_3"},
    {{4, 16, 2048, 128}, 1e-4f, 0.99f, 0.0f, 1e-3f, true, "Nesterov_L2_B4H16_S2048_C128_4"},
    {{4, 16, 2048, 128}, 1e-5f, 0.99f, 0.0f, 1e-3f, true, "Nesterov_L2_B4H16_S2048_C128_5"},

    {{4, 16, 2048, 128}, 1e-1f, 0.9f, 0.0f, 0.0f, true, "Nesterov_B4H16_S2048_C128_1"},
    {{4, 16, 2048, 128}, 1e-2f, 0.9f, 0.0f, 0.0f, true, "Nesterov_B4H16_S2048_C128_2"},
    {{4, 16, 2048, 128}, 1e-3f, 0.9f, 0.0f, 0.0f, true, "Nesterov_B4H16_S2048_C128_3"},
    {{4, 16, 2048, 128}, 1e-4f, 0.9f, 0.0f, 0.0f, true, "Nesterov_B4H16_S2048_C128_4"},
    {{4, 16, 2048, 128}, 1e-5f, 0.9f, 0.0f, 0.0f, true, "Nesterov_B4H16_S2048_C128_5"},

    // Wide MLP-style layer (~67.1M elems):
    // 8 * 16 * 2048 * 256 = 67,108,864
    {{8, 16, 2048, 256}, 5e-4f, 0.9f, 0.0f, 1e-4f, true, "Nesterov_L2_B8H16_S2048_C256"},
    {{8, 16, 2048, 256}, 1e-1f, 0.9f, 0.0f, 1e-4f, true, "Anotha_Nesterov_L2_B8H16_S2048_C256"},

    // Long-sequence stress (~134.2M elems):
    // 4 * 32 * 4096 * 256 = 134,217,728
    {{4, 32, 4096, 256}, 1e-2f, 0.9f, 0.0f, 1e-3f, true, "Nesterov_L2_B4H32_S4096_C256"},
};

INSTANTIATE_TEST_SUITE_P(SGDFusedVsCPUParity_Huge, SGDFusedParityTest, ::testing::ValuesIn(kHugeCases), CaseName);
