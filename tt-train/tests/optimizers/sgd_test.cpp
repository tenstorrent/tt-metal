// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/sgd.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdlib>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "xtensor/core/xtensor_forward.hpp"

struct ParityCase {
    std::array<std::size_t, 4> shape;  // (B, H, S, C)
    float lr{1e-3f};
    float momentum{0.0f};
    float dampening{0.0f};
    float weight_decay{0.0f};
    bool nesterov{false};
    std::string name;
};

// Custom printer for ParityCase used by gtest to make test output human-readable
void PrintTo(const ParityCase& pc, std::ostream* os) {
    *os << fmt::format("{}: lr={}", pc.name, pc.lr);

    if (pc.momentum != 0.0f)
        *os << " momentum=" << pc.momentum;
    if (pc.dampening != 0.0f)
        *os << " dampening=" << pc.dampening;
    if (pc.weight_decay != 0.0f)
        *os << " weight_decay=" << pc.weight_decay;
    if (pc.nesterov)
        *os << " nesterov=true";
}

class SGDParityTest : public ::testing::TestWithParam<ParityCase> {
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

static ttnn::Tensor to_tt(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor(x, &ttml::autograd::ctx().get_device());
}

static size_t compare_tensors(
    const xt::xarray<float>& expected,
    const xt::xarray<float>& actual,
    const xt::xarray<float>& params,
    const xt::xarray<float>& grads) {
    size_t num_mismatches = 0;
    struct MismatchInfo {
        size_t idx;
        float expected;
        float actual;
        float param;
        float grad;
    };
    std::vector<MismatchInfo> mismatches;
    mismatches.reserve(8);

    for (size_t i = 0; i < actual.size(); ++i) {
        float expected_val = expected(i);
        float actual_val = actual(i);

        if (expected_val != actual_val) {
            if (mismatches.size() < 8) {
                mismatches.push_back({i, expected_val, actual_val, params(i), grads(i)});
            }
            num_mismatches++;
        }
    }

    // Report results
    if (num_mismatches > 0) {
        fmt::print("Number of mismatches: {} out of {}\n", num_mismatches, actual.size());
        fmt::print("First {} mismatches:\n", mismatches.size());
        for (const auto& m : mismatches) {
            fmt::print(
                "  [{}] expected={}, actual={}, param={}, grad={}\n", m.idx, m.expected, m.actual, m.param, m.grad);
        }
    }

    return num_mismatches;
}

static void run_steps_and_compare(const ParityCase& pc, uint32_t steps) {
    using namespace ttml;

    ttml::autograd::ctx().set_seed(123U);
    auto& g = autograd::ctx().get_generator();
    const uint32_t seed_param = g();
    const uint32_t seed_grad = g();

    // Same data used for all optimizers
    xt::xarray<float> g0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_grad);
    xt::xarray<float> w0 = ttml::test_utils::make_uniform_xarray<float>(pc.shape, -1.0F, 1.0F, seed_param);

    xt::xarray<float> g_cpu = g0;
    xt::xarray<float> w_cpu = w0;

    auto theta_fused = autograd::create_tensor(to_tt(w0), true);
    theta_fused->set_grad(to_tt(g0));

    // Named parameters map for fused optimizer
    ttml::serialization::NamedParameters params_fused{{"theta", theta_fused}};

    // Build fused config
    ttml::optimizers::SGDConfig fused_cfg;
    fused_cfg.lr = pc.lr;
    fused_cfg.momentum = pc.momentum;
    fused_cfg.dampening = pc.dampening;
    fused_cfg.weight_decay = pc.weight_decay;
    fused_cfg.nesterov = pc.nesterov;

    // Create fused optimizer
    ttml::optimizers::SGD opt_fused(params_fused, fused_cfg);

    auto theta_ref = autograd::create_tensor(to_tt(w0), true);
    theta_ref->set_grad(to_tt(g0));
    ttml::serialization::NamedParameters params_ref{{"theta", theta_ref}};

    ttml::optimizers::SGDCompositeConfig ref_cfg;
    ref_cfg.lr = pc.lr;
    ref_cfg.momentum = pc.momentum;
    ref_cfg.dampening = pc.dampening;
    ref_cfg.weight_decay = pc.weight_decay;
    ref_cfg.nesterov = pc.nesterov;

    ttml::optimizers::SGDComposite opt_ref(params_ref, ref_cfg);

    // Run both optimizers for the specified number of steps
    for (uint32_t i = 0; i < steps; ++i) {
        opt_fused.step();
        opt_ref.step();
    }

    // Get results
    auto result_fused = theta_fused->get_value();
    auto result_ref = theta_ref->get_value();

    // Convert to CPU for comparison
    auto result_fused_cpu = core::to_xtensor(result_fused);
    auto result_ref_cpu = core::to_xtensor(result_ref);

    // Compare reference SGD result with fused SGD result
    size_t num_mismatches = compare_tensors(result_ref_cpu, result_fused_cpu, w_cpu, g_cpu);

    // Check for exact equality - both optimizers should produce identical results
    EXPECT_EQ(num_mismatches, 0) << "SGD fused results should match reference SGD implementation";

    // If momentum is enabled, also compare momentum buffers
    if (pc.momentum != 0.0f) {
        auto state_dict_fused = opt_fused.get_state_dict();
        auto state_dict_ref = opt_ref.get_state_dict();

        const auto& momentum_fused_params = std::get<serialization::NamedParameters>(state_dict_fused.at("momentum"));
        const auto& momentum_ref_params = std::get<serialization::NamedParameters>(state_dict_ref.at("theta"));

        // Compare momentum buffers for "theta" parameter
        auto momentum_fused = momentum_fused_params.at("theta")->get_value();
        auto momentum_ref = momentum_ref_params.at("theta")->get_value();

        auto momentum_fused_cpu = core::to_xtensor(momentum_fused);
        auto momentum_ref_cpu = core::to_xtensor(momentum_ref);

        size_t momentum_mismatches = compare_tensors(momentum_ref_cpu, momentum_fused_cpu, w_cpu, g_cpu);

        EXPECT_EQ(momentum_mismatches, 0) << "SGD fused momentum buffers should match reference SGD momentum buffers";
    }
}
static std::string CaseName(const ::testing::TestParamInfo<ParityCase>& info) {
    const auto& c = info.param;
    return fmt::format("{}_B{}H{}S{}C{}", c.name, c.shape[0], c.shape[1], c.shape[2], c.shape[3]);
}

TEST_P(SGDParityTest, UpdateParity) {
    const auto& pc = GetParam();
    // Run 2 steps if momentum is enabled, 1 step otherwise
    const uint32_t steps = (pc.momentum != 0.0f) ? 2 : 1;
    run_steps_and_compare(pc, steps);
}

static const ParityCase kVanillaCases[] = {
    {{1, 1, 32, 32}, 1.0f, 0.0f, 0.0f, 0.0f, false, "Vanilla"},
};

INSTANTIATE_TEST_SUITE_P(SGDVanillaParity, SGDParityTest, ::testing::ValuesIn(kVanillaCases), CaseName);

static const ParityCase kVanillaNightlyCases[] = {
    {{1, 1, 1, 262'144}, 1.0f, 0.0f, 0.0f, 0.0f, false, "Vanilla"},
    {{1, 8, 128, 256}, 1e-1f, 0.0f, 0.0f, 0.0f, false, "Vanilla"},
    {{2, 4, 128, 256}, 1e-3f, 0.0f, 0.0f, 0.0f, false, "Vanilla"},
    {{1, 64, 64, 64}, 1e-6f, 0.0f, 0.0f, 0.0f, false, "Vanilla"},
};

INSTANTIATE_TEST_SUITE_P(NIGHTLY_SGDVanillaParity, SGDParityTest, ::testing::ValuesIn(kVanillaNightlyCases), CaseName);

static const ParityCase kWDCases[] = {
    {{1, 1, 32, 32}, 1e-1f, 0.0f, 0.0f, 1.0f, false, "WD"},
};

INSTANTIATE_TEST_SUITE_P(SGDWeightDecayParity, SGDParityTest, ::testing::ValuesIn(kWDCases), CaseName);

static const ParityCase kWDNightlyCases[] = {
    {{1, 1, 1, 262'144}, 1.0f, 0.0f, 0.0f, 1.0f, false, "WD"},
    {{1, 8, 128, 256}, 1e-1f, 0.0f, 0.0f, 1e-1f, false, "WD"},
    {{2, 4, 128, 256}, 1e-2f, 0.0f, 0.0f, 1e-2f, false, "WD"},
    {{1, 64, 64, 64}, 1e-3f, 0.0f, 0.0f, 1e-3f, false, "WD"},
    {{1, 4, 256, 256}, 1e-4f, 0.0f, 0.0f, 1e-4f, false, "WD"},
    {{1, 256, 32, 32}, 1e-5f, 0.0f, 0.0f, 1e-5f, false, "WD"},
};

INSTANTIATE_TEST_SUITE_P(NIGHTLY_SGDWeightDecayParity, SGDParityTest, ::testing::ValuesIn(kWDNightlyCases), CaseName);

static const ParityCase kMomCases[] = {
    {{1, 2, 32, 64}, 1e-2f, 0.9f, 0.0f, 0.0f, false, "Mom"},
};

INSTANTIATE_TEST_SUITE_P(SGDMomentumParity, SGDParityTest, ::testing::ValuesIn(kMomCases), CaseName);

static const ParityCase kMomNightlyCases[] = {
    {{1, 8, 128, 256}, 1e-1f, 0.9f, 0.0f, 0.0f, false, "Mom"},
    {{2, 4, 128, 256}, 1e-1f, 0.95f, 0.0f, 0.0f, false, "Mom"},
    {{1, 16, 128, 128}, 1e-2f, 0.99f, 0.0f, 0.0f, false, "Mom"},
    {{1, 64, 64, 64}, 1e-2f, 0.5f, 0.0f, 0.0f, false, "Mom"},
    {{1, 4, 256, 256}, 1e-3f, 0.9f, 0.0f, 0.0f, false, "Mom"},
    {{1, 256, 32, 32}, 1e-4f, 0.1f, 0.0f, 0.0f, false, "Mom"},
};

INSTANTIATE_TEST_SUITE_P(NIGHTLY_SGDMomentumParity, SGDParityTest, ::testing::ValuesIn(kMomNightlyCases), CaseName);

static const ParityCase kMomWDCases[] = {
    {{1, 2, 32, 64}, 1e-2f, 0.9f, 0.0f, 1e-3f, false, "MomWD"},
};

INSTANTIATE_TEST_SUITE_P(SGDMomentumWeightDecayParity, SGDParityTest, ::testing::ValuesIn(kMomWDCases), CaseName);

static const ParityCase kMomWDNightlyCases[] = {
    {{1, 1, 1, 262'144}, 1e-2f, 0.9f, 0.0f, 1e-2f, false, "MomWD"},
    {{1, 8, 128, 256}, 1e-2f, 0.9f, 0.0f, 1e-3f, false, "MomWD"},
    {{2, 8, 128, 128}, 1e-2f, 0.5f, 0.0f, 1e-4f, false, "MomWD"},
    {{1, 64, 64, 64}, 1e-3f, 0.9f, 0.0f, 1e-3f, false, "MomWD"},
    {{1, 4, 256, 256}, 1e-4f, 0.95f, 0.0f, 1e-5f, false, "MomWD"},
};

INSTANTIATE_TEST_SUITE_P(
    NIGHTLY_SGDMomentumWeightDecayParity, SGDParityTest, ::testing::ValuesIn(kMomWDNightlyCases), CaseName);

static const ParityCase kMomDampCases[] = {
    {{1, 2, 32, 64}, 1e-2f, 0.9f, 0.1f, 0.0f, false, "MomDamp"},
};

INSTANTIATE_TEST_SUITE_P(SGDMomentumDampeningParity, SGDParityTest, ::testing::ValuesIn(kMomDampCases), CaseName);

static const ParityCase kMomDampNightlyCases[] = {
    {{1, 8, 128, 256}, 1e-2f, 0.9f, 0.1f, 0.0f, false, "MomDamp"},
    {{2, 4, 128, 256}, 1e-2f, 0.9f, 0.5f, 0.0f, false, "MomDamp"},
    {{2, 8, 128, 128}, 1e-2f, 0.5f, 0.9f, 0.0f, false, "MomDamp"},
    {{1, 64, 64, 64}, 1e-3f, 0.9f, 0.1f, 0.0f, false, "MomDamp"},
    {{1, 4, 256, 256}, 1e-3f, 0.95f, 0.5f, 0.0f, false, "MomDamp"},
};

INSTANTIATE_TEST_SUITE_P(
    NIGHTLY_SGDMomentumDampeningParity, SGDParityTest, ::testing::ValuesIn(kMomDampNightlyCases), CaseName);

static const ParityCase kMomDampWDCases[] = {
    {{1, 2, 32, 64}, 1e-2f, 0.9f, 0.1f, 1e-3f, false, "MomDampWD"},
};

INSTANTIATE_TEST_SUITE_P(
    SGDMomentumDampeningWeightDecayParity, SGDParityTest, ::testing::ValuesIn(kMomDampWDCases), CaseName);

static const ParityCase kMomDampWDNightlyCases[] = {
    {{1, 8, 128, 256}, 1e-2f, 0.9f, 0.1f, 1e-3f, false, "MomDampWD"},
    {{2, 4, 128, 256}, 1e-2f, 0.9f, 0.5f, 1e-4f, false, "MomDampWD"},
    {{1, 1, 1, 262'144}, 1e-3f, 0.9f, 0.9f, 1e-3f, false, "MomDampWD"},
    {{1, 64, 64, 64}, 1e-3f, 0.95f, 0.1f, 1e-4f, false, "MomDampWD"},
    {{1, 4, 256, 256}, 1e-3f, 0.95f, 0.5f, 1e-3f, false, "MomDampWD"},
};

INSTANTIATE_TEST_SUITE_P(
    NIGHTLY_SGDMomentumDampeningWeightDecayParity,
    SGDParityTest,
    ::testing::ValuesIn(kMomDampWDNightlyCases),
    CaseName);

static const ParityCase kNesterovCases[] = {
    {{1, 2, 32, 64}, 1e-2f, 0.9f, 0.0f, 0.0f, true, "Nesterov"},
};

INSTANTIATE_TEST_SUITE_P(SGDNesterovParity, SGDParityTest, ::testing::ValuesIn(kNesterovCases), CaseName);

static const ParityCase kNesterovNightlyCases[] = {
    {{1, 8, 128, 256}, 1e-1f, 0.9f, 0.0f, 0.0f, true, "Nesterov"},
    {{2, 4, 128, 256}, 1e-1f, 0.95f, 0.0f, 0.0f, true, "Nesterov"},
    {{1, 64, 64, 64}, 1e-2f, 0.99f, 0.0f, 0.0f, true, "Nesterov"},
    {{1, 4, 256, 256}, 1e-3f, 0.5f, 0.0f, 0.0f, true, "Nesterov"},
    {{1, 256, 32, 32}, 1e-4f, 0.1f, 0.0f, 0.0f, true, "Nesterov"},
};

INSTANTIATE_TEST_SUITE_P(
    NIGHTLY_SGDNesterovParity, SGDParityTest, ::testing::ValuesIn(kNesterovNightlyCases), CaseName);

static const ParityCase kNesterovWDCases[] = {
    {{1, 2, 32, 64}, 1e-2f, 0.9f, 0.0f, 1e-3f, true, "NesterovWD"},
};

INSTANTIATE_TEST_SUITE_P(SGDNesterovWeightDecayParity, SGDParityTest, ::testing::ValuesIn(kNesterovWDCases), CaseName);

static const ParityCase kNesterovWDNightlyCases[] = {
    {{1, 1, 1, 262'144}, 1e-2f, 0.9f, 0.0f, 1e-2f, true, "NesterovWD"},
    {{1, 8, 128, 256}, 1e-2f, 0.9f, 0.0f, 1e-3f, true, "NesterovWD"},
    {{2, 8, 128, 128}, 1e-2f, 0.5f, 0.0f, 1e-4f, true, "NesterovWD"},
    {{1, 64, 64, 64}, 1e-3f, 0.95f, 0.0f, 1e-3f, true, "NesterovWD"},
    {{1, 4, 256, 256}, 1e-4f, 0.99f, 0.0f, 1e-5f, true, "NesterovWD"},
};

INSTANTIATE_TEST_SUITE_P(
    NIGHTLY_SGDNesterovWeightDecayParity, SGDParityTest, ::testing::ValuesIn(kNesterovWDNightlyCases), CaseName);

// ====================================================================
// Late momentum-buffer initialization
// A buffer's first update must seed it with the raw gradient (buf = g,
// PyTorch semantics) even when other parameters have already advanced
// the global step count.
// ====================================================================

class SGDLateMomentumTest : public ::testing::Test {
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

// Runs two parameters where only "early" receives gradients for the first two steps and
// "late" receives its first gradient on the third, then verifies the late buffer holds the
// raw gradient despite nonzero dampening.
template <typename Optimizer, typename Config>
static void run_late_momentum_case(const char* buffers_key) {
    using namespace ttml;

    const std::array<std::size_t, 4> shape = {1, 1, 32, 32};
    const float dampening = 0.5f;

    autograd::ctx().set_seed(123U);
    auto& gen = autograd::ctx().get_generator();
    xt::xarray<float> w_early = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, gen());
    xt::xarray<float> g_early = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, gen());
    xt::xarray<float> w_late = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, gen());
    // Bounded away from zero so the raw gradient and its dampened value differ everywhere.
    xt::xarray<float> g_late = test_utils::make_uniform_xarray<float>(shape, 0.25F, 1.0F, gen());

    auto theta_early = autograd::create_tensor(to_tt(w_early), true);
    auto theta_late = autograd::create_tensor(to_tt(w_late), true);
    ttml::serialization::NamedParameters params{{"early", theta_early}, {"late", theta_late}};

    Config cfg;
    cfg.lr = 1e-2f;
    cfg.momentum = 0.9f;
    cfg.dampening = dampening;
    Optimizer opt(params, cfg);

    // Advance the global step with only the early parameter receiving gradients.
    theta_early->set_grad(to_tt(g_early));
    opt.step();
    opt.step();

    // The late parameter's first gradient arrives at a nonzero global step.
    theta_late->set_grad(to_tt(g_late));
    opt.step();

    auto state = opt.get_state_dict();
    const auto& buffers = std::get<ttml::serialization::NamedParameters>(state.at(buffers_key));
    auto buf_late = ttml::core::to_xtensor(buffers.at("late")->get_value());

    auto g_late_bf16 = ttml::core::to_xtensor(theta_late->get_grad());
    size_t num_mismatches = compare_tensors(g_late_bf16, buf_late, w_late, g_late);
    EXPECT_EQ(num_mismatches, 0) << "a buffer's first update must seed it with the raw gradient";

    // Guard against a vacuous pass: the dampened seed must be distinguishable from the raw
    // gradient for every element.
    for (size_t i = 0; i < g_late_bf16.size(); ++i) {
        ASSERT_NE(g_late_bf16.flat(i), (1.0f - dampening) * g_late_bf16.flat(i));
    }
}

TEST_F(SGDLateMomentumTest, FusedFirstLateUpdateSeedsRawGradient) {
    run_late_momentum_case<ttml::optimizers::SGD, ttml::optimizers::SGDConfig>("momentum");
}

TEST_F(SGDLateMomentumTest, CompositeFirstLateUpdateSeedsRawGradient) {
    run_late_momentum_case<ttml::optimizers::SGDComposite, ttml::optimizers::SGDCompositeConfig>("theta");
}

// ====================================================================
// Validation tests
// ====================================================================

using SGDValidationTest = SGDLateMomentumTest;

TEST_F(SGDValidationTest, RejectsLogicalShapeMismatchWithEqualPadding) {
    using namespace ttml;

    // Logical 31x32 and 32x32 both round up to one 32x32 tile, so only logical-shape
    // validation can tell them apart.
    const std::array<std::size_t, 4> param_shape = {1, 1, 32, 32};
    const std::array<std::size_t, 4> grad_shape = {1, 1, 31, 32};
    auto param = to_tt(test_utils::make_uniform_xarray<float>(param_shape, -1.0F, 1.0F, 123U));
    auto grad = to_tt(test_utils::make_uniform_xarray<float>(grad_shape, -1.0F, 1.0F, 124U));

    EXPECT_ANY_THROW(ttml::metal::sgd(
        param,
        grad,
        /* lr */ 1e-2f,
        /* momentum */ 0.0f,
        /* dampening */ 0.0f,
        /* weight_decay */ 0.0f,
        /* nesterov */ false,
        /* momentum_buffer */ std::nullopt));
}
