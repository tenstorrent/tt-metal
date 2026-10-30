// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/sgd.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/sgd_composite.hpp"
#include "test_utils/program_cache.hpp"
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
// Param groups
// Each group is stepped with its own options: here "b" is in a group with
// 10x the default lr, so after one plain SGD step (w -= lr * g) the two
// params must reflect different learning rates.
// ====================================================================

using SGDParamGroupsTest = SGDLateMomentumTest;

template <typename Optimizer, typename Config>
static void run_group_lr_case() {
    using namespace ttml;

    const std::array<std::size_t, 4> shape = {1, 1, 32, 32};
    const float default_lr = 1e-2F;
    const float group_lr = 1e-1F;

    autograd::ctx().set_seed(123U);
    auto& gen = autograd::ctx().get_generator();
    xt::xarray<float> w_a = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, gen());
    xt::xarray<float> w_b = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, gen());
    // Bounded away from zero so the two learning rates give clearly different updates everywhere.
    xt::xarray<float> g = test_utils::make_uniform_xarray<float>(shape, 0.5F, 1.0F, gen());

    auto theta_a = autograd::create_tensor(to_tt(w_a), true);
    auto theta_b = autograd::create_tensor(to_tt(w_b), true);
    ttml::serialization::NamedParameters params{{"a", theta_a}, {"b", theta_b}};

    Config cfg;
    cfg.lr = default_lr;
    Optimizer opt(params, cfg, {{.params = {"b"}, .options = {{"lr", group_lr}}}});
    ASSERT_EQ(opt.get_group_lrs(), (std::vector<float>{default_lr, group_lr}));

    const auto w_a_before = core::to_xtensor(theta_a->get_value());
    const auto w_b_before = core::to_xtensor(theta_b->get_value());
    theta_a->set_grad(to_tt(g));
    theta_b->set_grad(to_tt(g));
    const auto grad = core::to_xtensor(theta_a->get_grad());
    opt.step();

    const auto w_a_after = core::to_xtensor(theta_a->get_value());
    const auto w_b_after = core::to_xtensor(theta_b->get_value());
    const float tolerance = 1e-2F;  // bf16 rounding of the updated weights
    EXPECT_LE(xt::amax(xt::abs(w_a_after - (w_a_before - default_lr * grad)))(), tolerance)
        << "param in the default group must use the default lr";
    EXPECT_LE(xt::amax(xt::abs(w_b_after - (w_b_before - group_lr * grad)))(), tolerance)
        << "param in the custom group must use the group's lr";
    // Guard against a vacuous pass: the two learning rates must give clearly different updates.
    EXPECT_GT(xt::amin(xt::abs((group_lr - default_lr) * grad))(), 2 * tolerance);
}

TEST_F(SGDParamGroupsTest, FusedStepsEachGroupWithItsOwnLr) {
    run_group_lr_case<ttml::optimizers::SGD, ttml::optimizers::SGDConfig>();
}

TEST_F(SGDParamGroupsTest, CompositeStepsEachGroupWithItsOwnLr) {
    run_group_lr_case<ttml::optimizers::SGDComposite, ttml::optimizers::SGDCompositeConfig>();
}

TEST_F(SGDParamGroupsTest, ValidatesEveryGroup) {
    using namespace ttml;
    auto theta = autograd::create_tensor(to_tt(xt::zeros<float>({1, 1, 32, 32})), true);
    ttml::serialization::NamedParameters params{{"theta", theta}};

    // The defaults are valid, but a group with dampening and no momentum is not.
    optimizers::SGDConfig cfg;
    EXPECT_THROW(
        optimizers::SGD(params, cfg, {{.params = {"theta"}, .options = {{"dampening", 0.5F}}}}), std::runtime_error);

    // Nesterov applies to the whole optimizer, so every group needs momentum > 0 and no dampening.
    cfg.nesterov = true;
    cfg.momentum = 0.9F;
    EXPECT_THROW(
        optimizers::SGD(params, cfg, {{.params = {"theta"}, .options = {{"momentum", 0.0F}}}}), std::runtime_error);

    // Options changed after construction are checked before they are used: at the next step.
    optimizers::SGD valid(params, cfg);
    valid.set_group_options("momentum", {0.0F});
    EXPECT_THROW(valid.step(), std::runtime_error);
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

// ====================================================================
// Program-cache hits
// lr, momentum, dampening and weight decay are not part of the program hash, so a hit must re-apply them, and the
// use-weight-decay and use-dampening flags derived from them, with the new buffer addresses. Each test starts from an
// empty program cache, keeps the tensors of its earlier launches alive and compares every launch with SGDComposite.
// ====================================================================

class SGDCacheHitTest : public ttml::test_utils::ProgramCacheSuiteTest {};

namespace {

// The arguments of ttml::metal::sgd that are left out of the program hash.
struct SGDStepScalars {
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
};

// A device tensor and the host values it was made from. Parameters, gradients and momentum buffers are random and
// nonzero, so every scalar shows in the result.
struct HostAndDeviceTensor {
    xt::xarray<float> values;
    ttnn::Tensor tensor;
};

HostAndDeviceTensor upload_uniform(const std::vector<uint32_t>& shape, uint32_t seed) {
    auto values = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed);
    auto tensor = to_tt(values);
    return {std::move(values), std::move(tensor)};
}

// The tensors of one launch.
struct SGDLaunchTensors {
    HostAndDeviceTensor param;
    HostAndDeviceTensor grad;
    std::optional<HostAndDeviceTensor> momentum_buffer;
};

SGDLaunchTensors make_sgd_launch_tensors(const std::vector<uint32_t>& shape, bool with_momentum_buffer, uint32_t seed) {
    SGDLaunchTensors tensors{upload_uniform(shape, seed), upload_uniform(shape, seed + 1U), std::nullopt};
    if (with_momentum_buffer) {
        tensors.momentum_buffer = upload_uniform(shape, seed + 2U);
    }
    return tensors;
}

ttnn::Tensor launch_sgd(const SGDLaunchTensors& tensors, const SGDStepScalars& scalars, bool nesterov) {
    std::optional<ttnn::Tensor> momentum_buffer;
    if (tensors.momentum_buffer.has_value()) {
        momentum_buffer = tensors.momentum_buffer->tensor;
    }
    return ttml::metal::sgd(
        tensors.param.tensor,
        tensors.grad.tensor,
        scalars.lr,
        scalars.momentum,
        scalars.dampening,
        scalars.weight_decay,
        nesterov,
        momentum_buffer);
}

// The tensors a launch updates.
struct SGDValues {
    xt::xarray<float> param;
    std::optional<xt::xarray<float>> momentum;  // with a momentum buffer only
};

SGDValues read_back(const SGDLaunchTensors& tensors) {
    SGDValues values{ttml::core::to_xtensor(tensors.param.tensor), std::nullopt};
    if (tensors.momentum_buffer.has_value()) {
        values.momentum = ttml::core::to_xtensor(tensors.momentum_buffer->tensor);
    }
    return values;
}

// SGDComposite's step on fresh copies of a launch's inputs. The momentum buffer is restored through the state dict,
// so the composite treats it as past its first update and applies momentum and dampening to it, as the fused kernel
// does with any buffer it is given.
SGDValues composite_reference(const SGDLaunchTensors& tensors, const SGDStepScalars& scalars, bool nesterov) {
    using namespace ttml;
    auto theta = autograd::create_tensor(to_tt(tensors.param.values), true);
    theta->set_grad(to_tt(tensors.grad.values));

    optimizers::SGDCompositeConfig cfg;
    cfg.lr = scalars.lr;
    cfg.momentum = scalars.momentum;
    cfg.dampening = scalars.dampening;
    cfg.weight_decay = scalars.weight_decay;
    cfg.nesterov = nesterov;
    optimizers::SGDComposite opt(serialization::NamedParameters{{"theta", theta}}, cfg);
    if (tensors.momentum_buffer.has_value()) {
        auto state = opt.get_state_dict();
        state["theta"] = serialization::NamedParameters{
            {"theta", autograd::create_tensor(to_tt(tensors.momentum_buffer->values), false)}};
        opt.set_state_dict(state);
    }
    opt.step();

    SGDValues values{core::to_xtensor(theta->get_value()), std::nullopt};
    if (tensors.momentum_buffer.has_value()) {
        const auto state = opt.get_state_dict();
        const auto& buffers = std::get<serialization::NamedParameters>(state.at("theta"));
        values.momentum = core::to_xtensor(buffers.at("theta")->get_value());
    }
    return values;
}

void expect_bitwise_equal(
    const SGDValues& expected, const SGDValues& actual, const SGDLaunchTensors& tensors, const std::string& label) {
    ASSERT_EQ(expected.param.shape(), actual.param.shape()) << label;
    EXPECT_EQ(compare_tensors(expected.param, actual.param, tensors.param.values, tensors.grad.values), 0U)
        << label << ": param";
    ASSERT_EQ(expected.momentum.has_value(), actual.momentum.has_value()) << label;
    if (expected.momentum.has_value()) {
        ASSERT_EQ(expected.momentum->shape(), actual.momentum->shape()) << label;
        EXPECT_EQ(compare_tensors(*expected.momentum, *actual.momentum, tensors.param.values, tensors.grad.values), 0U)
            << label << ": momentum buffer";
    }
}

void expect_distinct_buffers(const SGDLaunchTensors& a, const SGDLaunchTensors& b) {
    EXPECT_NE(a.param.tensor.buffer()->address(), b.param.tensor.buffer()->address());
    EXPECT_NE(a.grad.tensor.buffer()->address(), b.grad.tensor.buffer()->address());
    if (a.momentum_buffer.has_value() && b.momentum_buffer.has_value()) {
        EXPECT_NE(a.momentum_buffer->tensor.buffer()->address(), b.momentum_buffer->tensor.buffer()->address());
    }
}

// Launch A compiles the program. Launch B updates fresh buffers with every hash-excluded scalar changed. Launch C
// updates fresh copies of launch A's buffers with launch A's scalars and must reproduce launch A bit for bit.
void run_new_scalars_on_fresh_buffers(bool nesterov, const SGDStepScalars& scalars_a, const SGDStepScalars& scalars_b) {
    auto& device = ttml::autograd::ctx().get_device();
    const auto shape = ttml::test_utils::two_core_group_rows_shape(32U);
    constexpr uint32_t kSeedA = 100U;

    const auto a = make_sgd_launch_tensors(shape, /*with_momentum_buffer=*/true, kSeedA);
    const auto entries_before_a = device.num_program_cache_entries();
    const auto a_out = launch_sgd(a, scalars_a, nesterov);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_a) << "launch A must compile a program";
    EXPECT_EQ(a_out.buffer()->address(), a.param.tensor.buffer()->address()) << "SGD updates the parameter in place";
    const auto a_after = read_back(a);
    expect_bitwise_equal(composite_reference(a, scalars_a, nesterov), a_after, a, "launch A");

    const auto b = make_sgd_launch_tensors(shape, /*with_momentum_buffer=*/true, 200U);
    const auto entries_before_b = device.num_program_cache_entries();
    launch_sgd(b, scalars_b, nesterov);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_b) << "launch B must reuse launch A's program";
    expect_distinct_buffers(a, b);
    const auto b_after = read_back(b);
    expect_bitwise_equal(composite_reference(b, scalars_b, nesterov), b_after, b, "launch B (cache hit, new scalars)");

    const auto c = make_sgd_launch_tensors(shape, /*with_momentum_buffer=*/true, kSeedA);
    const auto entries_before_c = device.num_program_cache_entries();
    launch_sgd(c, scalars_a, nesterov);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    expect_distinct_buffers(a, c);
    expect_distinct_buffers(b, c);
    expect_bitwise_equal(a_after, read_back(c), c, "launch C (launch A again, cache hit)");

    expect_bitwise_equal(a_after, read_back(a), a, "launch A's buffers after the later launches");
    expect_bitwise_equal(b_after, read_back(b), b, "launch B's buffers after launch C");
}

}  // namespace

// Launch A runs without dampening or weight decay, so launch B also switches both compute flags on.
TEST_F(SGDCacheHitTest, MomentumNewScalarsOnFreshBuffers_TwoCoreGroups) {
    run_new_scalars_on_fresh_buffers(
        /*nesterov=*/false,
        /*scalars_a=*/{1e-2f, 0.9f, 0.0f, 0.0f},
        /*scalars_b=*/{1e-1f, 0.5f, 0.5f, 1.0f});
}

// Nesterov momentum requires zero dampening (SGD::validate_config).
TEST_F(SGDCacheHitTest, NesterovNewScalarsOnFreshBuffers_TwoCoreGroups) {
    run_new_scalars_on_fresh_buffers(
        /*nesterov=*/true,
        /*scalars_a=*/{1e-2f, 0.9f, 0.0f, 0.0f},
        /*scalars_b=*/{1e-1f, 0.5f, 0.0f, 1.0f});
}

// Without a momentum buffer the op builds a program that neither reads nor writes one, and nesterov momentum, which the
// program hash keys, builds a third. The launches alternate between the three cached programs on fresh buffers, with
// new scalars on every hit.
TEST_F(SGDCacheHitTest, PlainMomentumAndNesterovProgramsAlternate_TwoCoreGroups) {
    using ttml::test_utils::CacheExpectation;
    using ttml::test_utils::expect_cache;
    auto& device = ttml::autograd::ctx().get_device();
    const auto shape = ttml::test_utils::two_core_group_rows_shape(32U);

    struct Launch {
        bool momentum_buffer;
        bool nesterov;
        SGDStepScalars scalars;
        bool cache_hit;
        const char* name;
    };
    const std::array<Launch, 6> launches = {{
        {false, false, {1.0f, 0.0f, 0.0f, 0.0f}, false, "launch A (plain, miss)"},
        {true, false, {1e-2f, 0.9f, 0.0f, 0.0f}, false, "launch B (momentum, miss)"},
        {true, true, {1e-2f, 0.9f, 0.0f, 0.0f}, false, "launch C (nesterov, miss)"},
        {false, false, {1e-1f, 0.0f, 0.0f, 1.0f}, true, "launch D (plain, cache hit)"},
        {true, false, {1e-1f, 0.5f, 0.5f, 1.0f}, true, "launch E (momentum, cache hit)"},
        {true, true, {1e-1f, 0.5f, 0.0f, 1.0f}, true, "launch F (nesterov, cache hit)"},
    }};

    std::vector<SGDLaunchTensors> tensors;
    std::vector<SGDValues> after;
    tensors.reserve(launches.size());
    after.reserve(launches.size());
    for (uint32_t i = 0; i < launches.size(); ++i) {
        const auto& launch = launches[i];
        auto launch_tensors = make_sgd_launch_tensors(shape, launch.momentum_buffer, 300U + 10U * i);
        for (const auto& earlier : tensors) {
            expect_distinct_buffers(earlier, launch_tensors);
        }
        const auto entries_before = device.num_program_cache_entries();
        launch_sgd(launch_tensors, launch.scalars, launch.nesterov);
        expect_cache(entries_before, launch.cache_hit ? CacheExpectation::Hit : CacheExpectation::Miss, launch.name);
        after.push_back(read_back(launch_tensors));
        expect_bitwise_equal(
            composite_reference(launch_tensors, launch.scalars, launch.nesterov),
            after.back(),
            launch_tensors,
            launch.name);
        tensors.push_back(std::move(launch_tensors));
    }
    for (uint32_t i = 0; i < launches.size(); ++i) {
        expect_bitwise_equal(
            after[i],
            read_back(tensors[i]),
            tensors[i],
            std::string(launches[i].name) + ": buffers after the later launches");
    }
}
