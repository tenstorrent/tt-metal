// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "core/not_null.hpp"
#include "optimizers/optimizer_base.hpp"
#include "schedulers/cosine_annealing_scheduler.hpp"
#include "schedulers/lambda_scheduler.hpp"
#include "schedulers/linear_scheduler.hpp"
#include "schedulers/sequential_scheduler.hpp"
#include "schedulers/step_scheduler.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {
// Optimizer with no params and one param group per entry of `lrs`, each only carrying an "lr".
class MockOptimizer : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override {
        return "MockOptimizer";
    }
    explicit MockOptimizer(const std::vector<float> &lrs) : OptimizerBase(ttml::serialization::NamedParameters{}) {
        std::vector<ParamGroup> groups;
        groups.reserve(lrs.size() - 1);
        for (auto it = std::next(lrs.begin()); it != lrs.end(); ++it) {
            groups.push_back({.params = {}, .options = {{"lr", *it}}});
        }
        init_param_groups({{"lr", lrs.front()}}, std::move(groups));
    }

    explicit MockOptimizer(float lr) : MockOptimizer(std::vector<float>{lr}) {
    }

    void zero_grad() override {};

    void step() override {};

    [[nodiscard]] serialization::StateDict get_state_dict() const override {
        serialization::StateDict dict;
        save_param_groups(dict);
        return dict;
    }

    void set_state_dict(const serialization::StateDict &dict) override {
        restore_param_groups(dict);
    }

    [[nodiscard]] size_t get_steps() const override {
        return {};
    };
    void set_steps(size_t steps) override {};

    [[nodiscard]] float lr() const {
        return get_group_lrs().at(0);
    }
};
}  // namespace ttml::optimizers

// ----------------------------------
// Tests for LambdaScheduler
// ----------------------------------
TEST(LambdaSchedulerTest, ConstantFactor) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(0.1F);

    // Lambda that keeps the LR constant
    // The learning rate of each parameter group is set to the initial lr times a given function. When last_epoch=-1,
    // sets initial lr as lr.

    ttml::schedulers::LambdaScheduler scheduler(optimizer.get(), [](int epoch) {
        (void)epoch;
        return 0.5F;
    });

    // Construction applies lambda(0) (matches PyTorch's LambdaLR).
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.1F * 0.5F);

    scheduler.step();  // epoch 0
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.1F * 0.5F);

    scheduler.step();  // epoch 1
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.1F * 0.5F);

    scheduler.step();  // epoch 2
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.1F * 0.5F);
}

TEST(LambdaSchedulerTest, VaryingFactor) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(1.0f);

    // Lambda: lr_factor = 1.0 / (epoch+1)
    ttml::schedulers::LambdaScheduler scheduler(optimizer.get(), [](int epoch) { return 1.0F / (epoch + 1); });

    // Before stepping
    EXPECT_FLOAT_EQ(optimizer->lr(), 1.0F);

    scheduler.step();  // epoch 0: factor = 1/1=0.5F
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.5F);

    scheduler.step();  // epoch 1: factor = 1/2=0.5 lr=1.0*0.5=0.5
    EXPECT_FLOAT_EQ(optimizer->lr(), 1.F / 3.F);

    scheduler.step();  // epoch 2: factor = 1/3≈0.3333 lr=1.0*0.3333=0.3333
    EXPECT_NEAR(optimizer->lr(), 1.F / 4.F, 1e-5);

    scheduler.step();  // epoch 3: factor = 1/5=0.2
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.2F);
}

// ----------------------------------
// Tests for StepLRScheduler
// ----------------------------------
TEST(StepLRSchedulerTest, BasicDecay) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);

    // Decrease LR by factor of 0.1 every 3 steps
    ttml::schedulers::StepScheduler scheduler(optimizer.get(), 3, 0.1F);

    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(optimizer->lr(), 0.2F);
        scheduler.step();
    }

    for (int i = 0; i < 3; ++i) {
        // After 3 steps: lr = base_lr * 0.1
        EXPECT_FLOAT_EQ(optimizer->lr(), 0.2F * 0.1F);
        scheduler.step();
    }
    // After 6 steps: lr = base_lr * 0.1^2
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.2F * 0.1F * 0.1F);
}

// ----------------------------------
// Tests for LinearScheduler
// ----------------------------------
TEST(LinearSchedulerTest, DecreasingLR) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);

    // Linearly go from 0.2 to 0.0 in 4 steps
    ttml::schedulers::LinearScheduler scheduler(optimizer.get(), 1.0F, 0.0F, 4);

    // step 1: progress = 1/4=0.25 lr = 0.2 + (0.0-0.2)*0.25 = 0.2 - 0.05=0.15
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.15F);

    // step 2: progress=0.5 lr=0.2+(0.0-0.2)*0.5=0.2-0.1=0.1
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.1F);

    // step 3: progress=0.75 lr=0.2+(0.0-0.2)*0.75=0.2-0.15=0.05
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.05F);

    // step 4: progress=1.0 lr=0.2+(0.0-0.2)*1.0=0.0
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.0f);

    // Extra steps keep it at 0.0
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.0f);
}

TEST(LinearSchedulerTest, ConstructorAppliesStartFactor) {
    // Matches PyTorch's LinearLR: the pre-first-step LR is base_lr * start_factor.
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);
    ttml::schedulers::LinearScheduler scheduler(optimizer.get(), 0.25F, 1.0F, 4);

    EXPECT_FLOAT_EQ(optimizer->lr(), 0.2F * 0.25F);
    EXPECT_FLOAT_EQ(scheduler.get_last_lrs()[0], 0.2F * 0.25F);

    // step 1: factor = 0.25 + 0.75 * 1/4
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.2F * (0.25F + 0.75F * 0.25F));
}

TEST(LinearSchedulerDeathTest, InvalidFactorsRejected) {
    // Bounds match PyTorch's LinearLR: 0 < start_factor <= 1, 0 <= end_factor <= 1.
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(0.1F);
    EXPECT_ANY_THROW(ttml::schedulers::LinearScheduler(optimizer.get(), /*start_factor=*/0.0F, 1.0F, 10));
    EXPECT_ANY_THROW(ttml::schedulers::LinearScheduler(optimizer.get(), /*start_factor=*/-0.1F, 1.0F, 10));
    EXPECT_ANY_THROW(ttml::schedulers::LinearScheduler(optimizer.get(), /*start_factor=*/1.5F, 1.0F, 10));
    EXPECT_ANY_THROW(ttml::schedulers::LinearScheduler(optimizer.get(), 1.0F, /*end_factor=*/-0.1F, 10));
    EXPECT_ANY_THROW(ttml::schedulers::LinearScheduler(optimizer.get(), 1.0F, /*end_factor=*/1.5F, 10));
    // Boundary values are legal: start_factor == 1, end_factor == 0 or 1.
    EXPECT_NO_THROW(ttml::schedulers::LinearScheduler(optimizer.get(), 1.0F, 0.0F, 10));
    EXPECT_NO_THROW(ttml::schedulers::LinearScheduler(optimizer.get(), 1.0F, 1.0F, 10));
}

// ----------------------------------
// Tests for CosineAnnealingScheduler
// ----------------------------------
//
// Closed-form formula (matches PyTorch CosineAnnealingLR):
//   lr(s) = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * s / T_max))
//
// Notable values:
//   s = 0          -> lr = base_lr
//   s = T_max / 2  -> lr = (base_lr + eta_min) / 2
//   s = T_max      -> lr = eta_min
//   s = 2 * T_max  -> lr = base_lr (cosine cycles every 2*T_max steps; no
//                                   modulo / restart is performed)
TEST(CosineAnnealingSchedulerTest, MatchesClosedFormTrajectory) {
    constexpr float kBaseLr = 0.1F;
    constexpr size_t kTMax = 20;
    constexpr float kEtaMin = 1e-4F;

    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(kBaseLr);
    ttml::schedulers::CosineAnnealingScheduler scheduler(optimizer.get(), kTMax, kEtaMin);

    // Before any step, the optimizer keeps its initial LR.
    EXPECT_FLOAT_EQ(optimizer->lr(), kBaseLr);

    // Walk through 2 * T_max + a few extra steps and compare against the
    // closed form at every step.
    for (size_t s = 1; s <= 2 * kTMax + 5; ++s) {
        scheduler.step();
        const float expected =
            kEtaMin +
            0.5F * (kBaseLr - kEtaMin) *
                (1.F + std::cos(static_cast<float>(M_PI) * static_cast<float>(s) / static_cast<float>(kTMax)));

        EXPECT_FLOAT_EQ(optimizer->lr(), expected) << "step " << s;
        EXPECT_EQ(scheduler.get_last_lrs(), std::vector<float>{optimizer->lr()}) << "step " << s;
    }
}

TEST(CosineAnnealingSchedulerTest, LrAtTMaxIsEtaMin) {
    constexpr float kBaseLr = 0.1F;
    constexpr size_t kTMax = 20;
    constexpr float kEtaMin = 1e-4F;

    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(kBaseLr);
    ttml::schedulers::CosineAnnealingScheduler scheduler(optimizer.get(), kTMax, kEtaMin);

    for (size_t i = 0; i < kTMax; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(optimizer->lr(), kEtaMin, 1e-7F);
}

TEST(CosineAnnealingSchedulerTest, LrAtTwoTMaxReturnsToBase) {
    constexpr float kBaseLr = 0.1F;
    constexpr size_t kTMax = 20;
    constexpr float kEtaMin = 1e-4F;

    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(kBaseLr);
    ttml::schedulers::CosineAnnealingScheduler scheduler(optimizer.get(), kTMax, kEtaMin);

    for (size_t i = 0; i < 2 * kTMax; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(optimizer->lr(), kBaseLr, 1e-7F);
}

TEST(CosineAnnealingSchedulerTest, EtaMinDefaultsToZero) {
    constexpr float kBaseLr = 0.1F;
    constexpr size_t kTMax = 20;

    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(kBaseLr);
    // eta_min defaults to 0 -> at step T_max the LR should be exactly 0.
    ttml::schedulers::CosineAnnealingScheduler scheduler(optimizer.get(), kTMax);

    for (size_t i = 0; i < kTMax; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(optimizer->lr(), 0.0F, 1e-7F);
}

TEST(CosineAnnealingSchedulerDeathTest, ZeroTMaxIsRejected) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(0.1F);
    EXPECT_ANY_THROW(ttml::schedulers::CosineAnnealingScheduler(optimizer.get(), /*T_max=*/0));
}

// ----------------------------------
// Tests for SequentialScheduler
// ----------------------------------
TEST(SequentialSchedulerTest, ChainSchedulers) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(1.0f);

    // First: StepLRScheduler for 3 steps (gamma=0.5 every step_size=1)
    auto step_scheduler = std::make_unique<ttml::schedulers::StepScheduler>(optimizer.get(), 1, 0.5F);

    // Then: LinearScheduler for 2 steps from current LR to 0.1
    auto linear_scheduler = std::make_unique<ttml::schedulers::LinearScheduler>(optimizer.get(), 1.0F, 0.1F, 2);

    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> schedulers;
    std::vector<size_t> milestones;
    schedulers.push_back(std::move(step_scheduler));
    schedulers.push_back(std::move(linear_scheduler));
    milestones.push_back(3);
    milestones.push_back(2);
    ttml::schedulers::SequentialScheduler seq_scheduler(optimizer.get(), std::move(schedulers), std::move(milestones));

    // Initial LR = 1.0
    // Run StepLRScheduler for 3 steps:
    // step_scheduler: every step reduces LR by factor 0.5
    seq_scheduler.step();  // 1st step: LR=1.0*0.5=0.5
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.5F);

    seq_scheduler.step();  // 2nd step: LR=0.5*0.5=0.25
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.25F);

    seq_scheduler.step();  // 3rd step: LR=0.25*0.5=0.125
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.125F);

    // total_steps=2, start_lr=0.125, end_lr=0.1
    // step 1: progress=1/2=0.5 lr=1.0+(0.1-1.0)*0.5=0.55
    seq_scheduler.step();
    EXPECT_NEAR(optimizer->lr(), 0.55, 1e-5);

    // step 2: progress=2/2=1.0 lr=1.0+(0.1-1.0)*1.0=0.1 (min lr in linear scheduler)
    seq_scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.1F);

    // Further steps do nothing (we finished all schedulers)
    seq_scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), 0.1F);
}

TEST(SequentialSchedulerTest, WarmupSetup) {
    auto start_lr = 3.e-4F;
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(start_lr);

    // First: LinearScheduler for 10 steps from start_lr / 10 to start_lr
    // (start_factor == 0 is rejected, matching PyTorch's LinearLR).
    const float warmup_start_factor = 1.0F / 10;
    auto warmup_scheduler =
        std::make_unique<ttml::schedulers::LinearScheduler>(optimizer.get(), warmup_start_factor, 1.0F, 10);

    // Then: LinearScheduler for 50 steps from start_lr to 0.1F * start_lr
    auto linear_scheduler = std::make_unique<ttml::schedulers::LinearScheduler>(optimizer.get(), 1.F, 0.1F, 50);

    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> schedulers;
    std::vector<size_t> milestones;
    schedulers.push_back(std::move(warmup_scheduler));
    schedulers.push_back(std::move(linear_scheduler));
    milestones.push_back(10);
    milestones.push_back(50);
    ttml::schedulers::SequentialScheduler seq_scheduler(optimizer.get(), std::move(schedulers), std::move(milestones));

    // Construction already applies the warmup child's start factor, so the
    // first optimizer step runs at start_lr / 10 (not the full start_lr).
    EXPECT_NEAR(optimizer->lr(), start_lr * warmup_start_factor, 1e-9);

    for (int i = 0; i < 10; i++) {
        // Linear warmup: 10 steps from start_lr / 10 to start_lr
        seq_scheduler.step();
        const float warmup_factor = warmup_start_factor + (1.0F - warmup_start_factor) * (i + 1) / 10.F;
        EXPECT_NEAR(optimizer->lr(), start_lr * warmup_factor, 1e-5);
    }
    for (int i = 0; i < 50; i++) {
        // Linear decay: 50 steps from start_lr to 0.1F * start_lr
        seq_scheduler.step();
        EXPECT_NEAR(optimizer->lr(), start_lr * (1.0F - 0.9F * (i + 1) / 50.F), 1e-5);
    }
}

TEST(SequentialSchedulerTest, ConstructionRestoresFirstChildInitialLr) {
    // Children are constructed back-to-back on the same optimizer, so the last
    // child's construction-time LR write would otherwise "win". The chain must
    // restore the FIRST child's initial LR, and the second child must have
    // captured the ORIGINAL base LR (via the optimizer's initial_lr), not the
    // scaled value left behind by the first child's construction.
    constexpr float kBaseLr = 0.1F;
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(kBaseLr);

    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> children;
    children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(
        optimizer.get(), /*start_factor=*/0.1F, /*end_factor=*/1.0F, /*total_steps=*/10));
    // After the line above the optimizer's LR is kBaseLr * 0.1.
    children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(
        optimizer.get(), /*start_factor=*/1.0F, /*end_factor=*/0.5F, /*total_steps=*/10));
    // The second child's construction wrote kBaseLr * 1.0 (its own initial LR,
    // computed from the shared initial_lr rather than the scaled live LR).
    EXPECT_FLOAT_EQ(optimizer->lr(), kBaseLr);

    ttml::schedulers::SequentialScheduler seq(
        optimizer.get(), std::move(children), /*milestones=*/std::vector<size_t>{10U, 10U});

    // Construction restored the first (active) child's initial LR.
    EXPECT_FLOAT_EQ(optimizer->lr(), kBaseLr * 0.1F);

    // Warmup proceeds from the original base LR, proving the first child's
    // base was not polluted either.
    seq.step();
    EXPECT_FLOAT_EQ(optimizer->lr(), kBaseLr * (0.1F + 0.9F * 0.1F));
}

// ----------------------------------
// State-dict round-trip tests
// ----------------------------------
//
// Each test verifies two things:
//   1. The state dict produced by ``get_state_dict`` contains the new
//      hyperparameter keys with the correct values.
//   2. ``set_state_dict`` restores those hyperparameters into a destination
//      scheduler that was deliberately constructed with different ones.

TEST(CosineAnnealingSchedulerTest, StateDictRoundTripRestoresHyperparameters) {
    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.1F);
    ttml::schedulers::CosineAnnealingScheduler src(src_opt.get(), /*T_max=*/20, /*eta_min=*/1e-4F);
    for (int i = 0; i < 7; ++i) {
        src.step();
    }

    auto state = src.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "m_T_max"), 20U);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "m_eta_min"), 1e-4F);
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "m_last_step"), 7U);

    // Destination intentionally uses different hyperparameters; they must be
    // overwritten by ``set_state_dict``.
    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.1F);
    ttml::schedulers::CosineAnnealingScheduler dst(dst_opt.get(), /*T_max=*/100, /*eta_min=*/0.0F);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(dst_state, "m_T_max"), 20U);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_eta_min"), 1e-4F);
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(dst_state, "m_last_step"), 7U);

    // Continuing src and dst from this point must produce the same trajectory.
    for (int i = 0; i < 10; ++i) {
        src.step();
        dst.step();
        EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
    }
}

TEST(StepLRSchedulerTest, StateDictRoundTripRestoresHyperparameters) {
    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);
    ttml::schedulers::StepScheduler src(src_opt.get(), /*step_size=*/3, /*gamma=*/0.5F);
    for (int i = 0; i < 4; ++i) {
        src.step();
    }

    auto state = src.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "m_step_size"), 3U);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "m_gamma"), 0.5F);
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "m_last_step"), 4U);

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);
    ttml::schedulers::StepScheduler dst(dst_opt.get(), /*step_size=*/10, /*gamma=*/0.9F);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(dst_state, "m_step_size"), 3U);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_gamma"), 0.5F);
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(dst_state, "m_last_step"), 4U);

    for (int i = 0; i < 10; ++i) {
        src.step();
        dst.step();
        EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
    }
}

TEST(LinearSchedulerTest, StateDictRoundTripRestoresHyperparameters) {
    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);
    ttml::schedulers::LinearScheduler src(src_opt.get(), /*start_factor=*/1.0F, /*end_factor=*/0.0F, /*total_steps=*/4);
    for (int i = 0; i < 2; ++i) {
        src.step();
    }

    auto state = src.get_state_dict();
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "m_start_factor"), 1.0F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "m_end_factor"), 0.0F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(state, "m_total_steps"), 4);
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "m_last_step"), 2U);

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);
    ttml::schedulers::LinearScheduler dst(
        dst_opt.get(), /*start_factor=*/0.25F, /*end_factor=*/0.5F, /*total_steps=*/100);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_start_factor"), 1.0F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_end_factor"), 0.0F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(dst_state, "m_total_steps"), 4);
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(dst_state, "m_last_step"), 2U);

    for (int i = 0; i < 5; ++i) {
        src.step();
        dst.step();
        EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
    }
}

TEST(LambdaSchedulerTest, StateDictRoundTripRestoresStepAndLR) {
    // ``std::function`` cannot be introspected, so the lambda itself is not
    // part of the saved state. The destination must be reconstructed with the
    // same callable; only ``m_last_step`` and the base/last LRs are restored.
    auto lr_lambda = [](int epoch) { return 1.0F / static_cast<float>(epoch + 1); };

    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(1.0F);
    ttml::schedulers::LambdaScheduler src(src_opt.get(), lr_lambda);
    for (int i = 0; i < 3; ++i) {
        src.step();
    }

    auto state = src.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "m_last_step"), 3U);
    EXPECT_EQ(ttml::serialization::get_value_type<std::vector<float>>(state, "m_last_lrs"), src.get_last_lrs());

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(1.0F);
    ttml::schedulers::LambdaScheduler dst(dst_opt.get(), lr_lambda);
    dst.set_state_dict(state);

    for (int i = 0; i < 5; ++i) {
        src.step();
        dst.step();
        EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
    }
}

TEST(LambdaSchedulerTest, StateDictRestoreDoesNotCallLambda) {
    // Restoring writes back the saved LRs; calling a stateful callable again would advance it.
    int src_calls = 0;
    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(1.0F);
    ttml::schedulers::LambdaScheduler src(
        src_opt.get(), [&src_calls](int) { return 1.0F / static_cast<float>(++src_calls); });
    for (int i = 0; i < 3; ++i) {
        src.step();
    }
    auto state = src.get_state_dict();

    int dst_calls = 0;
    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(1.0F);
    ttml::schedulers::LambdaScheduler dst(
        dst_opt.get(), [&dst_calls](int) { return 1.0F / static_cast<float>(++dst_calls); });
    ASSERT_EQ(dst_calls, 1);  // lambda(0) at construction
    dst.set_state_dict(state);

    EXPECT_EQ(dst_calls, 1);
    EXPECT_EQ(dst.get_last_lrs(), src.get_last_lrs());
    EXPECT_FLOAT_EQ(dst_opt->lr(), src_opt->lr());
}

TEST(SequentialSchedulerTest, StateDictSavesAllChildSchedulers) {
    // Build a Linear-warmup -> Linear-decay chain and step partway into the
    // *second* child. The state dict must contain entries for BOTH children
    // (not just the currently-active one), and restoring into a destination
    // chain constructed with different hyperparameters must overwrite both
    // children's hyperparameters.
    constexpr float kStartLr = 0.3e-3F;
    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(kStartLr);

    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> src_children;
    src_children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(
        src_opt.get(), /*start_factor=*/0.1F, /*end_factor=*/1.0F, /*total_steps=*/10));
    src_children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(
        src_opt.get(), /*start_factor=*/1.0F, /*end_factor=*/0.1F, /*total_steps=*/50));
    ttml::schedulers::SequentialScheduler src(
        src_opt.get(), std::move(src_children), /*milestones=*/std::vector<size_t>{10U, 50U});

    // Run through all of child 0 (warmup) plus 5 steps of child 1 (decay) so
    // child 0 is "spent" but its state is still meaningfully saved.
    for (int i = 0; i < 15; ++i) {
        src.step();
    }
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(src.get_state_dict(), "m_current_scheduler_index"), 1U);

    auto state = src.get_state_dict();

    // Both children's hyperparameters are present under their per-index prefix.
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "scheduler_0/m_start_factor"), 0.1F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "scheduler_0/m_end_factor"), 1.0F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(state, "scheduler_0/m_total_steps"), 10);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "scheduler_1/m_start_factor"), 1.0F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "scheduler_1/m_end_factor"), 0.1F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(state, "scheduler_1/m_total_steps"), 50);

    // Both children also persist their per-child step counters.
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "scheduler_0/m_last_step"), 10U);
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "scheduler_1/m_last_step"), 5U);

    // Destination chain uses deliberately wrong hyperparameters in BOTH
    // children -- the round trip must overwrite them.
    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(kStartLr);
    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> dst_children;
    dst_children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(
        dst_opt.get(), /*start_factor=*/0.5F, /*end_factor=*/0.5F, /*total_steps=*/999));
    dst_children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(
        dst_opt.get(), /*start_factor=*/0.5F, /*end_factor=*/0.5F, /*total_steps=*/999));
    ttml::schedulers::SequentialScheduler dst(
        dst_opt.get(), std::move(dst_children), /*milestones=*/std::vector<size_t>{10U, 50U});

    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_0/m_start_factor"), 0.1F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_0/m_end_factor"), 1.0F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(dst_state, "scheduler_0/m_total_steps"), 10);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_1/m_start_factor"), 1.0F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_1/m_end_factor"), 0.1F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(dst_state, "scheduler_1/m_total_steps"), 50);

    // And the resumed chain must follow the same LR trajectory as the source.
    for (int i = 0; i < 30; ++i) {
        src.step();
        dst.step();
        EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs()) << "step " << i;
    }
}

TEST(SequentialSchedulerTest, MismatchedMilestonesRejected) {
    // The constructor must reject any milestones vector whose size differs
    // from the schedulers vector. Without this check, ``step()`` would read
    // out-of-bounds from ``m_milestones`` (undefined behavior).
    auto opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.1F);

    auto make_two_schedulers = [&] {
        std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> v;
        v.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(opt.get(), 0.2F, 1.0F, 5));
        v.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(opt.get(), 1.0F, 0.1F, 5));
        return v;
    };

    // Too few milestones.
    EXPECT_THROW(
        ttml::schedulers::SequentialScheduler(opt.get(), make_two_schedulers(), /*milestones=*/{5U}),
        std::invalid_argument);

    // Too many milestones.
    EXPECT_THROW(
        ttml::schedulers::SequentialScheduler(opt.get(), make_two_schedulers(), /*milestones=*/{5U, 5U, 5U}),
        std::invalid_argument);

    // Matched lengths must NOT throw.
    EXPECT_NO_THROW(ttml::schedulers::SequentialScheduler(opt.get(), make_two_schedulers(), /*milestones=*/{5U, 5U}));
}

// ----------------------------------
// Base LRs survive a resume onto an optimizer whose LR has decayed
// ----------------------------------
//
// A scheduler constructed on top of a resumed optimizer reads that
// optimizer's (already-decayed) LR as its base. ``m_base_lrs`` is therefore
// part of the scheduler state dict, so restoring it brings back the original
// base LRs and the next ``step()`` matches the source run.
//
// Each test below:
//   1. Steps the source scheduler so the optimizer's LR has demonstrably decayed.
//   2. Builds a fresh "destination" optimizer initialized with that decayed LR
//      (mimicking what ``load_optimizer`` would produce).
//   3. Constructs a fresh destination scheduler (its base LR is the decayed one).
//   4. Round-trips the state dict.
//   5. Asserts the destination's ``m_base_lrs`` match the source's, and that
//      the next step produces the same LR on both.

namespace {
constexpr float kBugRepoBaseLr = 0.1F;
}

TEST(CosineAnnealingSchedulerTest, BaseLrPersistsWhenResumedOptimizerHasDecayedLr) {
    constexpr size_t kTMax = 20;
    constexpr float kEtaMin = 1e-4F;

    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(kBugRepoBaseLr);
    ttml::schedulers::CosineAnnealingScheduler src(src_opt.get(), kTMax, kEtaMin);
    for (int i = 0; i < 5; ++i) {
        src.step();
    }
    const float decayed_lr = src_opt->lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr) << "sanity: optimizer LR must have decayed";
    auto state = src.get_state_dict();

    // Mimic checkpoint resume: optimizer's LR is the decayed value.
    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::CosineAnnealingScheduler dst(dst_opt.get(), kTMax, kEtaMin);
    dst.set_state_dict(state);

    // dst must remember its ORIGINAL base lr, not the optimizer's current value.
    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lrs")) << "m_base_lrs must be in the state dict";
    EXPECT_EQ(
        ttml::serialization::get_value_type<std::vector<float>>(dst_state, "m_base_lrs"),
        std::vector<float>{kBugRepoBaseLr});

    src.step();
    dst.step();
    EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
}

TEST(StepLRSchedulerTest, BaseLrPersistsWhenResumedOptimizerHasDecayedLr) {
    constexpr size_t kStepSize = 3;
    constexpr float kGamma = 0.5F;

    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(kBugRepoBaseLr);
    ttml::schedulers::StepScheduler src(src_opt.get(), kStepSize, kGamma);
    // Step past one boundary so the LR has actually been multiplied by gamma.
    for (size_t i = 0; i < kStepSize + 1; ++i) {
        src.step();
    }
    const float decayed_lr = src_opt->lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr);
    auto state = src.get_state_dict();

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::StepScheduler dst(dst_opt.get(), kStepSize, kGamma);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lrs"));
    EXPECT_EQ(
        ttml::serialization::get_value_type<std::vector<float>>(dst_state, "m_base_lrs"),
        std::vector<float>{kBugRepoBaseLr});

    src.step();
    dst.step();
    EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
}

TEST(LinearSchedulerTest, BaseLrPersistsWhenResumedOptimizerHasDecayedLr) {
    constexpr float kStartFactor = 1.0F;
    constexpr float kEndFactor = 0.0F;
    constexpr int kTotalSteps = 30;

    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(kBugRepoBaseLr);
    ttml::schedulers::LinearScheduler src(src_opt.get(), kStartFactor, kEndFactor, kTotalSteps);
    for (int i = 0; i < 5; ++i) {
        src.step();
    }
    const float decayed_lr = src_opt->lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr);
    auto state = src.get_state_dict();

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::LinearScheduler dst(dst_opt.get(), kStartFactor, kEndFactor, kTotalSteps);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lrs"));
    EXPECT_EQ(
        ttml::serialization::get_value_type<std::vector<float>>(dst_state, "m_base_lrs"),
        std::vector<float>{kBugRepoBaseLr});

    src.step();
    dst.step();
    EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
}

TEST(LambdaSchedulerTest, BaseLrPersistsWhenResumedOptimizerHasDecayedLr) {
    auto lr_lambda = [](int epoch) { return 1.0F / static_cast<float>(epoch + 1); };

    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(kBugRepoBaseLr);
    ttml::schedulers::LambdaScheduler src(src_opt.get(), lr_lambda);
    for (int i = 0; i < 3; ++i) {
        src.step();
    }
    const float decayed_lr = src_opt->lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr);
    auto state = src.get_state_dict();

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::LambdaScheduler dst(dst_opt.get(), lr_lambda);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lrs"));
    EXPECT_EQ(
        ttml::serialization::get_value_type<std::vector<float>>(dst_state, "m_base_lrs"),
        std::vector<float>{kBugRepoBaseLr});

    src.step();
    dst.step();
    EXPECT_EQ(src.get_last_lrs(), dst.get_last_lrs());
}

// ----------------------------------
// Live LR is restored together with scheduler state
//
// Scheduler constructors write the construction-time LR to the optimizer, so
// in the resume order "load optimizer state -> construct scheduler -> load
// scheduler state" the checkpoint's live LR gets overwritten at construction.
// set_state_dict must push m_last_lrs back to the optimizer or the first
// resumed optimizer step runs at the wrong LR.
// ----------------------------------
TEST(SchedulerLiveLrTest, LiveLrRestoredWhenOptimizerLoadedBeforeSchedulerConstruction) {
    // Source run: decay a few steps mid-warmup, checkpoint both.
    ttml::optimizers::MockOptimizer src_opt(1.0F);
    ttml::schedulers::LinearScheduler src(&src_opt, /*start_factor=*/0.5F, /*end_factor=*/1.0F, /*total_steps=*/10);
    for (int i = 0; i < 3; ++i) {
        src.step();
    }
    const float live_lr = src_opt.lr();
    auto opt_state = src_opt.get_state_dict();
    auto sched_state = src.get_state_dict();

    // Resume in the hazardous order: optimizer state FIRST, then scheduler
    // construction (which overwrites the live LR with base_lr * 0.5)...
    ttml::optimizers::MockOptimizer dst_opt(1.0F);
    dst_opt.set_state_dict(opt_state);
    EXPECT_FLOAT_EQ(dst_opt.lr(), live_lr);
    ttml::schedulers::LinearScheduler dst(&dst_opt, /*start_factor=*/0.5F, /*end_factor=*/1.0F, /*total_steps=*/10);
    EXPECT_FLOAT_EQ(dst_opt.lr(), 0.5F);

    // ...then scheduler state: the live LR must come back BEFORE any step()
    // call, so the first resumed optimizer step uses it.
    dst.set_state_dict(sched_state);
    EXPECT_FLOAT_EQ(dst_opt.lr(), live_lr);
}

TEST(SchedulerLiveLrTest, SequentialRestoresActiveChildsLiveLr) {
    // Chain: 10-step warmup then 20-step decay; checkpoint 3 steps into the
    // warmup (child 0 active).
    auto make = [](ttml::optimizers::MockOptimizer &opt) {
        std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> children;
        children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(&opt, 0.1F, 1.0F, 10));
        children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(&opt, 1.0F, 0.01F, 20));
        return std::make_unique<ttml::schedulers::SequentialScheduler>(
            &opt, std::move(children), std::vector<size_t>{10, 20});
    };

    ttml::optimizers::MockOptimizer src_opt(1.0F);
    auto src = make(src_opt);
    for (int i = 0; i < 3; ++i) {
        src->step();
    }
    const float live_lr = src_opt.lr();
    auto state = src->get_state_dict();

    // Restoring child states pushes each child's saved live LR to the
    // optimizer in turn (the LAST child's would win); the chain must re-apply
    // the ACTIVE child's live LR afterwards.
    ttml::optimizers::MockOptimizer dst_opt(1.0F);
    auto dst = make(dst_opt);
    dst->set_state_dict(state);
    EXPECT_FLOAT_EQ(dst_opt.lr(), live_lr);
}

// ----------------------------------
// Optimizer initial_lr serialization
// (mirrors PyTorch, where param_group["initial_lr"] rides along in
// optimizer.state_dict())
// ----------------------------------
TEST(OptimizerInitialLrTest, StateDictAlwaysContainsInitialLr) {
    // Before any scheduler attaches, the base LR is simply the current LR.
    ttml::optimizers::MockOptimizer optimizer(1.0F);
    auto state = optimizer.get_state_dict();
    const std::string initial_lr_key = "param_groups.0.initial_lr";
    ASSERT_TRUE(state.contains(initial_lr_key));
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, initial_lr_key), 1.0F);
}

TEST(OptimizerInitialLrTest, InitialLrSurvivesOptimizerStateDictRoundTrip) {
    // Source run: a scheduler attaches (recording initial_lr = 1.0) and the
    // LR later decays before the checkpoint is taken.
    ttml::optimizers::MockOptimizer src(1.0F);
    ttml::schedulers::LinearScheduler src_sched(&src, /*start_factor=*/0.5F, /*end_factor=*/1.0F, /*total_steps=*/10);
    src.set_group_lrs({0.25F});  // simulate decay before checkpointing
    auto state = src.get_state_dict();
    const std::string initial_lr_key = "param_groups.0.initial_lr";
    ASSERT_TRUE(state.contains(initial_lr_key));
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, initial_lr_key), 1.0F);

    // Resume into a fresh optimizer BEFORE any scheduler attaches (the usual
    // restore order). The decayed LR is restored...
    ttml::optimizers::MockOptimizer dst(0.999F);
    dst.set_state_dict(state);
    EXPECT_FLOAT_EQ(dst.lr(), 0.25F);

    // ...and a brand-new scheduler attached afterwards must use the ORIGINAL
    // base LR from the checkpoint, not the decayed one.
    EXPECT_EQ(dst.get_initial_group_lrs(), std::vector<float>{1.0F});
    ttml::schedulers::StepScheduler dst_sched(&dst, /*step_size=*/10, /*gamma=*/0.1F);
    EXPECT_FLOAT_EQ(dst.lr(), 1.0F);
}

// ----------------------------------
// Param groups: every group is scheduled from its own base LR
// ----------------------------------
namespace {
const std::vector<float> kGroupBaseLrs = {0.1F, 0.01F};

float cosine_lr(float base_lr, float eta_min, size_t step, size_t t_max) {
    return eta_min +
           0.5F * (base_lr - eta_min) *
               (1.F + std::cos(static_cast<float>(M_PI) * static_cast<float>(step) / static_cast<float>(t_max)));
}
}  // namespace

TEST(ParamGroupSchedulingTest, CosineDecaysEveryGroupToEtaMin) {
    constexpr size_t kTMax = 10;
    constexpr float kEtaMin = 1e-3F;
    ttml::optimizers::MockOptimizer optimizer(kGroupBaseLrs);
    ttml::schedulers::CosineAnnealingScheduler scheduler(&optimizer, kTMax, kEtaMin);
    EXPECT_EQ(optimizer.get_group_lrs(), kGroupBaseLrs);

    for (size_t s = 1; s <= kTMax; ++s) {
        scheduler.step();
        const auto lrs = optimizer.get_group_lrs();
        ASSERT_EQ(lrs.size(), kGroupBaseLrs.size());
        for (size_t g = 0; g < lrs.size(); ++g) {
            EXPECT_FLOAT_EQ(lrs[g], cosine_lr(kGroupBaseLrs[g], kEtaMin, s, kTMax)) << "group " << g << " step " << s;
        }
        EXPECT_EQ(scheduler.get_last_lrs(), lrs);
    }
    EXPECT_NEAR(optimizer.get_group_lrs()[0], kEtaMin, 1e-7F);
    EXPECT_NEAR(optimizer.get_group_lrs()[1], kEtaMin, 1e-7F);
}

TEST(ParamGroupSchedulingTest, MultiplicativeSchedulersScaleEveryGroupsBase) {
    auto expect_factor = [](const ttml::optimizers::MockOptimizer &optimizer, float factor, int step) {
        const auto lrs = optimizer.get_group_lrs();
        for (size_t g = 0; g < lrs.size(); ++g) {
            EXPECT_FLOAT_EQ(lrs[g], kGroupBaseLrs[g] * factor) << "group " << g << " step " << step;
        }
    };
    {
        ttml::optimizers::MockOptimizer optimizer(kGroupBaseLrs);
        ttml::schedulers::LinearScheduler scheduler(&optimizer, /*start_factor=*/0.5F, /*end_factor=*/1.0F, 4);
        expect_factor(optimizer, 0.5F, 0);
        for (int s = 1; s <= 5; ++s) {
            scheduler.step();
            expect_factor(optimizer, 0.5F + 0.5F * std::min(s / 4.F, 1.F), s);
        }
    }
    {
        ttml::optimizers::MockOptimizer optimizer(kGroupBaseLrs);
        ttml::schedulers::StepScheduler scheduler(&optimizer, /*step_size=*/2, /*gamma=*/0.5F);
        for (int s = 1; s <= 5; ++s) {
            scheduler.step();
            const int num_decays = s / 2;  // gamma is applied once every step_size=2 steps
            expect_factor(optimizer, std::pow(0.5F, static_cast<float>(num_decays)), s);
        }
    }
    {
        ttml::optimizers::MockOptimizer optimizer(kGroupBaseLrs);
        ttml::schedulers::LambdaScheduler scheduler(&optimizer, [](int epoch) { return 1.0F / (epoch + 1); });
        for (int s = 1; s <= 5; ++s) {
            scheduler.step();
            expect_factor(optimizer, 1.0F / (s + 1), s);
        }
    }
}

TEST(ParamGroupSchedulingTest, SequentialWarmupThenCosinePerGroup) {
    constexpr size_t kWarmup = 4;
    constexpr size_t kTMax = 6;
    constexpr float kEtaMin = 1e-3F;
    ttml::optimizers::MockOptimizer optimizer(kGroupBaseLrs);
    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> children;
    children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(&optimizer, 0.25F, 1.0F, kWarmup));
    children.push_back(std::make_unique<ttml::schedulers::CosineAnnealingScheduler>(&optimizer, kTMax, kEtaMin));
    ttml::schedulers::SequentialScheduler scheduler(&optimizer, std::move(children), {kWarmup, kTMax});

    for (size_t g = 0; g < kGroupBaseLrs.size(); ++g) {
        EXPECT_FLOAT_EQ(optimizer.get_group_lrs()[g], kGroupBaseLrs[g] * 0.25F) << "group " << g;
    }
    for (size_t s = 1; s <= kWarmup; ++s) {
        scheduler.step();
        const float factor = 0.25F + 0.75F * static_cast<float>(s) / kWarmup;
        for (size_t g = 0; g < kGroupBaseLrs.size(); ++g) {
            EXPECT_FLOAT_EQ(optimizer.get_group_lrs()[g], kGroupBaseLrs[g] * factor) << "group " << g << " step " << s;
        }
    }
    for (size_t s = 1; s <= kTMax; ++s) {
        scheduler.step();
        for (size_t g = 0; g < kGroupBaseLrs.size(); ++g) {
            EXPECT_FLOAT_EQ(optimizer.get_group_lrs()[g], cosine_lr(kGroupBaseLrs[g], kEtaMin, s, kTMax))
                << "group " << g << " cosine step " << s;
        }
    }
}

TEST(ParamGroupSchedulingTest, StateDictRestoresEveryGroupsBaseAndLiveLr) {
    constexpr size_t kTMax = 10;
    constexpr float kEtaMin = 1e-3F;
    ttml::optimizers::MockOptimizer src_opt(kGroupBaseLrs);
    ttml::schedulers::CosineAnnealingScheduler src(&src_opt, kTMax, kEtaMin);
    for (int i = 0; i < 4; ++i) {
        src.step();
    }
    const auto state = src.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<std::vector<float>>(state, "m_base_lrs"), kGroupBaseLrs);
    EXPECT_EQ(ttml::serialization::get_value_type<std::vector<float>>(state, "m_last_lrs"), src_opt.get_group_lrs());

    // Resume onto an optimizer whose LRs have already decayed: the scheduler must keep the saved bases and
    // push the saved live LRs back to the optimizer.
    ttml::optimizers::MockOptimizer dst_opt(src_opt.get_group_lrs());
    ttml::schedulers::CosineAnnealingScheduler dst(&dst_opt, kTMax, kEtaMin);
    dst.set_state_dict(state);
    EXPECT_EQ(dst_opt.get_group_lrs(), src_opt.get_group_lrs());

    for (int i = 0; i < 4; ++i) {
        src.step();
        dst.step();
        EXPECT_EQ(dst_opt.get_group_lrs(), src_opt.get_group_lrs()) << "step " << i;
    }
}

TEST(ParamGroupSchedulingTest, RejectedStateDictLeavesSchedulerUnchanged) {
    // A checkpoint from an optimizer with 2 groups, loaded into a scheduler on a 1-group optimizer.
    ttml::optimizers::MockOptimizer two_groups(kGroupBaseLrs);
    ttml::schedulers::StepScheduler src(&two_groups, /*step_size=*/1, /*gamma=*/0.5F);
    src.step();
    const auto two_group_state = src.get_state_dict();

    ttml::optimizers::MockOptimizer one_group(0.3F);
    ttml::schedulers::StepScheduler dst(&one_group, /*step_size=*/1, /*gamma=*/0.5F);
    dst.step();
    EXPECT_THROW(dst.set_state_dict(two_group_state), std::runtime_error);
    const auto after = dst.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(after, "m_last_step"), 1U);
    EXPECT_EQ(ttml::serialization::get_value_type<std::vector<float>>(after, "m_base_lrs"), std::vector<float>{0.3F});
    EXPECT_FLOAT_EQ(one_group.lr(), 0.15F);
    dst.step();  // still usable
    EXPECT_FLOAT_EQ(one_group.lr(), 0.075F);
}

TEST(ParamGroupSchedulingTest, StateDictMissingBaseKeyLeavesSchedulerUnchanged) {
    // The base state is read before a derived scheduler assigns its own state, so a rejected dict changes nothing.
    ttml::optimizers::MockOptimizer src_opt(0.3F);
    ttml::schedulers::StepScheduler src(&src_opt, /*step_size=*/2, /*gamma=*/0.1F);
    src.step();
    auto state = src.get_state_dict();
    state.erase("m_last_step");

    ttml::optimizers::MockOptimizer dst_opt(0.3F);
    ttml::schedulers::StepScheduler dst(&dst_opt, /*step_size=*/1, /*gamma=*/0.5F);
    EXPECT_ANY_THROW(dst.set_state_dict(state));
    const auto after = dst.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(after, "m_step_size"), 1U);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(after, "m_gamma"), 0.5F);
}

TEST(SequentialSchedulerTest, StateDictMissingBaseKeyLeavesChildrenUnchanged) {
    auto make_chain = [](ttml::optimizers::OptimizerBase *opt, float gamma) {
        std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> children;
        children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(opt, 0.5F, 1.0F, 2));
        children.push_back(std::make_unique<ttml::schedulers::StepScheduler>(opt, 1, gamma));
        return ttml::schedulers::SequentialScheduler(opt, std::move(children), std::vector<size_t>{2U, 4U});
    };
    ttml::optimizers::MockOptimizer src_opt(0.3F);
    auto src = make_chain(&src_opt, /*gamma=*/0.1F);
    for (int i = 0; i < 3; ++i) {
        src.step();
    }
    auto state = src.get_state_dict();
    state.erase("m_last_step");

    ttml::optimizers::MockOptimizer dst_opt(0.3F);
    auto dst = make_chain(&dst_opt, /*gamma=*/0.5F);
    const auto before = dst.get_state_dict();
    EXPECT_ANY_THROW(dst.set_state_dict(state));
    const auto after = dst.get_state_dict();
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(after, "scheduler_1/m_gamma"), 0.5F);
    EXPECT_EQ(
        ttml::serialization::get_value_type<size_t>(after, "scheduler_1/m_last_step"),
        ttml::serialization::get_value_type<size_t>(before, "scheduler_1/m_last_step"));
    EXPECT_FLOAT_EQ(dst_opt.lr(), 0.3F * 0.5F);
}

TEST(ParamGroupSchedulingTest, StateDictWithWrongLastLrsCountLeavesSchedulerUnchanged) {
    ttml::optimizers::MockOptimizer src_opt(0.3F);
    ttml::schedulers::StepScheduler src(&src_opt, /*step_size=*/2, /*gamma=*/0.1F);
    auto state = src.get_state_dict();
    state["m_last_lrs"] = std::vector<float>{0.3F, 0.3F};

    ttml::optimizers::MockOptimizer dst_opt(0.3F);
    ttml::schedulers::StepScheduler dst(&dst_opt, /*step_size=*/1, /*gamma=*/0.5F);
    EXPECT_THROW(dst.set_state_dict(state), std::runtime_error);
    const auto after = dst.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(after, "m_step_size"), 1U);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(after, "m_gamma"), 0.5F);
}

TEST(SequentialSchedulerTest, RejectedChildStateRollsBack) {
    // A later child rejecting its state undoes the earlier children's restore and the optimizer's LRs.
    auto make = [](ttml::optimizers::MockOptimizer &opt) {
        std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> children;
        children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(&opt, 0.5F, 1.0F, 4));
        children.push_back(std::make_unique<ttml::schedulers::StepScheduler>(&opt, 1, 0.5F));
        return std::make_unique<ttml::schedulers::SequentialScheduler>(
            &opt, std::move(children), std::vector<size_t>{4, 4});
    };
    ttml::optimizers::MockOptimizer other_opt(1.0F);
    auto other = make(other_opt);
    for (int i = 0; i < 6; ++i) {
        other->step();
    }
    auto state = other->get_state_dict();
    state.erase("scheduler_1/m_gamma");  // the second child cannot restore

    ttml::optimizers::MockOptimizer opt(1.0F);
    auto sched = make(opt);
    sched->step();
    sched->step();
    const float lr_before = opt.lr();
    const auto before = sched->get_state_dict();
    EXPECT_ANY_THROW(sched->set_state_dict(state));
    const auto after = sched->get_state_dict();
    EXPECT_FLOAT_EQ(opt.lr(), lr_before);
    for (const auto *key : {"m_current_scheduler_index", "m_last_step", "scheduler_0/m_last_step"}) {
        EXPECT_EQ(
            ttml::serialization::get_value_type<size_t>(after, key),
            ttml::serialization::get_value_type<size_t>(before, key))
            << key;
    }
    sched->step();  // continues from where it was: step 3 of the warmup
    EXPECT_FLOAT_EQ(opt.lr(), 0.5F + 0.5F * 3.0F / 4.0F);
}

TEST(ParamGroupSchedulingTest, SetGroupLrsRejectsWrongCount) {
    ttml::optimizers::MockOptimizer grouped(kGroupBaseLrs);
    EXPECT_THROW(grouped.set_group_lrs({0.1F}), std::invalid_argument);
    ttml::optimizers::MockOptimizer single(0.1F);
    EXPECT_THROW(single.set_group_lrs({0.1F, 0.2F}), std::invalid_argument);
    single.set_group_lrs({0.3F});
    EXPECT_FLOAT_EQ(single.lr(), 0.3F);
}

TEST(SequentialSchedulerTest, StateDictWithOutOfRangeIndexRejected) {
    // An index past the last child or a step count past the child's milestone must be rejected, not dereferenced.
    ttml::optimizers::MockOptimizer optimizer(0.1F);
    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> children;
    children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(&optimizer, 0.5F, 1.0F, 4));
    children.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(&optimizer, 1.0F, 0.1F, 4));
    ttml::schedulers::SequentialScheduler scheduler(&optimizer, std::move(children), {4U, 4U});

    auto state = scheduler.get_state_dict();
    state["m_current_scheduler_index"] = size_t{2};
    EXPECT_THROW(scheduler.set_state_dict(state), std::invalid_argument);

    state = scheduler.get_state_dict();
    state["m_current_step_in_scheduler"] = 5;
    EXPECT_THROW(scheduler.set_state_dict(state), std::invalid_argument);
}
