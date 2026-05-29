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
class MockOptimizer : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override {
        return "MockOptimizer";
    }
    explicit MockOptimizer(float lr) : OptimizerBase(ttml::serialization::NamedParameters{}), m_lr(lr) {
    }

    void zero_grad() override {};

    void step() override {};

    [[nodiscard]] serialization::StateDict get_state_dict() const override {
        return {};
    }

    void set_state_dict(const serialization::StateDict &dict) override {};

    [[nodiscard]] size_t get_steps() const override {
        return {};
    };
    void set_steps(size_t steps) override {};

    void set_lr(float lr) override {
        m_lr = lr;
    }

    [[nodiscard]] float get_lr() const override {
        return m_lr;
    }

private:
    float m_lr = 0;
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

    // Initial LR
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.1F);

    scheduler.step();  // epoch 0
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.1F * 0.5F);

    scheduler.step();  // epoch 1
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.1F * 0.5F);

    scheduler.step();  // epoch 2
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.1F * 0.5F);
}

TEST(LambdaSchedulerTest, VaryingFactor) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(1.0f);

    // Lambda: lr_factor = 1.0 / (epoch+1)
    ttml::schedulers::LambdaScheduler scheduler(optimizer.get(), [](int epoch) { return 1.0F / (epoch + 1); });

    // Before stepping
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 1.0F);

    scheduler.step();  // epoch 0: factor = 1/1=0.5F
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.5F);

    scheduler.step();  // epoch 1: factor = 1/2=0.5 lr=1.0*0.5=0.5
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 1.F / 3.F);

    scheduler.step();  // epoch 2: factor = 1/3≈0.3333 lr=1.0*0.3333=0.3333
    EXPECT_NEAR(optimizer->get_lr(), 1.F / 4.F, 1e-5);

    scheduler.step();  // epoch 3: factor = 1/5=0.2
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.2F);
}

// ----------------------------------
// Tests for StepLRScheduler
// ----------------------------------
TEST(StepLRSchedulerTest, BasicDecay) {
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(0.2F);

    // Decrease LR by factor of 0.1 every 3 steps
    ttml::schedulers::StepScheduler scheduler(optimizer.get(), 3, 0.1F);

    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.2F);
        scheduler.step();
    }

    for (int i = 0; i < 3; ++i) {
        // After 3 steps: lr = base_lr * 0.1
        EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.2F * 0.1F);
        scheduler.step();
    }
    // After 6 steps: lr = base_lr * 0.1^2
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.2F * 0.1F * 0.1F);
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
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.15F);

    // step 2: progress=0.5 lr=0.2+(0.0-0.2)*0.5=0.2-0.1=0.1
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.1F);

    // step 3: progress=0.75 lr=0.2+(0.0-0.2)*0.75=0.2-0.15=0.05
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.05F);

    // step 4: progress=1.0 lr=0.2+(0.0-0.2)*1.0=0.0
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.0f);

    // Extra steps keep it at 0.0
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.0f);
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
    EXPECT_FLOAT_EQ(optimizer->get_lr(), kBaseLr);

    // Walk through 2 * T_max + a few extra steps and compare against the
    // closed form at every step.
    for (size_t s = 1; s <= 2 * kTMax + 5; ++s) {
        scheduler.step();
        const float expected =
            kEtaMin +
            0.5F * (kBaseLr - kEtaMin) *
                (1.F + std::cos(static_cast<float>(M_PI) * static_cast<float>(s) / static_cast<float>(kTMax)));

        EXPECT_FLOAT_EQ(optimizer->get_lr(), expected) << "step " << s;
        EXPECT_FLOAT_EQ(scheduler.get_last_lr(), optimizer->get_lr()) << "step " << s;
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
    EXPECT_NEAR(optimizer->get_lr(), kEtaMin, 1e-7F);
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
    EXPECT_NEAR(optimizer->get_lr(), kBaseLr, 1e-7F);
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
    EXPECT_NEAR(optimizer->get_lr(), 0.0F, 1e-7F);
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
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.5F);

    seq_scheduler.step();  // 2nd step: LR=0.5*0.5=0.25
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.25F);

    seq_scheduler.step();  // 3rd step: LR=0.25*0.5=0.125
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.125F);

    // total_steps=2, start_lr=0.125, end_lr=0.1
    // step 1: progress=1/2=0.5 lr=1.0+(0.1-1.0)*0.5=0.55
    seq_scheduler.step();
    EXPECT_NEAR(optimizer->get_lr(), 0.55, 1e-5);

    // step 2: progress=2/2=1.0 lr=1.0+(0.1-1.0)*1.0=0.1 (min lr in linear scheduler)
    seq_scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.1F);

    // Further steps do nothing (we finished all schedulers)
    seq_scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->get_lr(), 0.1F);
}

TEST(SequentialSchedulerTest, WarmupSetup) {
    auto start_lr = 3.e-4F;
    auto optimizer = std::make_unique<ttml::optimizers::MockOptimizer>(start_lr);

    // First: LinearScheduler for 10 steps from 0 to start_lr
    auto warmup_scheduler = std::make_unique<ttml::schedulers::LinearScheduler>(optimizer.get(), 0.0F, 1.0F, 10);

    // Then: LinearScheduler for 50 steps from start_lr to 0.1F * start_lr
    auto linear_scheduler = std::make_unique<ttml::schedulers::LinearScheduler>(optimizer.get(), 1.F, 0.1F, 50);

    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> schedulers;
    std::vector<size_t> milestones;
    schedulers.push_back(std::move(warmup_scheduler));
    schedulers.push_back(std::move(linear_scheduler));
    milestones.push_back(10);
    milestones.push_back(50);
    ttml::schedulers::SequentialScheduler seq_scheduler(optimizer.get(), std::move(schedulers), std::move(milestones));

    for (int i = 0; i < 10; i++) {
        // Linear warmup: 10 steps from 0 to start_lr
        seq_scheduler.step();
        EXPECT_NEAR(optimizer->get_lr(), start_lr * (i + 1) / 10, 1e-5);
    }
    for (int i = 0; i < 50; i++) {
        // Linear decay: 50 steps from start_lr to 0.1F * start_lr
        seq_scheduler.step();
        EXPECT_NEAR(optimizer->get_lr(), start_lr * (1.0F - 0.9F * (i + 1) / 50.F), 1e-5);
    }
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
        EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
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
        EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
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
        EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
    }
}

TEST(LambdaSchedulerTest, StateDictRoundTripRestoresStepAndLR) {
    // ``std::function`` cannot be introspected, so the lambda itself is not
    // part of the saved state. The destination must be reconstructed with the
    // same callable; only ``m_last_step`` and ``m_last_lr`` are restored.
    auto lr_lambda = [](int epoch) { return 1.0F / static_cast<float>(epoch + 1); };

    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(1.0F);
    ttml::schedulers::LambdaScheduler src(src_opt.get(), lr_lambda);
    for (int i = 0; i < 3; ++i) {
        src.step();
    }

    auto state = src.get_state_dict();
    EXPECT_EQ(ttml::serialization::get_value_type<size_t>(state, "m_last_step"), 3U);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "m_last_lr"), src.get_last_lr());

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(1.0F);
    ttml::schedulers::LambdaScheduler dst(dst_opt.get(), lr_lambda);
    dst.set_state_dict(state);

    for (int i = 0; i < 5; ++i) {
        src.step();
        dst.step();
        EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
    }
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
        src_opt.get(), /*start_factor=*/0.0F, /*end_factor=*/1.0F, /*total_steps=*/10));
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
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(state, "scheduler_0/m_start_factor"), 0.0F);
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
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_0/m_start_factor"), 0.0F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_0/m_end_factor"), 1.0F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(dst_state, "scheduler_0/m_total_steps"), 10);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_1/m_start_factor"), 1.0F);
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "scheduler_1/m_end_factor"), 0.1F);
    EXPECT_EQ(ttml::serialization::get_value_type<int>(dst_state, "scheduler_1/m_total_steps"), 50);

    // And the resumed chain must follow the same LR trajectory as the source.
    for (int i = 0; i < 30; ++i) {
        src.step();
        dst.step();
        EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr()) << "step " << i;
    }
}

TEST(SequentialSchedulerTest, MismatchedMilestonesRejected) {
    // The constructor must reject any milestones vector whose size differs
    // from the schedulers vector. Without this check, ``step()`` would read
    // out-of-bounds from ``m_milestones`` (undefined behavior).
    auto opt = std::make_unique<ttml::optimizers::MockOptimizer>(0.1F);

    auto make_two_schedulers = [&] {
        std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> v;
        v.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(opt.get(), 0.0F, 1.0F, 5));
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
// Repro tests for the ``m_base_lr`` not-saved bug
// ----------------------------------
//
// Each stateful scheduler captures ``m_base_lr`` from ``optimizer->get_lr()``
// in its constructor but does NOT persist it in ``get_state_dict``. When a
// real training run resumes from a checkpoint, the optimizer is loaded with
// its CURRENT (already-decayed) LR; constructing a new scheduler on top of
// that optimizer captures the decayed value as the new ``m_base_lr``, and
// every subsequent ``step()`` is silently mis-scaled.
//
// Each test below:
//   1. Steps the source scheduler so the optimizer's LR has demonstrably decayed.
//   2. Builds a fresh "destination" optimizer initialized with that decayed LR
//      (mimicking what ``load_optimizer`` would produce).
//   3. Constructs a fresh destination scheduler -- its ``m_base_lr`` is now WRONG.
//   4. Round-trips the state dict.
//   5. Asserts the destination's ``m_base_lr`` matches the source's, and that
//      the next step produces the same LR on both.
//
// These tests are EXPECTED TO FAIL until ``m_base_lr`` is added to each
// scheduler's state dict.

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
    const float decayed_lr = src_opt->get_lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr) << "sanity: optimizer LR must have decayed";
    auto state = src.get_state_dict();

    // Mimic checkpoint resume: optimizer's LR is the decayed value.
    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::CosineAnnealingScheduler dst(dst_opt.get(), kTMax, kEtaMin);
    dst.set_state_dict(state);

    // dst must remember its ORIGINAL base lr, not the optimizer's current value.
    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lr")) << "m_base_lr must be in the state dict";
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_base_lr"), kBugRepoBaseLr);

    src.step();
    dst.step();
    EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
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
    const float decayed_lr = src_opt->get_lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr);
    auto state = src.get_state_dict();

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::StepScheduler dst(dst_opt.get(), kStepSize, kGamma);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lr"));
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_base_lr"), kBugRepoBaseLr);

    src.step();
    dst.step();
    EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
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
    const float decayed_lr = src_opt->get_lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr);
    auto state = src.get_state_dict();

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::LinearScheduler dst(dst_opt.get(), kStartFactor, kEndFactor, kTotalSteps);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lr"));
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_base_lr"), kBugRepoBaseLr);

    src.step();
    dst.step();
    EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
}

TEST(LambdaSchedulerTest, BaseLrPersistsWhenResumedOptimizerHasDecayedLr) {
    auto lr_lambda = [](int epoch) { return 1.0F / static_cast<float>(epoch + 1); };

    auto src_opt = std::make_unique<ttml::optimizers::MockOptimizer>(kBugRepoBaseLr);
    ttml::schedulers::LambdaScheduler src(src_opt.get(), lr_lambda);
    for (int i = 0; i < 3; ++i) {
        src.step();
    }
    const float decayed_lr = src_opt->get_lr();
    ASSERT_NE(decayed_lr, kBugRepoBaseLr);
    auto state = src.get_state_dict();

    auto dst_opt = std::make_unique<ttml::optimizers::MockOptimizer>(decayed_lr);
    ttml::schedulers::LambdaScheduler dst(dst_opt.get(), lr_lambda);
    dst.set_state_dict(state);

    auto dst_state = dst.get_state_dict();
    ASSERT_TRUE(dst_state.count("m_base_lr"));
    EXPECT_FLOAT_EQ(ttml::serialization::get_value_type<float>(dst_state, "m_base_lr"), kBugRepoBaseLr);

    src.step();
    dst.step();
    EXPECT_FLOAT_EQ(src.get_last_lr(), dst.get_last_lr());
}
