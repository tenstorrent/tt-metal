// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "core/not_null.hpp"
#include "optimizers/optimizer_base.hpp"
#include "schedulers/lambda_scheduler.hpp"
#include "schedulers/linear_scheduler.hpp"
#include "schedulers/sequential_scheduler.hpp"
#include "schedulers/step_scheduler.hpp"

namespace ttml::optimizers {
class MockOptimizer : public OptimizerBase {
public:
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
