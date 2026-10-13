// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/training_utils.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

using ttml::utils::epochs_completed;
using ttml::utils::resolve_effective_max_steps;
using ttml::utils::steps_per_epoch;

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST(TrainingUtilsTest, StepsPerEpochCountsTokens) {
    EXPECT_DOUBLE_EQ(steps_per_epoch(1000U, 4U, 10U), 25.0);
    EXPECT_DOUBLE_EQ(steps_per_epoch(1000U, 8U, 10U), 12.5);
    EXPECT_DOUBLE_EQ(steps_per_epoch(0U, 8U, 10U), 0.0);
}

TEST(TrainingUtilsTest, StepsPerEpochRejectsEmptyStep) {
    EXPECT_THAT(
        [] { (void)steps_per_epoch(1000U, 0U, 10U); },
        ThrowsMessage<std::runtime_error>(HasSubstr("global_batch_size and sequence_length must be positive")));
    EXPECT_THAT(
        [] { (void)steps_per_epoch(1000U, 4U, 0U); },
        ThrowsMessage<std::runtime_error>(HasSubstr("global_batch_size and sequence_length must be positive")));
}

TEST(TrainingUtilsTest, SlidingWindowsDoNotCountAsSamples) {
    constexpr size_t corpus_tokens = 1024U;
    constexpr uint32_t batch_size = 8U;
    constexpr uint32_t sequence_length = 128U;
    EXPECT_EQ(resolve_effective_max_steps(5000U, 1U, steps_per_epoch(corpus_tokens, batch_size, sequence_length)), 1U);
}

TEST(TrainingUtilsTest, GradientAccumulationIsPartOfGlobalBatch) {
    constexpr uint32_t batch_size = 4U;
    constexpr uint32_t accumulation_steps = 4U;
    EXPECT_EQ(resolve_effective_max_steps(0U, 1U, steps_per_epoch(4096U, batch_size * accumulation_steps, 32U)), 8U);
}

TEST(TrainingUtilsTest, ZeroEpochsIsUncapped) {
    EXPECT_EQ(resolve_effective_max_steps(5000U, 0U, 3.2), 5000U);
}

TEST(TrainingUtilsTest, ZeroMaxStepsIsUncapped) {
    EXPECT_EQ(resolve_effective_max_steps(0U, 2U, 12.5), 25U);
}

TEST(TrainingUtilsTest, EpochCapRoundsPartialStepUp) {
    EXPECT_EQ(resolve_effective_max_steps(5000U, 1U, 12.5), 13U);
    EXPECT_EQ(resolve_effective_max_steps(5000U, 3U, 10.9), 33U);
}

TEST(TrainingUtilsTest, EarlierCapWins) {
    EXPECT_EQ(resolve_effective_max_steps(10U, 2U, 12.5), 10U);
    EXPECT_EQ(resolve_effective_max_steps(100U, 2U, 12.5), 25U);
    EXPECT_EQ(resolve_effective_max_steps(25U, 2U, 12.5), 25U);
}

TEST(TrainingUtilsTest, EpochCapIsAtLeastOneStep) {
    EXPECT_EQ(resolve_effective_max_steps(0U, 1U, 0.01), 1U);
    EXPECT_EQ(resolve_effective_max_steps(0U, 1U, 0.0), 1U);
}

TEST(TrainingUtilsTest, NoCapThrows) {
    EXPECT_THAT(
        [] { (void)resolve_effective_max_steps(0U, 0U, 12.5); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("No stop condition: set max_steps > 0 or num_epochs > 0 in training_config.")));
}

TEST(TrainingUtilsTest, EpochsCompletedTruncates) {
    EXPECT_EQ(epochs_completed(0U, 17.02), 0U);
    EXPECT_EQ(epochs_completed(17U, 17.02), 0U);
    EXPECT_EQ(epochs_completed(18U, 17.02), 1U);
    EXPECT_EQ(epochs_completed(35U, 17.02), 2U);
    EXPECT_EQ(epochs_completed(3U, 0.5), 6U);
    EXPECT_EQ(epochs_completed(20U, 0.0), 0U);
}

TEST(TrainingUtilsTest, EpochCapBeyondStepCounterThrows) {
    const double steps = static_cast<double>(std::numeric_limits<uint32_t>::max());
    EXPECT_THAT(
        [&] { (void)resolve_effective_max_steps(0U, 2U, steps); },
        ThrowsMessage<std::runtime_error>(HasSubstr("exceeds the uint32 step counter")));
    EXPECT_EQ(resolve_effective_max_steps(7U, 2U, steps), 7U);
}
