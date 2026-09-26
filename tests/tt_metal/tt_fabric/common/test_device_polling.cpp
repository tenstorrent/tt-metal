// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

#include "common/device_polling.hpp"

namespace tt::tt_metal {
namespace {

TEST(DevicePollingTest, ProgressesControllableExecutionBetweenObservations) {
    uint32_t device_state = 0;
    uint32_t observations = 0;
    uint32_t progress_calls = 0;

    const bool reached_state = poll_until(
        [&]() {
            observations++;
            return device_state == 3;
        },
        [&]() {
            progress_calls++;
            device_state++;
        },
        std::chrono::milliseconds(100),
        std::chrono::milliseconds(0));

    EXPECT_TRUE(reached_state);
    EXPECT_EQ(progress_calls, 3);
    EXPECT_EQ(observations, 4);
}

TEST(DevicePollingTest, DoesNotProgressAfterSuccessfulObservation) {
    uint32_t progress_calls = 0;
    EXPECT_TRUE(poll_until(
        []() { return true; },
        [&]() { progress_calls++; },
        std::chrono::milliseconds(100),
        std::chrono::milliseconds(0)));
    EXPECT_EQ(progress_calls, 0);
}

}  // namespace
}  // namespace tt::tt_metal
