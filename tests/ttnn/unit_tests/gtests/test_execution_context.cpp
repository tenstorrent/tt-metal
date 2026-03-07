// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <exception>
#include <optional>

#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/execution_context.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

class ExecutionContextFixture : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        if (num_devices_ < 1) {
            GTEST_SKIP() << "No device available; skipping execution context tests.";
        }
        try {
            TTNNFixtureWithDevice::SetUp();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Device open failed (" << e.what() << "); skipping execution context tests.";
        }
    }
};

TEST_F(ExecutionContextFixture, GetCurrentSubDeviceIdDefaultsToFirst) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    const auto default_id = ttnn::execution_context::get_current_sub_device_id(device);
    EXPECT_EQ(default_id, device->get_sub_device_ids().at(0));
}

TEST_F(ExecutionContextFixture, SetCurrentSubDeviceUpdatesGetCurrent) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    const auto default_id = ttnn::execution_context::get_current_sub_device_id(device);

    {
        auto guard = ttnn::execution_context::set_current_sub_device(device, tt::tt_metal::SubDeviceId{1});
        EXPECT_EQ(ttnn::execution_context::get_current_sub_device_id(device), tt::tt_metal::SubDeviceId{1});
    }
    EXPECT_EQ(ttnn::execution_context::get_current_sub_device_id(device), default_id);
}

TEST_F(ExecutionContextFixture, GetEffectiveSubDeviceIdUsesExplicitWhenPresent) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    const auto explicit_id = tt::tt_metal::SubDeviceId{2};
    EXPECT_EQ(ttnn::execution_context::get_effective_sub_device_id(device, std::optional{explicit_id}), explicit_id);
}

TEST_F(ExecutionContextFixture, GetEffectiveSubDeviceIdUsesCurrentWhenNullopt) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    const auto default_id = ttnn::execution_context::get_current_sub_device_id(device);
    EXPECT_EQ(ttnn::execution_context::get_effective_sub_device_id(device, std::nullopt), default_id);

    auto guard = ttnn::execution_context::set_current_sub_device(device, tt::tt_metal::SubDeviceId{1});
    EXPECT_EQ(ttnn::execution_context::get_effective_sub_device_id(device, std::nullopt), tt::tt_metal::SubDeviceId{1});
}

}  // namespace ttnn::test
