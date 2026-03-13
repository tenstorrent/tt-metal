// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <exception>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/device_context.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

// Creates a sub-device manager with 3 sub-devices (IDs 0, 1, 2), each with 2 cores in one row,
// loads it, and sets the stall group so get_sub_device_ids() returns {0, 1, 2}.
// Returns the SubDeviceManagerId for teardown (remove_sub_device_manager).
static tt::tt_metal::SubDeviceManagerId setup_three_sub_devices(tt::tt_metal::distributed::MeshDevice* device) {
    constexpr int cores_per_subdevice = 2;
    constexpr int num_sub_devices = 3;
    std::vector<tt::tt_metal::CoreRangeSet> core_range_sets;
    core_range_sets.reserve(num_sub_devices);
    for (int row = 0; row < num_sub_devices; ++row) {
        tt::tt_metal::CoreRange range(
            tt::tt_metal::CoreCoord(0, row), tt::tt_metal::CoreCoord(cores_per_subdevice - 1, row));
        core_range_sets.push_back(tt::tt_metal::CoreRangeSet(range));
    }
    std::vector<tt::tt_metal::SubDevice> sub_devices;
    sub_devices.reserve(num_sub_devices);
    for (int i = 0; i < num_sub_devices; ++i) {
        sub_devices.push_back(
            tt::tt_metal::SubDevice(ttsl::Span<const tt::tt_metal::CoreRangeSet>(&core_range_sets[i], 1)));
    }
    const auto id = device->create_sub_device_manager(
        ttsl::Span<const tt::tt_metal::SubDevice>(sub_devices.data(), sub_devices.size()), tt::tt_metal::DeviceAddr{0});
    device->load_sub_device_manager(id);
    const std::array<tt::tt_metal::SubDeviceId, num_sub_devices> ids = {
        tt::tt_metal::SubDeviceId{0}, tt::tt_metal::SubDeviceId{1}, tt::tt_metal::SubDeviceId{2}};
    device->set_sub_device_stall_group(ttsl::Span<const tt::tt_metal::SubDeviceId>(ids.data(), ids.size()));
    return id;
}

class ExecutionContextFixture : public TTNNFixtureWithDevice {
protected:
    tt::tt_metal::SubDeviceManagerId sub_device_manager_id_;

    void SetUp() override {
        if (num_devices_ < 1) {
            GTEST_SKIP() << "No device available; skipping execution context tests.";
        }
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Sub-device managers require fast dispatch; skipping execution context tests "
                            "(TT_METAL_SLOW_DISPATCH_MODE=1).";
        }
        try {
            TTNNFixtureWithDevice::SetUp();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Device open failed (" << e.what() << "); skipping execution context tests.";
        }
        sub_device_manager_id_ = setup_three_sub_devices(device_);
    }

    void TearDown() override {
        if (device_ != nullptr) {
            device_->reset_sub_device_stall_group();
            device_->clear_loaded_sub_device_manager();
            device_->remove_sub_device_manager(sub_device_manager_id_);
        }
        TTNNFixtureWithDevice::TearDown();
    }
};

TEST_F(ExecutionContextFixture, GetCurrentSubDeviceIdDefaultsToFirst) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    ttnn::DeviceContext ctx(device);
    const auto default_id = ctx.get_current_sub_device_id();
    EXPECT_EQ(default_id, device->get_sub_device_ids().at(0));
}

TEST_F(ExecutionContextFixture, SetCurrentSubDeviceUpdatesGetCurrent) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    ttnn::DeviceContext ctx(device);
    const auto default_id = ctx.get_current_sub_device_id();

    {
        auto guard = ctx.set_current_sub_device(tt::tt_metal::SubDeviceId{1});
        EXPECT_EQ(ttnn::DeviceContext(device).get_current_sub_device_id(), tt::tt_metal::SubDeviceId{1});
    }
    EXPECT_EQ(ttnn::DeviceContext(device).get_current_sub_device_id(), default_id);
}

TEST_F(ExecutionContextFixture, GetEffectiveSubDeviceIdUsesExplicitWhenPresent) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    const auto explicit_id = tt::tt_metal::SubDeviceId{2};
    EXPECT_EQ(ttnn::DeviceContext(device).get_effective_sub_device_id(std::optional{explicit_id}), explicit_id);
}

TEST_F(ExecutionContextFixture, GetEffectiveSubDeviceIdUsesCurrentWhenNullopt) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    ttnn::DeviceContext ctx(device);
    const auto default_id = ctx.get_current_sub_device_id();
    EXPECT_EQ(ctx.get_effective_sub_device_id(std::nullopt), default_id);

    auto guard = ctx.set_current_sub_device(tt::tt_metal::SubDeviceId{1});
    EXPECT_EQ(ttnn::DeviceContext(device).get_effective_sub_device_id(std::nullopt), tt::tt_metal::SubDeviceId{1});
}

}  // namespace ttnn::test
