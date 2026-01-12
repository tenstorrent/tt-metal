// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <cstdlib>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal::test {

// Check if slow dispatch mode is enabled
inline bool IsSlowDispatch() { return std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr; }

enum class DispatchConstraint { SlowOnly, FastOnly, Either };

// Device fixture with dispatch mode constraints for legacy tests
template <DispatchConstraint Constraint = DispatchConstraint::Either>
class LegacyDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        if constexpr (Constraint == DispatchConstraint::SlowOnly) {
            if (!IsSlowDispatch()) {
                GTEST_SKIP() << "Test requires slow dispatch mode (set TT_METAL_SLOW_DISPATCH_MODE=1)";
            }
        } else if constexpr (Constraint == DispatchConstraint::FastOnly) {
            if (IsSlowDispatch()) {
                GTEST_SKIP() << "Test requires fast dispatch mode (unset TT_METAL_SLOW_DISPATCH_MODE)";
            }
        }
        mesh_device_ = distributed::MeshDevice::create_unit_mesh(0);
        device_ = mesh_device_->get_devices()[0];
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    IDevice* device() { return device_; }
    std::shared_ptr<distributed::MeshDevice> mesh_device() { return mesh_device_; }
    distributed::MeshCommandQueue& command_queue() { return mesh_device_->mesh_command_queue(); }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_ = nullptr;
};

// Convenience type aliases
using SlowDispatchFixture = LegacyDeviceFixture<DispatchConstraint::SlowOnly>;
using FastDispatchFixture = LegacyDeviceFixture<DispatchConstraint::FastOnly>;
using EitherDispatchFixture = LegacyDeviceFixture<DispatchConstraint::Either>;

// Host-only fixture (no device required)
using HostOnlyFixture = ::testing::Test;

}  // namespace tt::tt_metal::test
