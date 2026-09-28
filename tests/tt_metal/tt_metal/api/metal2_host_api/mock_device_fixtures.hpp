// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Mock-device gtest fixtures shared by the Metal 2.0 Host API unit tests.
//
// The fixtures live in a header (not in each test file) because gtest requires every test of a
// suite to use the same fixture type, and the suites below span several translation units.

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>

#include "metal2_host_api/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

// One 1x1 MeshDevice on a mock cluster of the given architecture.
//
// Forces TT_METAL_SLOW_DISPATCH_MODE so that MeshDevice::create() succeeds against the
// mock cluster (mock Quasar has no dispatch-core reservation in its descriptor). These
// tests exercise pure API behavior, so slow dispatch is functionally fine.
template <tt::ARCH Arch>
class MockMeshDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        // Configuring global mock mode initializes the HAL for arch checks and Program creation.
        experimental::configure_mock_mode(Arch, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        experimental::disable_mock_mode();
        slow_dispatch_override_.reset();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::optional<ScopedSlowDispatchOverride> slow_dispatch_override_;
};

// ProgramSpec validation and lowering (MakeProgramFromSpec).
class ProgramSpecTestQuasar : public MockMeshDeviceFixture<tt::ARCH::QUASAR> {};
class ProgramSpecTestGen1 : public MockMeshDeviceFixture<tt::ARCH::WORMHOLE_B0> {};
class ProgramSpecTestBlackhole : public MockMeshDeviceFixture<tt::ARCH::BLACKHOLE> {};

// ProgramRunArgs (SetProgramRunArgs / UpdateProgramRunArgs / UpdateTensorArgs).
class ProgramRunArgsTestQuasar : public MockMeshDeviceFixture<tt::ARCH::QUASAR> {};
class ProgramRunArgsTestGen1 : public MockMeshDeviceFixture<tt::ARCH::WORMHOLE_B0> {};

}  // namespace tt::tt_metal::experimental::test_helpers
