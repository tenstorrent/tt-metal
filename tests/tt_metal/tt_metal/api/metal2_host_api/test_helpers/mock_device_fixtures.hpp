// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Mock-device gtest fixtures shared by the Metal 2.0 Host API unit tests.
//
// The fixtures live in a header (not in each test file) because gtest requires every test of a
// suite to use the same fixture type, and the suites below span several translation units.

#include <memory>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>

namespace tt::tt_metal::experimental::test_helpers {

// One 1x1 MeshDevice on a mock cluster of the given architecture.
//
// Each test owns a private mock MetalEnv, so no process-global mock mode is toggled and any
// runtime state shared with other suites in the same binary is left untouched. The env is declared
// before the MeshDevice so that it outlives it, and TearDown releases them in that order.
template <tt::ARCH Arch>
class MockMeshDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(
            MetalEnvDescriptor{.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(Arch, 1)});
        mesh_device_ = distributed::MeshDevice::open_mesh_device(
            *env_, distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        mesh_device_.reset();
        env_.reset();
    }

    std::unique_ptr<MetalEnv> env_;
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

// ProgramSpec validation and lowering (MakeProgramFromSpec).
class ProgramSpecTestQuasar : public MockMeshDeviceFixture<tt::ARCH::QUASAR> {};
class ProgramSpecTestGen1 : public MockMeshDeviceFixture<tt::ARCH::WORMHOLE_B0> {};
class ProgramSpecTestBlackhole : public MockMeshDeviceFixture<tt::ARCH::BLACKHOLE> {};

// ProgramRunArgs (SetProgramRunArgs / UpdateProgramRunArgs / UpdateTensorArgs).
class ProgramRunArgsTestQuasar : public MockMeshDeviceFixture<tt::ARCH::QUASAR> {};
class ProgramRunArgsTestGen1 : public MockMeshDeviceFixture<tt::ARCH::WORMHOLE_B0> {};

}  // namespace tt::tt_metal::experimental::test_helpers
