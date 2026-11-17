// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <map>
#include <memory>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::test_helpers {

// Two unit meshes on a mock 2-chip Wormhole cluster, with 1D fabric configured.
// Kernels never run; Programs are built and inspected on the host.
class ProgramSpecFabricTestGen1 : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor{
            .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
        env_->configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D});
        devices_ = distributed::MeshDevice::open_unit_meshes(*env_, std::array{0, 1});
    }
    void TearDown() override {
        devices_.clear();
        env_.reset();
    }

    // The device that the Program under test runs on.
    distributed::MeshDevice& fixture_source_device() { return *devices_.at(0); }

    // The fabric node of fixture_source_device().
    tt_fabric::FabricNodeId fixture_source_fabric_node_id() const {
        return devices_.at(0)->get_fabric_node_id(distributed::MeshCoordinate(0, 0));
    }

    // The fabric node of the other device, a neighbor of fixture_source_device().
    tt_fabric::FabricNodeId fixture_destination_fabric_node_id() const {
        return devices_.at(1)->get_fabric_node_id(distributed::MeshCoordinate(0, 0));
    }

private:
    std::unique_ptr<MetalEnv> env_;  // outlives the meshes
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> devices_;
};

}  // namespace tt::tt_metal::test_helpers
