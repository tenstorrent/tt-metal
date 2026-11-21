// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Mock-device fixtures and helpers for Metal 2.0 fabric connection tests (ProgramSpec::fabric_connections).
// Kernels are compiled but never run.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::test_helpers {

// Two unit meshes on a mock 2-chip Wormhole cluster, with the fabric in mode `Config` and with the tensix config
// `TensixConfig` and the UDM mode `UDMMode`.
// The fabric config is scoped to the fixture's own mock MetalEnv.
template <
    tt_fabric::FabricConfig Config,
    tt_fabric::FabricTensixConfig TensixConfig = tt_fabric::FabricTensixConfig::DISABLED,
    tt_fabric::FabricUDMMode UDMMode = tt_fabric::FabricUDMMode::DISABLED>
class FabricMockDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor{
            .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
        env_->configure_fabric(
            {.fabric_config = Config, .fabric_tensix_config = TensixConfig, .fabric_udm_mode = UDMMode});
        devices_ = distributed::MeshDevice::open_unit_meshes(*env_, std::array{0, 1});
    }
    void TearDown() override {
        devices_.clear();
        env_.reset();
    }

    // The device that the Program under test runs on.
    distributed::MeshDevice& fixture_device() { return *devices_.at(0); }

    // The fabric node of fixture_device().
    tt_fabric::FabricNodeId fixture_fabric_node_id() const {
        return devices_.at(0)->get_fabric_node_id(distributed::MeshCoordinate(0, 0));
    }

    // The fabric node of the other device, a neighbour of fixture_device().
    tt_fabric::FabricNodeId fixture_neighbour_fabric_node_id() const {
        return devices_.at(1)->get_fabric_node_id(distributed::MeshCoordinate(0, 0));
    }

    // The routing direction from fixture_device() to its neighbour.
    tt_fabric::RoutingDirection fixture_neighbour_direction() const {
        const auto direction = std::ranges::find_if(
            kDirections, [this](tt_fabric::RoutingDirection d) { return !neighbours_in(d).empty(); });
        TT_FATAL(direction != kDirections.end(), "The mock cluster has no neighbour of chip 0");
        return *direction;
    }

    // A routing direction in which fixture_device() has no neighbour.
    tt_fabric::RoutingDirection fixture_direction_without_neighbour() const {
        const auto direction = std::ranges::find_if(
            kDirections, [this](tt_fabric::RoutingDirection d) { return neighbours_in(d).empty(); });
        TT_FATAL(direction != kDirections.end(), "Chip 0 of the mock cluster has a neighbour in every direction");
        return *direction;
    }

private:
    static constexpr std::array kDirections = {
        tt_fabric::RoutingDirection::E,
        tt_fabric::RoutingDirection::W,
        tt_fabric::RoutingDirection::N,
        tt_fabric::RoutingDirection::S};

    std::vector<ChipId> neighbours_in(tt_fabric::RoutingDirection direction) const {
        const auto& control_plane = devices_.at(0)->impl().metal_context().get_control_plane();
        const tt_fabric::FabricNodeId src = devices_.at(0)->get_fabric_node_id(distributed::MeshCoordinate(0, 0));
        const auto neighbours = control_plane.get_chip_neighbors(src, direction);
        const auto chips = neighbours.find(src.mesh_id);
        return chips == neighbours.end() ? std::vector<ChipId>{}
                                         : std::vector<ChipId>(chips->second.begin(), chips->second.end());
    }

    std::unique_ptr<MetalEnv> env_;  // outlives the meshes
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> devices_;
};

class ProgramSpecFabric1DTest : public FabricMockDeviceFixture<tt_fabric::FabricConfig::FABRIC_1D> {};
class ProgramSpecFabric2DTest : public FabricMockDeviceFixture<tt_fabric::FabricConfig::FABRIC_2D> {};
class ProgramSpecFabric2DUDMTest : public FabricMockDeviceFixture<
                                       tt_fabric::FabricConfig::FABRIC_2D,
                                       tt_fabric::FabricTensixConfig::UDM,
                                       tt_fabric::FabricUDMMode::ENABLED> {};

// A fabric connection spec "conn" with `targets`, opened from node {0, 0} over link 0.
inline FabricConnectionSpec MakeFabricConnectionForTest(std::variant<Fabric1DTargets, Fabric2DTargets> targets) {
    return FabricConnectionSpec{
        .unique_id = FabricConnectionSpecName{"conn"}, .targets = std::move(targets), .links = {{NodeCoord{0, 0}, 0}}};
}

// A ProgramSpec whose data-movement kernel "writer" runs on node {0, 0} and binds `connection` as fabric::conn.
inline ProgramSpec MakeSpecWithFabricConnectionForTest(FabricConnectionSpec connection) {
    KernelSpec writer = MakeMinimalGen1DMKernel("writer");
    writer.fabric_connection_bindings = {
        FabricConnectionBinding{.fabric_connection_spec_name = connection.unique_id, .accessor_name = "conn"}};
    ProgramSpec spec;
    spec.name = "fabric_connection";
    spec.kernels = {writer};
    spec.fabric_connections = {std::move(connection)};
    spec.work_units = {MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"writer"})};
    return spec;
}

// Not a public API: the runtime args of kernel "writer" on `node`, as the device reads them.
inline std::vector<uint32_t> WriterRuntimeArgsForTest(distributed::MeshWorkload& workload, const NodeCoord& node) {
    Program& program = workload.get_programs().begin()->second;
    return program.impl().get_kernel_by_spec_name("writer")->runtime_args(node);
}

// Not a public API: the connection args that append_fabric_connection_rt_args gives a worker on node {0, 0}
// of a new Program, for the connection from `device` to `dst` over link 0.
inline std::vector<uint32_t> LegacyConnectionArgsForTest(
    distributed::MeshDevice& device, const tt_fabric::FabricNodeId& dst) {
    Program legacy_program = CreateProgram();
    std::vector<uint32_t> args;
    tt_fabric::append_fabric_connection_rt_args(
        device, device.get_fabric_node_id(distributed::MeshCoordinate(0, 0)), dst, 0, legacy_program, {0, 0}, args);
    return args;
}

}  // namespace tt::tt_metal::test_helpers
