// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Mock-device fixtures and helpers for Metal 2.0 fabric connection tests (ProgramAdvancedOptions::fabric_connections).
// Kernels are compiled but never run.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/mesh/mesh_device.hpp>
#include <tt-metalium/workload/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::test_helpers {

// The four unit meshes of a mock 4-chip (2x2) Wormhole cluster, with the fabric in mode `Config` and with the tensix
// config `TensixConfig` and the UDM mode `UDMMode`.
// The fabric config is scoped to the fixture's own mock MetalEnv.
template <
    tt_fabric::FabricConfig Config,
    tt_fabric::FabricTensixConfig TensixConfig = tt_fabric::FabricTensixConfig::DISABLED,
    tt_fabric::FabricUDMMode UDMMode = tt_fabric::FabricUDMMode::DISABLED>
class FabricMockDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor{
            .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 4)});
        env_->configure_fabric(
            {.fabric_config = Config, .fabric_tensix_config = TensixConfig, .fabric_udm_mode = UDMMode});
        devices_ = distributed::MeshDevice::open_unit_meshes(*env_, std::array{0, 1, 2, 3});
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

    // The fabric node of device 1, a neighbour of fixture_device().
    tt_fabric::FabricNodeId fixture_neighbour_fabric_node_id() const {
        return devices_.at(1)->get_fabric_node_id(distributed::MeshCoordinate(0, 0));
    }

    // The routing direction from fixture_device() to the device of fixture_neighbour_fabric_node_id().
    tt_fabric::RoutingDirection fixture_neighbour_direction() const {
        const auto& control_plane = devices_.at(0)->impl().metal_context().get_control_plane();
        const auto direction =
            control_plane.get_forwarding_direction(fixture_fabric_node_id(), fixture_neighbour_fabric_node_id());
        TT_FATAL(direction.has_value(), "The mock cluster has no route from chip 0 to chip 1");
        return *direction;
    }

    // A routing direction, other than fixture_neighbour_direction(), in which fixture_device() has a neighbour.
    tt_fabric::RoutingDirection fixture_other_neighbour_direction() const {
        const auto direction = std::ranges::find_if(kDirections, [this](tt_fabric::RoutingDirection d) {
            return d != fixture_neighbour_direction() && !neighbours_in(d).empty();
        });
        TT_FATAL(direction != kDirections.end(), "Chip 0 of the mock cluster has only one neighbour");
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
// One 2x2 mesh on a mock 4-chip Wormhole cluster, with the fabric in mode `Config`.
// The fabric config is scoped to the fixture's own mock MetalEnv.
template <tt_fabric::FabricConfig Config>
class FabricMockMeshFixture : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor{
            .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 4)});
        env_->configure_fabric({.fabric_config = Config});
        mesh_device_ = distributed::MeshDevice::open_mesh_device(
            *env_, distributed::MeshDeviceConfig(distributed::MeshShape{2, 2}));
    }
    void TearDown() override {
        mesh_device_.reset();
        env_.reset();
    }

    // The 2x2 mesh that the Program under test runs on.
    distributed::MeshDevice& fixture_mesh_device() { return *mesh_device_; }

    // The fabric node of the device at `device`.
    tt_fabric::FabricNodeId fixture_fabric_node_id(const distributed::MeshCoordinate& device) const {
        return mesh_device_->get_fabric_node_id(device);
    }

    // The routing direction from the device at `from` to the device at `to`, its neighbour.
    tt_fabric::RoutingDirection fixture_direction(
        const distributed::MeshCoordinate& from, const distributed::MeshCoordinate& to) const {
        const auto& control_plane = mesh_device_->impl().metal_context().get_control_plane();
        const auto direction =
            control_plane.get_forwarding_direction(fixture_fabric_node_id(from), fixture_fabric_node_id(to));
        TT_FATAL(direction.has_value(), "The mock mesh has no route from {} to {}", from, to);
        return *direction;
    }

private:
    std::unique_ptr<MetalEnv> env_;  // outlives the mesh
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

class ProgramSpecFabric1DMeshTest : public FabricMockMeshFixture<tt_fabric::FabricConfig::FABRIC_1D> {};
class ProgramSpecFabric2DMeshTest : public FabricMockMeshFixture<tt_fabric::FabricConfig::FABRIC_2D> {};

class ProgramSpecFabric2DUDMTest : public FabricMockDeviceFixture<
                                       tt_fabric::FabricConfig::FABRIC_2D,
                                       tt_fabric::FabricTensixConfig::UDM,
                                       tt_fabric::FabricUDMMode::ENABLED> {};

// A fabric connections spec "conn" with `targets`, opened from node {0, 0} over link 0.
inline experimental::FabricConnectionsSpec MakeFabricConnectionForTest(experimental::FabricTargets targets) {
    return experimental::FabricConnectionsSpec{
        .unique_id = experimental::FabricConnectionsSpecName{"conn"},
        .targets = std::move(targets),
        .links = {{NodeCoord{0, 0}, 0}}};
}

// A ProgramSpec whose data-movement kernel "writer" runs on node {0, 0} and binds `connection` as fabric::conn.
inline ProgramSpec MakeSpecWithFabricConnectionForTest(experimental::FabricConnectionsSpec connection) {
    KernelSpec writer = MakeMinimalGen1DMKernel("writer");
    writer.advanced_options.fabric_connection_bindings = {experimental::FabricConnectionBinding{
        .fabric_connections_spec_name = connection.unique_id, .accessor_name = "conn"}};
    ProgramSpec spec;
    spec.name = "fabric_connection";
    spec.kernels = {writer};
    spec.advanced_options.fabric_connections = {std::move(connection)};
    spec.work_units = {MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"writer"})};
    return spec;
}

// Not a public API: the runtime args of kernel "writer" on `node`, as the device reads them.
inline std::vector<uint32_t> WriterRuntimeArgsForTest(distributed::MeshWorkload& workload, const NodeCoord& node) {
    Program& program = workload.get_programs().begin()->second;
    return program.impl().get_kernel_by_spec_name("writer")->runtime_args(node);
}

// Not a public API: the connection args that append_fabric_connection_rt_args gives a worker on node {0, 0}
// of a new Program, for the connection from the device at `src` of `mesh_device` to `dst` over link 0.
inline std::vector<uint32_t> LegacyConnectionArgsForTest(
    distributed::MeshDevice& mesh_device,
    const tt_fabric::FabricNodeId& dst,
    const distributed::MeshCoordinate& src = distributed::MeshCoordinate(0, 0)) {
    Program legacy_program = CreateProgram();
    std::vector<uint32_t> args;
    tt_fabric::append_fabric_connection_rt_args(
        mesh_device, mesh_device.get_fabric_node_id(src), dst, 0, legacy_program, {0, 0}, args);
    return args;
}

// Not a public API: the runtime args of kernel `kernel_name` on `node` of the device at `device`, as that device reads
// them: the Program's runtime args, with the workload's per-device values of that device on top.
inline std::vector<uint32_t> KernelRuntimeArgsOnDeviceForTest(
    distributed::MeshWorkload& workload,
    const std::string& kernel_name,
    const distributed::MeshCoordinate& device,
    const NodeCoord& node) {
    auto& [range, program] = *workload.get_programs().begin();
    const KernelHandle kernel = program.impl().get_kernel_handle(kernel_name);
    std::vector<uint32_t> args = program.impl().get_kernel(kernel)->runtime_args(node);
    const distributed::PerDeviceRuntimeArgs* per_device = workload.impl().per_device_runtime_args(range);
    if (per_device == nullptr) {
        return args;
    }
    const size_t index = distributed::coordinate_index_in_range(range, device);
    for (const auto& field : per_device->fields) {
        if (field->kernel == kernel && field->core == node && field->present[index] != 0) {
            std::copy_n(
                field->values.begin() + (index * field->num_words), field->num_words, args.begin() + field->first_word);
        }
    }
    return args;
}

// Not a public API: KernelRuntimeArgsOnDeviceForTest of kernel "writer".
inline std::vector<uint32_t> WriterRuntimeArgsOnDeviceForTest(
    distributed::MeshWorkload& workload, const distributed::MeshCoordinate& device, const NodeCoord& node) {
    return KernelRuntimeArgsOnDeviceForTest(workload, "writer", device, node);
}

}  // namespace tt::tt_metal::test_helpers
