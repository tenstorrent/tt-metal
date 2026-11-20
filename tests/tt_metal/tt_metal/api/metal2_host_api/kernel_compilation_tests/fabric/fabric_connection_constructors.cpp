// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile tests for building fabric senders and connection managers from a fabric connection binding token
// (mock 2-chip Wormhole). The kernels are compiled but never run.

#include <gtest/gtest.h>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DTest;
using test_helpers::ProgramSpecFabric2DTest;

constexpr const char* kFabricSenderKernel = R"(
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
void kernel_main() {
    if (tt::tt_fabric::has_fabric_connection(fabric::conn, 0)) {
        auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::conn);
        sender.open();
        sender.close();
    }
}
)";

constexpr const char* kConnectionManagerKernel = R"(
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
void kernel_main() {
    auto connections = FabricConnectionManager::build_from_binding<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(fabric::conn);
    connections.close();
}
)";

constexpr const char* kRoutingPlaneKernel = R"(
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::common::experimental;
void kernel_main() {
    tt::tt_fabric::RoutingPlaneConnectionManager connections;
    open_connections(connections, fabric::conn);
    close_connections(connections);
}
)";

TEST_F(ProgramSpecFabric1DTest, CPU_SenderBuiltFromBindingJITSmoke) {
    // A kernel builds a WorkerToFabricEdmSender from a one-target binding.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kFabricSenderKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric2DTest, CPU_SenderBuiltFromBindingJITSmoke) {
    // A kernel builds a WorkerToFabricEdmSender from a one-target binding on a 2D fabric.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kFabricSenderKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionManagerBuiltFromBindingJITSmoke) {
    // A kernel builds a FabricConnectionManager from a {forward, backward} binding.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        Fabric1DTargets{fixture_neighbour_direction(), fixture_direction_without_neighbour()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kConnectionManagerKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_RoutingPlaneOpenedFromBindingJITSmoke) {
    // A kernel opens a RoutingPlaneConnectionManager from a binding.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        Fabric1DTargets{fixture_neighbour_direction(), fixture_direction_without_neighbour()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kRoutingPlaneKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric2DTest, CPU_RoutingPlaneOpenedFromBindingJITSmoke) {
    // A kernel opens a RoutingPlaneConnectionManager from a binding on a 2D fabric.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kRoutingPlaneKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

}  // namespace
}  // namespace tt::tt_metal
