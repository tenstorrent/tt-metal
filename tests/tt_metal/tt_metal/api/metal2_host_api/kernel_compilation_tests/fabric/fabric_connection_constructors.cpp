// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile tests for building fabric senders and connection managers from a fabric connection binding token
// (mock 4-chip Wormhole). The kernels are compiled but never run.

#include <gtest/gtest.h>

#include <tt-metalium/workload/mesh_workload.hpp>
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
    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::conn);
    sender.open();
    sender.close();
}
)";

constexpr const char* kSenderIfBoundKernel = R"(
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
void kernel_main() {
    if (const auto* conn = fabric::get_token_if_present<"conn">()) {
        auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(*conn);
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
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kFabricSenderKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric2DTest, CPU_SenderBuiltFromBindingJITSmoke) {
    // A kernel builds a WorkerToFabricEdmSender from a one-target binding on a 2D fabric.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kFabricSenderKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_SenderIfBoundBuiltFromTokenIfPresentJITSmoke) {
    // A kernel builds a WorkerToFabricEdmSender from fabric::get_token_if_present when the binding is there.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kSenderIfBoundKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_SenderIfBoundWithoutBindingJITSmoke) {
    // The same kernel source compiles in a KernelSpec that binds no fabric connection.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.advanced_options.fabric_connections.clear();
    spec.kernels[0].advanced_options.fabric_connection_bindings.clear();
    spec.kernels[0].source = KernelSpec::SourceCode{kSenderIfBoundKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionManagerBuiltFromBindingJITSmoke) {
    // A kernel builds a FabricConnectionManager from a {forward, backward} binding.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_neighbour_direction(), fixture_other_neighbour_direction()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kConnectionManagerKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_RoutingPlaneOpenedFromBindingJITSmoke) {
    // A kernel opens a RoutingPlaneConnectionManager from a binding.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_neighbour_direction(), fixture_other_neighbour_direction()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kRoutingPlaneKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric2DTest, CPU_RoutingPlaneOpenedFromBindingJITSmoke) {
    // A kernel opens a RoutingPlaneConnectionManager from a binding on a 2D fabric.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kRoutingPlaneKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

}  // namespace
}  // namespace tt::tt_metal
