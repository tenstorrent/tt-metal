// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile test for the fabric kernel defines: with fabric enabled, a ProgramSpec data-movement
// kernel is compiled with ControlPlane::get_fabric_kernel_defines(), as a CreateKernel one is
// (mock 2-chip Wormhole).

#include <gtest/gtest.h>

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

// Fabric needs every chip open, so both mock chips are opened and the program is built on the first.
// The fabric config is scoped to the test's own mock MetalEnv, so nothing needs restoring afterwards.
class ProgramSpecFabricDefinesTest : public ::testing::TestWithParam<tt_fabric::FabricConfig> {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor{
            .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
        env_->configure_fabric({.fabric_config = GetParam()});
        devices_ = distributed::MeshDevice::open_unit_meshes(*env_, std::array{0, 1});
    }
    void TearDown() override {
        devices_.clear();
        env_.reset();
    }

    std::unique_ptr<MetalEnv> env_;  // outlives the meshes
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> devices_;
};

TEST_P(ProgramSpecFabricDefinesTest, CPU_DataMovementKernelGetsFabricKernelDefinesJITSmoke) {
    const auto fabric_defines = devices_.at(0)->impl().metal_context().get_control_plane().get_fabric_kernel_defines();
    // Without fabric up the map is empty and the kernel would check nothing.
    ASSERT_TRUE(fabric_defines.contains("ROUTING_MODE"));

    // Each define must be present, with the control plane's value.
    std::string source;
    for (const auto& [name, value] : fabric_defines) {
        source += "#ifndef " + name + "\n#error \"" + name + " not forwarded\"\n#endif\n";
        source += "static_assert((" + name + ") == (" + value + "), \"" + name + " != " + value + "\");\n";
    }
    source += "void kernel_main() {}\n";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{source};

    ProgramSpec spec;
    spec.name = "fabric_kernel_defines";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};

    auto& mesh_device = devices_.at(0);
    Program program = MakeProgramFromSpec(*mesh_device, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device.get()));
}

INSTANTIATE_TEST_SUITE_P(
    MockWormhole,
    ProgramSpecFabricDefinesTest,
    ::testing::Values(tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricConfig::FABRIC_2D),
    [](const ::testing::TestParamInfo<tt_fabric::FabricConfig>& info) {
        return info.param == tt_fabric::FabricConfig::FABRIC_1D ? std::string("Fabric1D") : std::string("Fabric2D");
    });

}  // namespace
}  // namespace tt::tt_metal::experimental
