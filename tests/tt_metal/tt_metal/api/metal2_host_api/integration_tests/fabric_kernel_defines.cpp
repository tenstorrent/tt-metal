// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric kernel defines on WH/BH silicon: with fabric enabled, a ProgramSpec data-movement kernel is
// compiled with ControlPlane::get_fabric_kernel_defines() (routing mode and packet-header sizing),
// as a CreateKernel data-movement kernel is. Fabric kernels (e.g. a fabric mux client) read these
// to size their packet headers and pick a routing API, so a kernel built without them breaks.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/context/metal_context.hpp"
#include "multi_device_fixture.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;

constexpr uint32_t kReportAddr = 100 * 1024;  // host-known fixed L1 addr
constexpr uint32_t kSlotBytes = 64;           // must match the kernel

// Opens the whole system mesh (fabric needs every chip active) with the parameter's fabric config.
class ProgramSpecFabricDefinesHWTest : public MeshDeviceFixtureBase,
                                       public ::testing::WithParamInterface<tt_fabric::FabricConfig> {
protected:
    ProgramSpecFabricDefinesHWTest() : MeshDeviceFixtureBase(Config{.num_cqs = 1, .fabric_config = GetParam()}) {}

    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        if (mesh_device_->arch() != tt::ARCH::WORMHOLE_B0 && mesh_device_->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

// What this proves end-to-end: every define the control plane hands CreateKernel also reaches the
// compiler for a ProgramSpec data-movement kernel, with the same value.
//
// Verification approach (see fabric_kernel_defines_probe.cpp for the device side): the host lists the
// define names in one user define, FABRIC_DEFINE_PROBES, and the kernel writes each name's
// stringified expansion into its own L1 slot. A forwarded define reads back as its value, a missing
// one as its bare name. Names and values come from get_fabric_kernel_defines(), so the test follows
// that list as it changes.
TEST_P(ProgramSpecFabricDefinesHWTest, DataMovementKernelGetsFabricKernelDefines) {
    const std::map<std::string, std::string> fabric_defines =
        MetalContext::instance().get_control_plane().get_fabric_kernel_defines();
    // Catches a fixture that did not bring fabric up, which would make the comparison below vacuous.
    ASSERT_TRUE(fabric_defines.contains("ROUTING_MODE")) << "Fabric is enabled but there is no ROUTING_MODE define";

    std::string probes;
    for (const auto& [name, _] : fabric_defines) {
        probes += "PROBE(" + name + ")";
    }

    auto dm_kernel = MakeMinimalGen1DMKernel("probe");
    dm_kernel.source =
        "tests/tt_metal/tt_metal/api/metal2_host_api/integration_tests/kernels/fabric_kernel_defines_probe.cpp";
    dm_kernel.runtime_arg_schema = {.runtime_arg_names = {"report_addr"}};
    dm_kernel.compiler_options.defines = {{"FABRIC_DEFINE_PROBES", probes}};

    const NodeCoord node{0, 0};
    ProgramSpec spec{
        .name = "fabric_kernel_defines",
        .kernels = {dm_kernel},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {KernelSpecName{"probe"}}, .target_nodes = node}},
    };
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"probe"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"report_addr", kReportAddr}}),
    }};
    SetProgramRunArgs(program, params);

    // Run on the first chip only; every chip compiles the same kernel. Pre-fill the report with a
    // byte that is not a printable character, so a slot the kernel never wrote cannot pass.
    const auto coordinate = distributed::MeshCoordinate::zero_coordinate(mesh_device_->shape().dims());
    IDevice* device = mesh_device_->get_view().get_devices(distributed::MeshCoordinateRange(coordinate)).at(0);
    const uint32_t report_bytes = fabric_defines.size() * kSlotBytes;
    std::vector<uint32_t> fill(report_bytes / sizeof(uint32_t), 0xFFFFFFFFu);
    detail::WriteToDeviceL1(device, node, kReportAddr, fill);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(coordinate), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> report;
    detail::ReadFromDeviceL1(device, node, kReportAddr, report_bytes, report);
    ASSERT_EQ(report.size() * sizeof(uint32_t), report_bytes);
    const char* slots = reinterpret_cast<const char*>(report.data());

    size_t slot = 0;
    for (const auto& [name, value] : fabric_defines) {
        const char* text = slots + (slot++ * kSlotBytes);
        const std::string seen(text, strnlen(text, kSlotBytes));
        EXPECT_EQ(seen, value) << "Fabric kernel define " << name << " reached the kernel as '" << seen
                               << "' (the bare name means it was not defined)";
    }
}

INSTANTIATE_TEST_SUITE_P(
    FabricConfigs,
    ProgramSpecFabricDefinesHWTest,
    ::testing::Values(tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricConfig::FABRIC_2D),
    [](const ::testing::TestParamInfo<tt_fabric::FabricConfig>& info) {
        return info.param == tt_fabric::FabricConfig::FABRIC_1D ? std::string("Fabric1D") : std::string("Fabric2D");
    });

}  // namespace
}  // namespace tt::tt_metal::experimental
