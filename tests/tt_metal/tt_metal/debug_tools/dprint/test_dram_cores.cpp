// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"

////////////////////////////////////////////////////////////////////////////////
// A test for printing from DRAM programmable cores (DRISC, Blackhole only).
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

const std::string golden_output =
    R"(Test Debug Print: DRISC
Basic Types:
101-1.618@0.122559
1015551234569123456789
-17-343-44444-5123456789
10
Pointer:
123
456
789
SETPRECISION/FIXED/DEFAULTFLOAT:
3.1416
3.14159012
3.14159
3.141590118
SETW:
    123456123456  ab
HEX/OCT/DEC:
1e240361100123456)";

void RunTest(
    DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = mesh_device->get_devices()[0];

    // Test printing on a single DRAM core
    std::vector<CoreCoord> dram_cores = {CoreCoord{0, 0}};

    for (const auto& logical_core : dram_cores) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_print.cpp",
            logical_core,
            DramConfig{.noc = tt_metal::NOC::NOC_0});

        log_info(
            tt::LogTest,
            "Running DPRINT test on DRAM core device {} logical ({},{})",
            device->id(),
            logical_core.x,
            logical_core.y);
        fixture->RunProgram(mesh_device, workload);

        EXPECT_TRUE(FilesMatchesString(fixture->dprint_file_name, golden_output));

        MetalContext::instance().dprint_server()->clear_log_file();
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DPrintMeshFixture, DramTestPrint) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "DRAM cores only support Slow Dispatch (Fast Dispatch not yet supported).");
        GTEST_SKIP();
    }
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        log_info(tt::LogTest, "Skipping: DRAM programmable cores not available on this architecture.");
        GTEST_SKIP();
    }
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device);
            },
            mesh_device);
    }
}
