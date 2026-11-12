// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of DataMovementHardwareConfig::config_1xx (data_movement_hardware_config.hpp).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestGen1, CPU_DMKernelWithoutGen1SpecificFails) {
    // Gen1 DM has no default processor/NOC. A disengaged config_1xx is rejected at
    // validation, not later at lowering.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    std::get<DataMovementHardwareConfig>(kernel.hw_config).config_1xx.reset();

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no config_1xx processor/NOC")));
}

TEST_F(ProgramSpecTestGen1, CPU_DMProcessorBeyondRiscv1Fails) {
    // Gen1 has only RISCV_0 (BRISC) and RISCV_1 (NCRISC); RISCV_2..7 are Gen2/Quasar-only. A Gen1 DM
    // kernel requesting one must be rejected (parity with the legacy CreateDataMovementKernel guard).
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    (*std::get<DataMovementHardwareConfig>(kernel.hw_config).config_1xx).processor = DataMovementProcessor::RISCV_2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Gen1 has only")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DMKernelWithDefaultConfigSucceeds) {
    // On Gen2 a DM kernel needs no explicit tuning: Gen2 has a unified NOC and fully automated DM
    // placement. A default DataMovementHardwareConfig{} (no generation extras) is all that's required.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

}  // namespace
}  // namespace tt::tt_metal::experimental
