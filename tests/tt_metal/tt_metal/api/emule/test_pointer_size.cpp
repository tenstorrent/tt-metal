// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="MeshDeviceFixture.Local_L1_Alignment_*"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(MeshDeviceFixture, Local_L1_Alignment_SanityCheck) {
    GTEST_SKIP() << "Temporarily disabled.";

    auto* device = this->devices_.at(0)->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // Inline kernel that tries to get an unaligned local L1 pointer
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            // 0x5 is not 4-byte aligned.
            // This calls __emule_local_l1_to_ptr inside the emulator.
            volatile uint32_t* bad_ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(0x5);
            *bad_ptr = 0xDEADBEEF;
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(detail::LaunchProgram(device, program), ".*Local L1 Alignment: Offset 0x5 must be 4-byte aligned.*");
}

}  // namespace tt::tt_metal
