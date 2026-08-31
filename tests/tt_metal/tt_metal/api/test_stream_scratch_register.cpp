// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

// Test stream scratch register APIs on Tensix cores (both RISC0 and RISC1)
TEST_F(UnitMeshFixture, StreamScratchRegisterTensixCores) {
    Program program = CreateProgram();

    // Use core (0,0) for testing
    CoreCoord test_core = {0, 0};

    // Create kernel for RISC0 (DataMovementProcessor::RISCV_0)
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_stream_scratch_register.cpp",
        test_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Execute the program
    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);
}

// Test stream scratch register APIs on Erisc cores
TEST_F(UnitMeshFixture, StreamScratchRegisterEriscCores) {
    // Check if device has active ethernet cores
    auto ethernet_cores = this->device().get_devices()[0]->get_active_ethernet_cores(true);
    if (ethernet_cores.empty()) {
        GTEST_SKIP() << "No active ethernet cores available on this device";
    }

    Program program = CreateProgram();

    // Get first available ethernet core
    auto eth_core = *ethernet_cores.begin();

    // Create CoreRangeSet for this single ethernet core
    std::set<CoreRange> eth_core_ranges;
    eth_core_ranges.insert(CoreRange(eth_core, eth_core));

    // Create kernel for ethernet core with EthernetConfig
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_stream_scratch_register.cpp",
        eth_core_ranges,
        EthernetConfig{.noc = NOC::NOC_0});

    // Execute the program
    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);
}

}  // namespace tt::tt_metal
