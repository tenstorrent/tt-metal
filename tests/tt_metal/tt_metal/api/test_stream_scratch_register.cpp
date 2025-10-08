// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal {

// Test stream scratch register APIs on Tensix cores (both RISC0 and RISC1)
TEST_F(MeshDeviceSingleCardFixture, StreamScratchRegisterTensixCores) {
    // Get device from fixture
    auto mesh_device = this->devices_[0];

    // Create workload for mesh device
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    // Use core (0,0) for testing
    CoreCoord test_core = {0, 0};

    // Create kernel for RISC0 (DataMovementProcessor::RISCV_0)
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_stream_scratch_register.cpp",
        test_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Execute the program
    this->RunProgram(mesh_device, workload);
}

// Test stream scratch register APIs on Erisc cores
TEST_F(MeshDeviceSingleCardFixture, StreamScratchRegisterEriscCores) {
    // Get device from fixture
    auto mesh_device = this->devices_[0];
    auto device = mesh_device->get_devices()[0];

    // Check if device has active ethernet cores
    if (device->get_active_ethernet_cores(true).empty()) {
        GTEST_SKIP() << "No active ethernet cores available on this device";
    }

    // Create workload for mesh device
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    // Get first available ethernet core
    auto eth_core = *device->get_active_ethernet_cores(true).begin();

    // Create CoreRangeSet for this single ethernet core
    std::set<CoreRange> eth_core_ranges;
    eth_core_ranges.insert(CoreRange(eth_core, eth_core));

    // Create kernel for ethernet core with EthernetConfig
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_stream_scratch_register.cpp",
        eth_core_ranges,
        EthernetConfig{.noc = NOC::NOC_0});

    // Execute the program
    this->RunProgram(mesh_device, workload);
}

}  // namespace tt::tt_metal
