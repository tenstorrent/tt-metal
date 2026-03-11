// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <map>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// Compile-only test for 2D (mesh) API kernels.
// Verifies mesh/api.h and linear/api.h headers compile with the device toolchain
// when FABRIC_2D is defined. Does NOT run the kernels on hardware.
TEST_F(Fabric2DFixture, CompileOnlyAutoPacketization2D) {
    auto device = get_devices()[0]->get_devices()[0];
    tt::tt_metal::Program program;
    auto core = CoreCoord{0, 0};
    std::map<std::string, std::string> defines = {{"FABRIC_2D", "1"}};

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    tt::tt_metal::detail::CompileProgram(device, program);

    // Second program for additional compile probes (unicast + multicast families)
    tt::tt_metal::Program program2;

    tt::tt_metal::CreateKernel(
        program2,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/compile_probe_unicast_families.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt::tt_metal::CreateKernel(
        program2,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/compile_probe_multicast_families.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    tt::tt_metal::detail::CompileProgram(device, program2);
}

// Compile-only test for 1D (linear) API kernels.
// Uses a separate kernel that only includes linear/api.h (no mesh/api.h).
// Verifies linear/api.h headers compile without FABRIC_2D defined.
TEST_F(Fabric1DFixture, CompileOnlyAutoPacketization1D) {
    auto device = get_devices()[0]->get_devices()[0];
    tt::tt_metal::Program program;
    auto core = CoreCoord{0, 0};

    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/linear_unicast_tx_writer_raw.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::detail::CompileProgram(device, program);

    // Second program for linear compile probes covering all missing families
    tt::tt_metal::Program program2;

    tt::tt_metal::CreateKernel(
        program2,
        "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/linear_compile_probe_all_families.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::detail::CompileProgram(device, program2);
}

}  // namespace tt::tt_fabric::fabric_router_tests
