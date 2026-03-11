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

void RunCompileOnlyKernelsTest(BaseFabricFixture* fixture) {
    auto device = fixture->get_devices()[0]->get_devices()[0];
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
}

TEST_F(BaseFabricFixture, CompileOnlyAutoPacketizationKernels) {
    RunCompileOnlyKernelsTest(this);
}

}  // namespace tt::tt_fabric::fabric_router_tests
