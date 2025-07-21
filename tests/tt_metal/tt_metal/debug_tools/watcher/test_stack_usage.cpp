// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <gtest/gtest.h>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <fmt/base.h>
#include <string>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {
void RunOneTest(WatcherFixture* fixture, IDevice* device, unsigned free) {
    static const char *const names[] =
        {"brisc", "ncrisc", "trisc0", "trisc1", "trisc2", "aerisc", "ierisc"};
    const std::string path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack.cpp";
    auto msg = [&](std::vector<std::string> &msgs, const char *cpu, unsigned free) {
        if (msgs.empty()) {
            msgs.push_back("Stack usage summary:");
        }
        msgs.push_back(fmt::format("{} highest stack usage: {} bytes free, on core "
                                   "* running kernel {} ({})",
                                   cpu, free, path, !free ? "OVERFLOW" : "Close to overflow"));
    };

    // Set up program
    Program program = Program();
    CoreCoord coord = {0, 0};
    std::vector<uint32_t> compile_args{free};
    std::vector<std::string> expected;

    CreateKernel(program, path, coord,
                 DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                     .noc = NOC::RISCV_0_default,
                     .compile_args = compile_args});
    msg(expected, names[0], free);

    CreateKernel(program, path, coord,
                 DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                     .noc = NOC::RISCV_1_default,
                     .compile_args = compile_args});
    msg(expected, names[1], free);

    CreateKernel(program, path, coord, ComputeConfig{.compile_args = compile_args});
    for (unsigned ix = 0; ix != 2; ix++) {
        msg(expected, names[2 + ix], free);
    }

    // Also run on idle ethernet, if present
    auto const &inactive_eth_cores = device->get_inactive_ethernet_cores();
    if (!inactive_eth_cores.empty() && fixture->IsSlowDispatch()) {
        // Just pick the first core
        CoreCoord idle_coord = CoreCoord(*inactive_eth_cores.begin());
        CreateKernel(program, path, idle_coord,
                     tt_metal::EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0,
                         .processor = DataMovementProcessor::RISCV_0, .compile_args = compile_args});
        msg(expected, names[6], free);
    }

    fixture->RunProgram(device, program, true);

    EXPECT_TRUE(FileContainsAllStringsInOrder(fixture->log_file_name, expected));
}

template<uint32_t Free>
void RunTest(WatcherFixture* fixture, IDevice* device) {
    RunOneTest(fixture, device, Free);
}

} // namespace

TEST_F(WatcherFixture, TestWatcherStackUsage0) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(RunTest<0>, device);
    }
    // Trigger a watcher re-init, so that the stack usage is reset.
    this->reset_server = true;
}

TEST_F(WatcherFixture, TestWatcherStackUsage16) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(RunTest<16>, device);
    }
    // Trigger a watcher re-init, so that the stack usage is reset.
    this->reset_server = true;
}
