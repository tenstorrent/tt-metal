// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <string>

#include <gtest/gtest.h>

namespace {
    void RunOneTest(WatcherFixture* fixture, IDevice* device, unsigned usage, bool warning) {
    // Set up program
    Program program = Program();
    CoreCoord coord = {0, 0};
    std::vector<uint32_t> compile_args{usage};

    // Run a kernel that posts waypoints and waits on certain gating values to be written before
    // posting the next waypoint.
    auto brisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack.cpp",
        coord,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
            .compile_args = compile_args});
    auto ncrisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack.cpp",
        coord,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
            .compile_args = compile_args});
    auto trisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack.cpp",
        coord,
        ComputeConfig{.compile_args = compile_args});

    // Also run on ethernet cores if they're present
    bool has_eth_cores = false && !device->get_active_ethernet_cores(true).empty();
    bool has_idle_eth_cores = false && fixture->IsSlowDispatch() &&
        !device->get_inactive_ethernet_cores().empty();
    // FIXME: Implement eth

#if 0
    auto check_core = [&](const CoreCoord& logical_core, const CoreCoord& virtual_core,
                          bool is_eth_core, bool is_active,
                          uint32_t usage, bool warning) {
        vector<string> expected;
        // FIXME: implement
        expected.push_back("bla");
        expected.push_back("bla");
        expected.push_back("bla");
        expected.push_back("bla");
        expected.push_back("bla");
        expected.push_back("bla");
        expected.push_back("bla");
        EXPECT_TRUE(
            FileContainsAllStringsInOrder(
                fixture->log_file_name,
                expected
            )
        );
    };
#endif
    vector<string> expected;
    // FIXME: implement
    expected.push_back("bla");
    expected.push_back("bla");
    expected.push_back("bla");

    fixture->RunProgram(device, program, true);
#if 0
    CoreCoord virtual_core = device->worker_core_from_logical_core(coord);
    check_core(coord, virtual_core, false, false, usage, warning);
    if (has_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            check_core(core, virtual_core, true, true, usage, warning);
        }
    }
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            check_core(core, virtual_core, true, false, usage, warning);
        }
    }
#endif
    EXPECT_TRUE(
        FileContainsAllStrings(
            fixture->log_file_name,
            expected
            )
        );
}

void RunTest(WatcherFixture* fixture, IDevice* device) {
    RunOneTest(fixture, device, 0, true);
    RunOneTest(fixture, device, 16, true);
    RunOneTest(fixture, device, 128, false);
    RunOneTest(fixture, device, 8192, false);
}

} // namespace

TEST_F(WatcherFixture, TestWatcherStackUsage) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
