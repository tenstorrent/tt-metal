// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_fixture.hpp"
#include "test_utils.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher pause feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

static void RunTest(WatcherFixture* fixture, Device* device) {
    // Set up program
    Program program = Program();

    // Test runs on a 5x5 grid
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = {4, 4};

    // Create all kernels
    auto brisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_pause.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto ncrisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_pause.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto trisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_pause.cpp",
        CoreRange(xy_start, xy_end),
        ComputeConfig{});

    // Write runtime args
    uint32_t clk_mhz = tt::Cluster::instance().get_device_aiclk(device->id());
    uint32_t delay_cycles = clk_mhz * 500000; // .5 secons
    const std::vector<uint32_t> args = { delay_cycles };
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            SetRuntimeArgs(program, brisc_kid, CoreCoord{x, y}, args);
            SetRuntimeArgs(program, ncrisc_kid, CoreCoord{x, y}, args);
            SetRuntimeArgs(program, trisc_kid, CoreCoord{x, y}, args);
        }
    }


    // Also run on ethernet cores if they're present
    bool has_eth_cores = !device->get_active_ethernet_cores(true).empty();
    //bool has_eth_cores = false;
    bool has_ieth_cores = !device->get_inactive_ethernet_cores().empty();

    // TODO: Enable this when FD-on-idle-eth is supported.
    if (!fixture->IsSlowDispatch())
        has_ieth_cores = false;

    if (has_eth_cores) {
        KernelHandle erisc_kid;
        std::set<CoreRange> eth_core_ranges;
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            log_info(LogTest, "Running on eth core {}({})", core.str(), device->ethernet_core_from_logical_core(core).str());
            eth_core_ranges.insert(CoreRange(core, core));
        }
        erisc_kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_pause.cpp",
            eth_core_ranges,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});

        for (const auto& core : device->get_active_ethernet_cores(true)) {
            SetRuntimeArgs(program, erisc_kid, core, args);
        }
    }
    if (has_ieth_cores) {
        KernelHandle ierisc_kid;
        std::set<CoreRange> eth_core_ranges;
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            log_info(LogTest, "Running on inactive eth core {}({})", core.str(), device->ethernet_core_from_logical_core(core).str());
            eth_core_ranges.insert(CoreRange(core, core));
        }
        ierisc_kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_pause.cpp",
            eth_core_ranges,
            tt_metal::EthernetConfig{
                .eth_mode = Eth::IDLE,
                .noc = tt_metal::NOC::NOC_0
            }
        );

        for (const auto& core : device->get_inactive_ethernet_cores()) {
            SetRuntimeArgs(program, ierisc_kid, core, args);
        }
    }

    // Run the program
    fixture->RunProgram(device, program);

    // Check that the pause message is present for each core in the watcher log.
    vector<string> expected_strings;
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            CoreCoord phys_core = device->worker_core_from_logical_core({x, y});
            for (auto &risc_str : {"brisc", "ncrisc", "trisc0", "trisc1", "trisc2"}) {
                string expected = fmt::format("{}:{}", phys_core.str(), risc_str);
                expected_strings.push_back(expected);
            }
        }
    }
    if (has_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            CoreCoord phys_core = device->ethernet_core_from_logical_core(core);
            string expected = fmt::format("{}:erisc", phys_core.str());
            expected_strings.push_back(expected);
        }
    }
    if (has_ieth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            CoreCoord phys_core = device->ethernet_core_from_logical_core(core);
            string expected = fmt::format("{}:ierisc", phys_core.str());
            expected_strings.push_back(expected);
        }
    }
    // See #10527
    // EXPECT_TRUE(FileContainsAllStrings(fixture->log_file_name, expected_strings));
}

TEST_F(WatcherFixture, TestWatcherPause) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
