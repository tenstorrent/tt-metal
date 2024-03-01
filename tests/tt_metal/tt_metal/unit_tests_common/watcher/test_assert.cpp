// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_fixture.hpp"
#include "test_utils.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher asserts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

static void RunTest(WatcherFixture *fixture, Device *device, riscv_id_t riscv_type) {
    // Set up program
    Program program = Program();

    // Depending on riscv type, choose one core to run the test on (since the test hangs the board).
    CoreCoord logical_core, phys_core;
    if (riscv_type == DebugErisc) {
        if (device->get_active_ethernet_cores().empty()) {
            log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
            GTEST_SKIP();
        }
        logical_core = *(device->get_active_ethernet_cores().begin());
        phys_core = device->ethernet_core_from_logical_core(logical_core);
    } else {
        logical_core = CoreCoord{0, 0};
        phys_core = device->worker_core_from_logical_core(logical_core);
    }
    log_info(LogTest, "Running test on device {} core {}...", device->id(), phys_core.str());

    // Set up the kernel on the correct risc
    KernelHandle assert_kernel;
    switch(riscv_type) {
        case DebugBrisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default
                }
            );
            break;
        case DebugNCrisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default
                }
            );
            break;
        case DebugTrisc0:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC0", "1"}}
                }
            );
            break;
        case DebugTrisc1:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC1", "1"}}
                }
            );
            break;
        case DebugTrisc2:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC2", "1"}}
                }
            );
            break;
        case DebugErisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0
                }
            );
            break;
        default:
            log_info("Unsupported risc type: {}, skipping test...", riscv_type);
            GTEST_SKIP();
    }

    // Write runtime args that should not trip an assert.
    const std::vector<uint32_t> safe_args = { 3, 4 };
    SetRuntimeArgs(program, assert_kernel, logical_core, safe_args);

    // Run the kernel, don't expect an issue here.
    log_info(LogTest, "Running args that shouldn't assert...");
    fixture->RunProgram(device, program);
    log_info(LogTest, "Args did not assert!");

    // Write runtime args that should trip an assert.
    const std::vector<uint32_t> unsafe_args = { 3, 3 };
    SetRuntimeArgs(program, assert_kernel, logical_core, unsafe_args);

    // Run the kerel, expect an exit due to the assert.
    try {
        log_info(LogTest, "Running args that should assert...");
        fixture->RunProgram(device, program);
    } catch (std::runtime_error &e) {
        string expected = "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.\n";
        expected += tt::watcher_get_log_file_name();
        const string error = string(e.what());
        log_info(LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find(expected) != string::npos);
    }

    // We should be able to find the expected watcher error in the log as well,
    // expected error message depends on the risc we're running on.
    string risc;
    string kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp";
    string debug_status = "*,*,*,*,*";
    int line_num = 27;
    switch(riscv_type) {
        case DebugBrisc:
            risc = "brisc";
            break;
        case DebugNCrisc:
            risc = "ncrisc";
            debug_status = "***,*,*,*,*";
            break;
        case DebugTrisc0:
            risc = "trisc0";
            debug_status = "***,*,*,*,*";
            break;
        case DebugTrisc1:
            risc = "trisc1";
            debug_status = "***,*,*,*,*";
            break;
        case DebugTrisc2:
            risc = "trisc2";
            debug_status = "***,*,*,*,*";
            break;
        case DebugErisc:
            risc = "erisc";
            // TODO: Once #5917 is fixed, remove this hack
            kernel = "blank";
            break;
        default:
            break;
    }

    // TODO: Once #5601 is fixed, we can check the whole string again.
    /*string expected = fmt::format(
        "Device {}, Core (x={},y={}):    {}  {} tripped assert {}:{}",
        device->id(),
        phys_core.x,
        phys_core.y,
        debug_status,
        risc,
        kernel,
        line_num
    );*/
    string expected = fmt::format(
        "  {} tripped assert on line {}. Current kernel: {}",
        risc,
        line_num,
        kernel
    );

    EXPECT_TRUE(
        FileContainsAllStrings(
            fixture->log_file_name,
            {expected}
        )
    );
}

TEST_F(WatcherFixture, TestWatcherAssertBrisc) {
    if (this->slow_dispatch_)
        GTEST_SKIP();

    // Only run on device 0 because this test takes the watcher server down.
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugBrisc);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertNCrisc) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugNCrisc);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertTrisc0) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugTrisc0);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertTrisc1) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugTrisc1);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertTrisc2) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugTrisc2);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertErisc) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugErisc);},
        this->devices_[0]
    );
}
