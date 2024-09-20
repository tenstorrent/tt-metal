// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_fixture.hpp"
#include "test_utils.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking debug ring buffer feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

std::vector<std::string> expected = {
    "debug_ring_buffer=",
    "[0x00270028,0x00260027,0x00250026,0x00240025,0x00230024,0x00220023,0x00210022,0x00200021,",
    " 0x001f0020,0x001e001f,0x001d001e,0x001c001d,0x001b001c,0x001a001b,0x0019001a,0x00180019,",
    " 0x00170018,0x00160017,0x00150016,0x00140015,0x00130014,0x00120013,0x00110012,0x00100011,",
    " 0x000f0010,0x000e000f,0x000d000e,0x000c000d,0x000b000c,0x000a000b,0x0009000a,0x00080009,",
    "]"
};

static void RunTest(WatcherFixture *fixture, Device *device, riscv_id_t riscv_type) {
    // Set up program
    Program program = Program();

    // Depending on riscv type, choose one core to run the test on.
    CoreCoord logical_core, phys_core;
    if (riscv_type == DebugErisc) {
        if (device->get_active_ethernet_cores().empty()) {
            log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
            GTEST_SKIP();
        }
        logical_core = *(device->get_active_ethernet_cores().begin());
        phys_core = device->ethernet_core_from_logical_core(logical_core);
    } else if (riscv_type == DebugIErisc) {
        if (device->get_inactive_ethernet_cores().empty()) {
            log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
            GTEST_SKIP();
        }
        logical_core = *(device->get_inactive_ethernet_cores().begin());
        phys_core = device->ethernet_core_from_logical_core(logical_core);
    } else {
        logical_core = CoreCoord{0, 0};
        phys_core = device->worker_core_from_logical_core(logical_core);
    }
    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, phys_core);


    // Set up the kernel on the correct risc
    KernelHandle assert_kernel;
    switch(riscv_type) {
        case DebugBrisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
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
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
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
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC0", "1"}}
                }
            );
            break;
        case DebugTrisc1:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC1", "1"}}
                }
            );
            break;
        case DebugTrisc2:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                ComputeConfig{
                    .defines = {{"TRISC2", "1"}}
                }
            );
            break;
        case DebugErisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                EthernetConfig{
                    .noc = tt_metal::NOC::NOC_0
                }
            );
            break;
        case DebugIErisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt_metal::NOC::NOC_0
                }
            );
            break;
        default:
            log_info("Unsupported risc type: {}, skipping test...", riscv_type);
            GTEST_SKIP();
    }

    // Run the program
    fixture->RunProgram(device, program, true);

    log_info("Checking file: {}", fixture->log_file_name);

    // Check log
    EXPECT_TRUE(
        FileContainsAllStringsInOrder(
            fixture->log_file_name,
            expected
        )
    );
}

TEST_F(WatcherFixture, TestWatcherRingBufferBrisc) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugBrisc);},
            device
        );
    }
}
TEST_F(WatcherFixture, TestWatcherRingBufferNCrisc) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugNCrisc);},
            device
        );
    }
}
TEST_F(WatcherFixture, TestWatcherRingBufferTrisc0) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugTrisc0);},
            device
        );
    }
}
TEST_F(WatcherFixture, TestWatcherRingBufferTrisc1) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugTrisc1);},
            device
        );
    }
}
TEST_F(WatcherFixture, TestWatcherRingBufferTrisc2) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugTrisc2);},
            device
        );
    }
}
TEST_F(WatcherFixture, TestWatcherRingBufferErisc) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugErisc);},
            device
        );
    }
}
TEST_F(WatcherFixture, TestWatcherRingBufferIErisc) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](WatcherFixture *fixture, Device *device){RunTest(fixture, device, DebugIErisc);},
            device
        );
    }
}
