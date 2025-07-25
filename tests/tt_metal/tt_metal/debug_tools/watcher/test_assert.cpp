// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "dev_msgs.h"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/utils.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher asserts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {
static void RunTest(
    WatcherFixture* fixture,
    IDevice* device,
    riscv_id_t riscv_type,
    debug_assert_type_t assert_type = DebugAssertTripped) {
    // Set up program
    Program program = Program();

    // Depending on riscv type, choose one core to run the test on (since the test hangs the board).
    CoreCoord logical_core, virtual_core;
    if (riscv_type == DebugErisc) {
        if (device->get_active_ethernet_cores(true).empty()) {
            log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
            GTEST_SKIP();
        }
        logical_core = *(device->get_active_ethernet_cores(true).begin());
        virtual_core = device->ethernet_core_from_logical_core(logical_core);
    } else if (riscv_type == DebugIErisc) {
        if (device->get_inactive_ethernet_cores().empty()) {
            log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
            GTEST_SKIP();
        }
        logical_core = *(device->get_inactive_ethernet_cores().begin());
        virtual_core = device->ethernet_core_from_logical_core(logical_core);
    } else {
        logical_core = CoreCoord{0, 0};
        virtual_core = device->worker_core_from_logical_core(logical_core);
    }
    log_info(LogTest, "Running test on device {} core {}...", device->id(), virtual_core.str());

    // Set up the kernel on the correct risc
    KernelHandle assert_kernel;
    std::string risc;
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
            risc = " brisc";
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
            risc = "ncrisc";
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
            risc = "trisc0";
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
            risc = "trisc1";
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
            risc = "trisc2";
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
            risc = "erisc";
            break;
        case DebugIErisc:
            assert_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt_metal::NOC::NOC_0
                }
            );
            risc = "erisc";
            break;
        default: log_info(tt::LogTest, "Unsupported risc type: {}, skipping test...", riscv_type); GTEST_SKIP();
    }

    // Write runtime args that should not trip an assert.
    const std::vector<uint32_t> safe_args = {3, 4, static_cast<uint32_t>(assert_type)};
    SetRuntimeArgs(program, assert_kernel, logical_core, safe_args);

    // Run the kernel, don't expect an issue here.
    log_info(LogTest, "Running args that shouldn't assert...");
    // TODO: #24887, ND issue with this test - only run once below when issue is fixed
    fixture->RunProgram(device, program);
    fixture->RunProgram(device, program);
    fixture->RunProgram(device, program);
    log_info(LogTest, "Args did not assert!");

    // Write runtime args that should trip an assert.
    const std::vector<uint32_t> unsafe_args = {3, 3, static_cast<uint32_t>(assert_type)};
    SetRuntimeArgs(program, assert_kernel, logical_core, unsafe_args);

    // Run the kerel, expect an exit due to the assert.
    log_info(LogTest, "Running args that should assert...");
    fixture->RunProgram(device, program);

    // We should be able to find the expected watcher error in the log as well,
    // expected error message depends on the risc we're running on and the assert type.
    const std::string kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp";
    std::string expected;
    if (assert_type == DebugAssertTripped) {
        const uint32_t line_num = 67;
        expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} tripped an assert on line {}. Current kernel: "
            "{}.",
            device->id(),
            (riscv_type == DebugErisc) ? "acteth" : "worker",
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            risc,
            line_num,
            kernel);
        expected +=
            " Note that file name reporting is not yet implemented, and the reported line number for the assert may be "
            "from a different file.";
    } else {
        std::string barrier;
        if (assert_type == DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped) {
            barrier = "NOC non-posted atomics flushed";
        } else if (assert_type == DebugAssertNCriscNOCNonpostedWritesSentTripped) {
            barrier = "NOC non-posted writes sent";
        } else if (assert_type == DebugAssertNCriscNOCPostedWritesSentTripped) {
            barrier = "NOC posted writes sent";
        } else if (assert_type == DebugAssertNCriscNOCReadsFlushedTripped) {
            barrier = "NOC reads flushed";
        }

        expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} detected an inter-kernel data race due to "
            "kernel completing with pending NOC transactions (missing {} barrier). Current kernel: "
            "{}.",
            device->id(),
            (riscv_type == DebugErisc) ? "acteth" : "worker",
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            risc,
            barrier,
            kernel);
    }

    log_info(LogTest, "Expected error: {}", expected);
    std::string exception = "";
    do {
        exception = MetalContext::instance().watcher_server()->exception_message();
    } while (exception == "");
    log_info(LogTest, "Reported error: {}", exception);
    EXPECT_TRUE(expected == MetalContext::instance().watcher_server()->exception_message());
}
}

TEST_F(WatcherFixture, TestWatcherAssertBrisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_)
        GTEST_SKIP();

    // Only run on device 0 because this test takes the watcher server down.
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){RunTest(fixture, device, DebugBrisc);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertNCrisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){RunTest(fixture, device, DebugNCrisc);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertTrisc0) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){RunTest(fixture, device, DebugTrisc0);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertTrisc1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){RunTest(fixture, device, DebugTrisc1);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertTrisc2) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){RunTest(fixture, device, DebugTrisc2);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertErisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){RunTest(fixture, device, DebugErisc);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherAssertIErisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, IDevice* device){RunTest(fixture, device, DebugIErisc);},
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherNonDefaultAssertBrisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }

    // Only run on device 0 because this test takes the watcher server down.
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) { RunTest(fixture, device, DebugBrisc); }, this->devices_[0]);
}

TEST_F(WatcherFixture, TestWatcherNonDefaultAssertNCrisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            RunTest(fixture, device, DebugNCrisc, DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TestWatcherNonDefaultAssertTrisc0) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            RunTest(fixture, device, DebugTrisc0, DebugAssertNCriscNOCNonpostedWritesSentTripped);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TestWatcherNonDefaultAssertTrisc1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            RunTest(fixture, device, DebugTrisc1, DebugAssertNCriscNOCPostedWritesSentTripped);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TestWatcherNonDefaultAssertTrisc2) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            RunTest(fixture, device, DebugTrisc2, DebugAssertNCriscNOCReadsFlushedTripped);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TestWatcherNonDefaultAssertErisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            RunTest(fixture, device, DebugErisc, DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TestWatcherNonDefaultAssertIErisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            RunTest(fixture, device, DebugIErisc, DebugAssertNCriscNOCReadsFlushedTripped);
        },
        this->devices_[0]);
}
