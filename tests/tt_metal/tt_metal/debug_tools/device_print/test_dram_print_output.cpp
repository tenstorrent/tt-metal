// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"

////////////////////////////////////////////////////////////////////////////////
// DEVICE_PRINT tests for DRAM programmable cores (DRISC, Blackhole only).
// Mirrors select tests from test_print_output.cpp but targets DRAM core 0,0.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

class DevicePrintDramFixture : public DevicePrintFixture {
public:
    void RunDramProgram(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        const std::string& kernel_path,
        stl::Span<const uint32_t> runtime_args = {}) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        constexpr CoreCoord core = {0, 0};
        KernelHandle kernel_handle = CreateKernel(program_, kernel_path, core, DramConfig{.noc = tt_metal::NOC::NOC_0});

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();
    }

    void TestDramOutput(
        const std::string& kernel_path,
        const std::vector<std::string>& expected_messages,
        stl::Span<const uint32_t> runtime_args = {}) {
        for (auto& mesh_device : this->devices_) {
            RunDramProgram(mesh_device, kernel_path, runtime_args);
            EXPECT_TRUE(FileContainsAllStrings(dprint_file_name, expected_messages));
        }
    }
};

#define DRAM_SKIP_GUARDS()                                                                                  \
    do {                                                                                                    \
        if (!this->IsSlowDispatch()) {                                                                      \
            log_info(tt::LogTest, "DRAM cores only support Slow Dispatch (Fast Dispatch not yet supported).");  \
            GTEST_SKIP();                                                                                   \
        }                                                                                                   \
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();                                     \
        if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {                               \
            log_info(tt::LogTest, "Skipping: DRAM programmable cores not available on this architecture."); \
            GTEST_SKIP();                                                                                   \
        }                                                                                                   \
    } while (0)

TEST_F(DevicePrintDramFixture, DramPrintSimpleString) {
    DRAM_SKIP_GUARDS();
    std::vector<std::string> messages = {
        "Hello world!",
        "First line.",
        "Second line.",
    };
    TestDramOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_simple_string.cpp", messages);
}

TEST_F(DevicePrintDramFixture, DramPrintSingleUintArg) {
    DRAM_SKIP_GUARDS();
    std::vector<uint32_t> runtime_args = {42};
    std::vector<std::string> messages = {
        "Printing uint32_t from arg: 42",
    };
    TestDramOutput(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_single_uint_arg.cpp", messages, runtime_args);
}

TEST_F(DevicePrintDramFixture, DramPrintBasicTypes) {
    DRAM_SKIP_GUARDS();
    std::vector<std::string> messages = {
        "int8_t: -8",
        "uint8_t: 8",
        "int16_t: -16",
        "uint16_t: 16",
        "int32_t: -32",
        "uint32_t: 32",
        "int64_t: -64",
        "uint64_t: 64",
        "float: 3.14",
        "double: 6.28",
        "bool: true",
        "bf4_t: 0.5",
        "bf8_t: 0.375",
        "bf16_t: 0.122558594",
        "Reordered args: true -16 -32 -64",
        "Reordered args: true -16 -32 -64",
    };
    TestDramOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_basic_types.cpp", messages);
}

TEST_F(DevicePrintDramFixture, DramPrintFactorial) {
    DRAM_SKIP_GUARDS();
    std::vector<uint32_t> runtime_args = {5};
    std::vector<std::string> messages = {
        "factorial(5) = 120",
    };
    TestDramOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_factorial.cpp", messages, runtime_args);
}

#undef DRAM_SKIP_GUARDS
