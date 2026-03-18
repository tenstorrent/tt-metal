// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;

class DevicePrintOutputFixture : public DevicePrintFixture {
public:
    void TestOutput(
        const std::string& kernel_path,
        const std::vector<std::string>& expected_messages,
        stl::Span<const uint32_t> runtime_args = {}) {
        for (auto& mesh_device : this->devices_) {
            RunProgram(mesh_device, kernel_path, runtime_args);
            EXPECT_TRUE(FileContainsAllStrings(dprint_file_name, expected_messages));
        }
    }
};

TEST_F(DevicePrintOutputFixture, PrintSimpleString) {
    std::vector<std::string> messages = {
        "Hello world!",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_simple_string.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintSingleUintArg) {
    std::vector<uint32_t> runtime_args = {42};
    std::vector<std::string> messages = {
        "Printing uint32_t from arg: 42",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_single_uint_arg.cpp", messages, runtime_args);
}

TEST_F(DevicePrintOutputFixture, PrintBasicTypes) {
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

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_basic_types.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintWithFormatSpecified) {
    std::vector<std::string> messages = {
        "int8_t:         -8",
        "uint8_t: 0B1000",
        "int16_t: -16       ",
        "uint16_t: 0X10",
        "int32_t:    -32    ",
        "uint32_t: 0x20",
        "int64_t: -64",
        "uint64_t: 0X000040",
        "float: 3.14",
        "double: 6.28000",
        "bool: true",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_with_format_specified.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintManyIterations) {
    uint32_t iterations = 1000;
    std::vector<uint32_t> runtime_args = {iterations};
    std::vector<std::string> messages;

    messages.reserve(iterations);
    for (uint32_t i = 0; i < iterations; i++) {
        messages.push_back("Test iteration: " + std::to_string(i));
    }

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_iterations.cpp", messages, runtime_args);
}

// Test that printing from multiple RISCs on the same core works and doesn't interleave messages.
// We detect interleaving by having garbage data in server output.
// If all messages are present and correctly formatted, we can be reasonably sure that there was no interleaving.
TEST_F(DevicePrintOutputFixture, PrintConcurrentAllRiscs) {
    size_t device_counter = 0;
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->arch() != tt::ARCH::WORMHOLE_B0 && mesh_device->arch() != tt::ARCH::BLACKHOLE) {
            // Test currently works only on WH and BH
            continue;
        }
        device_counter++;

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        constexpr CoreCoord core = {0, 0};
        uint32_t iterations_count = 100;
        std::vector<uint32_t> runtime_args = {iterations_count};

        // BRISC
        auto kernel_handle = CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/device_print/print_iterations.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

        // NCRISC
        kernel_handle = CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/device_print/print_iterations.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

        // TRISC0 (Unpack), TRISC1 (Math), TRISC2 (Pack)
        kernel_handle = CreateKernel(
            program_, "tests/tt_metal/tt_metal/test_kernels/device_print/print_iterations.cpp", core, ComputeConfig{});

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);

        DebugToolsMeshFixture::RunProgram(mesh_device, workload);
        MetalContext::instance().dprint_server()->await();

        // Verify all 5*N messages (5 RISCs x N iterations) are correctly formatted.
        // Each iteration value must appear exactly 5 times — once per RISC.
        std::fstream log_file;
        ASSERT_TRUE(OpenFile(dprint_file_name, log_file, std::fstream::in));
        std::vector<int> counts(iterations_count, 0);
        std::string line;
        for (;;) {
            if (!getline(log_file, line)) {
                break;
            }
            int iter = -1;
            if (sscanf(line.c_str(), "Test iteration: %d", &iter) == 1 && iter >= 0 && iter < counts.size()) {
                counts[iter]++;
            }
        }
        for (int i = 0; i < static_cast<int>(counts.size()); i++) {
            EXPECT_EQ(counts[i], 5 * device_counter) << "Iteration " << i << " appeared " << counts[i]
                                                     << " times (expected " << 5 * device_counter << " times)";
        }
    }
}
