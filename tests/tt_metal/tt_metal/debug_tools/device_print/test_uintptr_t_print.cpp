// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal_proc_set.hpp"
#include <deque>
#include <sstream>
#include <string>
#include <string_view>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <umd/device/types/arch.hpp>

using namespace tt::tt_metal;

namespace {

std::string BuildDprintOutputMismatchMessage(
    const std::string& file_name, const std::vector<std::string>& expected_messages) {
    std::fstream log_file;
    if (!OpenFile(file_name, log_file, std::fstream::in)) {
        return "Could not open DPRINT file: " + file_name;
    }

    std::ostringstream full_output;
    std::deque<std::string_view> remaining(expected_messages.begin(), expected_messages.end());

    for (std::string line; getline(log_file, line);) {
        full_output << line << '\n';
        for (; !remaining.empty(); remaining.pop_front()) {
            if (!StringContainsGlob(line, remaining.front())) {
                break;
            }
        }
    }

    std::ostringstream message;
    message << "DPRINT file: " << file_name << '\n';
    if (!remaining.empty()) {
        message << "Missing expected strings (in order):";
        for (const auto& expected : remaining) {
            message << "\n  - \"" << expected << '"';
        }
        message << '\n';
    }
    message << "\nActual output:\n" << full_output.str();
    return message.str();
}

void ExpectDprintOutputInOrder(const std::string& dprint_file_name, const std::vector<std::string>& expected_messages) {
    if (!FileContainsAllStringsInOrder(dprint_file_name, expected_messages)) {
        FAIL() << BuildDprintOutputMismatchMessage(dprint_file_name, expected_messages);
    }
}

void RunQuasarDprintKernel(
    DevicePrintFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    std::string_view kernel_id,
    std::string_view kernel_path,
    std::string_view program_name) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    experimental::KernelSpec kernel_spec{
        .unique_id = std::string(kernel_id),
        .source = std::string(kernel_path),
        .num_threads = 1,
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{},
            },
    };

    experimental::WorkUnitSpec work_unit{
        .name = "main",
        .kernels = {std::string(kernel_id)},
        .target_nodes = experimental::NodeCoord{0, 0},
    };

    experimental::ProgramSpec spec{
        .name = std::string(program_name),
        .kernels = {kernel_spec},
        .work_units = {work_unit},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);
}

}  // namespace

class DevicePrintUintptrOutputFixture : public DevicePrintFixture {
protected:
    void ExtraSetUp() override {
        // Quasar uses a shared per-core DPRINT buffer. Restrict host polling to DM2 where the
        // Metal 2.0 test kernel runs (num_threads=1 maps to the first user DM).
        HalProcessorSet processors;
        processors.add(HalProgrammableCoreType::TENSIX, 2);
        MetalContext::instance().rtoptions().set_feature_processors(tt::llrt::RunTimeDebugFeatureDprint, processors);
    }

public:
    void TestOutput(
        std::string_view kernel_id,
        std::string_view kernel_path,
        std::string_view program_name,
        const std::vector<std::string>& expected_messages) {
        if (MetalContext::instance().hal().get_arch() != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Test only applicable to Quasar";
        }

        for (auto& mesh_device : this->devices_) {
            RunQuasarDprintKernel(this, mesh_device, kernel_id, kernel_path, program_name);
            ExpectDprintOutputInOrder(dprint_file_name, expected_messages);
        }
    }
};

namespace {
constexpr const char* PRINT_UINTPTR_KERNEL = "print_uintptr";
constexpr const char* PRINT_UINTPTR_KERNEL_PATH = "tests/tt_metal/tt_metal/test_kernels/device_print/print_uintptr.cpp";
constexpr const char* PRINT_UINTPTR_WRAP_KERNEL = "print_uintptr_buffer_wrap";
constexpr const char* PRINT_UINTPTR_WRAP_KERNEL_PATH =
    "tests/tt_metal/tt_metal/test_kernels/device_print/print_uintptr_buffer_wrap.cpp";
}  // namespace

TEST_F(DevicePrintUintptrOutputFixture, PrintUintptrValues) {
    std::vector<std::string> messages = {
        "idx=42 uintptr=305419896",
        "uintptr=305419896 idx=42",
        "uintptr=305419896",
        "idx=42 void_ptr=0xabcdef00",
        "idx=42 const_char_ptr=0x*",
        "idx=42 char_ptr=0x*",
        "idx=42 float=3.14",
        "idx=42 ctstr=compile time string",
        "idx=42 u64=1311768467463790320",
        "idx=42 i64=-4886718345",
        "idx=42 double=6.25",
    };

    TestOutput(PRINT_UINTPTR_KERNEL, PRINT_UINTPTR_KERNEL_PATH, "print_uintptr", messages);
}

TEST_F(DevicePrintUintptrOutputFixture, PrintUintptrBufferWrap) {
    // After many small prints fill and wrap the ring buffer, mixed 4/8-byte args must still parse correctly.
    TestOutput(
        PRINT_UINTPTR_WRAP_KERNEL,
        PRINT_UINTPTR_WRAP_KERNEL_PATH,
        "print_uintptr_buffer_wrap",
        {
            "fill 0",
            "fill 399",
            "wrap idx=1 float=2.5 u64=1311768467463790320",
            "wrap_done=1",
        });
}
