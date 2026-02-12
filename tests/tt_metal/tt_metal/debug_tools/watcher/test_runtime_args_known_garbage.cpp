// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <gtest/gtest.h>

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "debug_tools_fixture.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

// Fixture for RTA/CRTA watcher bounds check tests
class RTATestFixture : public MeshWatcherFixture {
protected:
    void SetUp() override {
        bool watcher_assert_disabled = MetalContext::instance().rtoptions().watcher_assert_disabled();
        if (watcher_assert_disabled) {
            GTEST_SKIP() << "This test requires watcher assert checks (RTA/CRTA) to be enabled";
        }
        MeshWatcherFixture::SetUp();
    }
    // Common test data
    const std::vector<uint32_t> default_rtas{0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};
    const std::vector<uint32_t> default_crtas{0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777};

    distributed::MeshCoordinate zero_coord{0, 0};
    distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

    // Helper: Create CB config for RTA/CRTA tests
    CircularBufferConfig CreateArgCBConfig(uint32_t word_count) {
        const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
        uint32_t cb_size = tt::align(word_count * sizeof(uint32_t), l1_alignment);
        return CircularBufferConfig(cb_size, {{0, tt::DataFormat::Float32}}).set_page_size(0, cb_size);
    }

    // Helper: Read and validate RTA/CRTA results
    void ValidateArgResults(
        IDevice* device,
        const CoreCoord& core,
        uint32_t cb_addr,
        const std::vector<uint32_t>& expected_rtas,
        const std::vector<uint32_t>& expected_crtas) {
        const uint32_t total_size = (2 + expected_rtas.size() + expected_crtas.size()) * sizeof(uint32_t);
        std::vector<uint32_t> read_result;

        tt::tt_metal::detail::ReadFromDeviceL1(device, core, cb_addr, total_size, read_result);

        // Validate counts
        EXPECT_EQ(read_result[0], expected_rtas.size());
        EXPECT_EQ(read_result[1], expected_crtas.size());

        // Validate RTA payload
        for (uint32_t i = 0; i < expected_rtas.size(); i++) {
            EXPECT_EQ(read_result[i + 2], expected_rtas[i]);
        }

        // Validate CRTA payload
        for (uint32_t i = 0; i < expected_crtas.size(); i++) {
            EXPECT_EQ(read_result[i + expected_rtas.size() + 2], expected_crtas[i]);
        }
    }

    // Helper: Wait for watcher exception with specific message
    void ExpectWatcherException(
        const std::string& expected_message, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::string exception;
        auto start = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - start < timeout) {
            exception = MetalContext::instance().watcher_server()->exception_message();
            if (!exception.empty()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ASSERT_FALSE(exception.empty()) << "Timeout (" << timeout.count() << "ms) waiting for watcher exception.\n"
                                        << "Expected: " << expected_message;

        EXPECT_TRUE(exception.find(expected_message) != std::string::npos)
            << "Watcher exception mismatch:\n"
            << "  Expected substring: " << expected_message << "\n"
            << "  Actual exception:   " << exception;
    }
};

// Test parameters structure
struct RTAAssertTestParams {
    std::string test_name;                  // For readable test names
    bool test_rta;                          // true = test RTA, false = test CRTA
    std::string expected_message;           // Expected watcher exception message
    HalProcessorClassType processor_class;  // DM (BRISC) or COMPUTE (TRISC0)
};

// Parameterized test fixture for RTA/CRTA out-of-bounds assertions
class RTAAssertTest : public RTATestFixture, public ::testing::WithParamInterface<RTAAssertTestParams> {};

// Verifies watcher handles 0xFFFF sentinel pattern (used when no RTAs set):
// - Device interprets 0xBEEF#### as rta_count = 0 (validated on BRISC/TRISC0)
// - Kernels not accessing args run successfully with sentinel pattern
// - Kernels accessing args when count = 0 trigger "out of bounds" assert
// - Catches bug: kernel placed on cores but SetRuntimeArgs() only called for subset
TEST_F(RTATestFixture, SentinelPatternHandlingAndMissingRTADetection) {
    if (IsSlowDispatch()) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    // Part A: Zero-arg kernels should see rta_count = 0, crta_count = 0
    {
        CoreRange core_range(CoreCoord(0, 0), CoreCoord(2, 2));
        CoreRangeSet core_range_set(std::vector{core_range});

        const uint32_t total_read_size = 2 * sizeof(uint32_t);
        const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
        uint32_t cb_size = tt::align(total_read_size, l1_alignment);
        CircularBufferConfig cb0(cb_size, {{tt::CBIndex::c_0, tt::DataFormat::Float32}});
        cb0.set_page_size(tt::CBIndex::c_0, cb_size);
        // For TRISC0 writeback
        uint32_t compute_scratch_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);

        distributed::MeshWorkload workload;
        Program program;
        CBHandle cb0_handle = CreateCircularBuffer(program, core_range_set, cb0);

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
            core_range_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
            core_range_set,
            ComputeConfig{.compile_args = {compute_scratch_addr}});

        // No SetRuntimeArgs called - all cores have RTA/CRTA offset = 0xFFFF pattern
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device, workload);

        auto& program_from_workload = workload.get_programs().at(device_range);
        std::vector<uint32_t> read_result;

        for (const auto& core : core_range) {
            // BRISC: verify rta_count = 0, crta_count = 0
            read_result.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(
                device,
                core,
                program_from_workload.impl().get_circular_buffer(cb0_handle)->address(),
                total_read_size,
                read_result);
            EXPECT_EQ(read_result[0], 0);  // rta_count
            EXPECT_EQ(read_result[1], 0);  // crta_count

            // TRISC0: verify rta_count = 0, crta_count = 0
            read_result.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(device, core, compute_scratch_addr, total_read_size, read_result);
            EXPECT_EQ(read_result[0], 0);  // rta_count
            EXPECT_EQ(read_result[1], 0);  // crta_count
        }
    }

    // Part B: Accessing unset RTAs on a subset of cores should trigger assert
    {
        CoreRange cores_with_rtas(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange cores_without_rtas(CoreCoord(2, 2), CoreCoord(3, 3));
        CoreRangeSet core_range_set(std::vector{cores_with_rtas, cores_without_rtas});

        CircularBufferConfig cb_config = CreateArgCBConfig(default_rtas.size());

        distributed::MeshWorkload workload;
        Program program;
        CreateCircularBuffer(program, core_range_set, cb_config);

        auto kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
            core_range_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = {{"MAX_RTA_IDX", "0"}}});  // Kernel will try to read RTA[0]

        SetRuntimeArgs(program, kernel, cores_with_rtas, default_rtas);
        // cores_without_rtas has NO RTAs set - 0xBEEF#### pattern -> rta_count = 0 -> assert

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device, workload);
        ExpectWatcherException("unique runtime arg index out of bounds");
    }
}

// Correct RTA/CRTA Dispatch and Payload Validation
// Validates happy path: RTAs/CRTAs are correctly dispatched and read back
// Tests multiple core ranges with different RTA sizes, common RTAs shared across
// cores, and re-dispatching on subsequent runs. Ensures arg counts and payload
// values match what was set via SetRuntimeArgs/SetCommonRuntimeArgs
TEST_F(RTATestFixture, CorrectArgDispatchAndPayloadValidation) {
    // First run the program with the initial runtime args
    CoreRange core_range1(CoreCoord(0, 0), CoreCoord(1, 1));
    CoreRange core_range2(CoreCoord(2, 2), CoreCoord(3, 3));
    CoreRangeSet core_range_set(std::vector{core_range1, core_range2});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    // Configure CB to store read-back args
    const uint32_t total_read_size = 2 + default_rtas.size() + default_crtas.size();
    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    CircularBufferConfig cb_config = CreateArgCBConfig(total_read_size);

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    // No out-of-bounds testing here: that's covered by RTAAssertTest
    // Kernel will read back all RTAs/CRTAs for per-core validation
    std::map<std::string, std::string> defines = {};

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    // Range 1: 5 RTAs per core
    SetRuntimeArgs(program, kernel, core_range1, default_rtas);

    // Range 2: 3 RTAs per core (different size than core_range1)
    std::vector<uint32_t> rtas_range2 = {0x1000, 0x1001, 0x1002};
    SetRuntimeArgs(program, kernel, core_range2, rtas_range2);

    // Common args
    SetCommonRuntimeArgs(program, kernel, default_crtas);
    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device, workload);

    // Validate first run for both core ranges
    for (const auto& core : core_range1) {
        ValidateArgResults(device, core, cb_addr, default_rtas, default_crtas);
    }

    for (const auto& core : core_range2) {
        ValidateArgResults(device, core, cb_addr, rtas_range2, default_crtas);
    }

    // Second run: call SetRuntimeArgs again. This tests the case when we're memcpying new data
    // directly into the command issue queue with the arg count
    auto& program_from_workload = workload.get_programs().at(device_range);
    SetRuntimeArgs(program_from_workload, kernel, core_range1, default_rtas);
    SetRuntimeArgs(program_from_workload, kernel, core_range2, rtas_range2);
    RunProgram(mesh_device, workload);

    // Validate second run for both core ranges
    for (const auto& core : core_range1) {
        ValidateArgResults(device, core, cb_addr, default_rtas, default_crtas);
    }
    for (const auto& core : core_range2) {
        ValidateArgResults(device, core, cb_addr, rtas_range2, default_crtas);
    }
}

// Parameterized test: Out-of-Bounds Arg Access Detection
// Verifies watcher detects kernels accessing args beyond bounds (index >= count)
// Tests RTA/CRTA access on both BRISC and TRISC0
TEST_P(RTAAssertTest, OutOfBoundsArgAccessDetection) {
    if (IsSlowDispatch()) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }
    const auto& params = GetParam();

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    const uint32_t total_read_size = 2 + default_rtas.size() + default_crtas.size();
    CircularBufferConfig cb_config = CreateArgCBConfig(total_read_size);

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    std::map<std::string, std::string> defines;
    if (params.test_rta) {
        defines["MAX_RTA_IDX"] = std::to_string(default_rtas.size());  // Out of bounds
    } else {
        defines["MAX_CRTA_IDX"] = std::to_string(default_crtas.size());  // Out of bounds
    }

    // Create kernel based on processor class
    KernelHandle kernel;
    switch (params.processor_class) {
        case HalProcessorClassType::DM: {
            kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
                core_range_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});
        } break;
        case HalProcessorClassType::COMPUTE: {
            // For compute kernel, pass scratch address as compile arg
            uint32_t compute_scratch_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
            kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
                core_range_set,
                ComputeConfig{.compile_args = {compute_scratch_addr}, .defines = defines});
        } break;
        default: TT_THROW("Unsupported processor class");
    }

    if (params.test_rta) {
        SetRuntimeArgs(program, kernel, core_range, default_rtas);
    } else {
        SetCommonRuntimeArgs(program, kernel, default_crtas);
    }
    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device, workload);

    ExpectWatcherException(params.expected_message);
}

INSTANTIATE_TEST_SUITE_P(
    WatcherArgAsserts,
    RTAAssertTest,
    ::testing::Values(
        // RTA tests on BRISC
        RTAAssertTestParams{"RTA_RISCV0", true, "unique runtime arg index out of bounds", HalProcessorClassType::DM},
        // RTA tests on TRISC0
        RTAAssertTestParams{
            "RTA_TRISC0", true, "unique runtime arg index out of bounds", HalProcessorClassType::COMPUTE},
        // CRTA tests on BRISC
        RTAAssertTestParams{"CRTA_RISCV0", false, "common runtime arg index out of bounds", HalProcessorClassType::DM},
        // CRTA tests on TRISC0
        RTAAssertTestParams{
            "CRTA_TRISC0", false, "common runtime arg index out of bounds", HalProcessorClassType::COMPUTE}),
    [](const ::testing::TestParamInfo<RTAAssertTestParams>& info) { return info.param.test_name; });
}  // namespace tt::tt_metal
