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

#include "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h"
#include "debug_tools_fixture.hpp"
#include "impl/buffers/circular_buffer.hpp"

namespace tt::tt_metal {

// Fixture for RTA/CRTA watcher bounds check tests
class RTATestFixture : public MeshWatcherFixture {
protected:
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
    void ExpectWatcherException(const std::string& expected_message) {
        std::string exception;
        while (exception.empty()) {
            exception = MetalContext::instance().watcher_server()->exception_message();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        EXPECT_TRUE(exception.find(expected_message) != std::string::npos) << exception;

        // Force abort for watcher kill
        auto& base_cq = devices_[0]->mesh_command_queue();
        auto* fd_cq = dynamic_cast<tt::tt_metal::distributed::FDMeshCommandQueue*>(&base_cq);
        if (fd_cq != nullptr) {
            tt::tt_metal::tt_dispatch_tests::Common::FDMeshCQTestAccessor::force_abort_for_watcher_kill(*fd_cq);
        }
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

// Test if RTA payload region initialized by HostMemDeviceCommand
// is filled with known garbage (with 0xBEEF####) for cores with unset RTAs
// This test is useful only when watcher asserts are disabled
TEST_F(RTATestFixture, WatcherKnownGarbageRTAs) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_enabled = !MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_enabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are disabled";
    }

    // First run the program with the initial runtime args
    CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
    CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    // Configure CB to store read-back args
    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    CircularBufferConfig cb_config = CreateArgCBConfig(default_rtas.size());

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_runtime_args_prefill.cpp",
        core_range_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // SetRuntimeArgs only for first_core_range ; second_core_range has unset RTAs and should read known garbage back
    SetRuntimeArgs(program, kernel, first_core_range, default_rtas);

    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device, workload);

    std::vector<uint32_t> read_result;
    // Verify each core in first_core_range (valid data)
    for (const auto& core : first_core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, core, cb_addr, default_rtas.size() * sizeof(uint32_t), read_result);
        EXPECT_EQ(read_result.size(), default_rtas.size());
        for (uint32_t i = 0; i < read_result.size(); i++) {
            EXPECT_EQ(read_result[i], default_rtas[i]);
        }
    }

    // Verify each core in second_core_range (expect known garbage: 0xBEEF####)
    for (const auto& core : second_core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, core, cb_addr, default_rtas.size() * sizeof(uint32_t), read_result);
        EXPECT_EQ(read_result.size(), default_rtas.size());
        for (const auto& result : read_result) {
            EXPECT_EQ(result & 0xFFFF0000, 0xBEEF0000);
        }
    }
}

// This test reads back RTA and CRTA counts and verifies all RTA and CRTAs payload as dispatched.
// There should be no watcher asserts here as MAX_RTA_IDX/MAX_CRTA_IDX accessed on device side kernel
// are within the allocated RTA/CRTA bounds
TEST_F(RTATestFixture, WatcherArgCountCheck) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_disabled = MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_disabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are enabled";
    }

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

// This test sets MAX_RTA_IDX or MAX_CRTA_IDX equal to payload size
// This should trigger out-of-bounds watcher asserts
// This tests only run on BRISC and TRISC0
TEST_P(RTAAssertTest, WatcherArgOutOfBoundsAssert) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_disabled = MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_disabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are enabled";
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
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    ExpectWatcherException(params.expected_message);
}

// In this test no RTA or CRTA are set, so counts read back should be zero
// This tests an edge case: the dispatcher doesn't dispatch anything when arg count  = 0,
// but RTA and CRTA offsets are initialized to 0xBEEF for device to interpret it as no arg case (no payload)
// This tests only run on BRISC and TRISC0
TEST_F(RTATestFixture, WatcherZeroArgCheck) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_disabled = MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_disabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are enabled";
    }

    // First run the program with the initial runtime args
    CoreRange core_range(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    const uint32_t total_read_size = 2 * sizeof(uint32_t);

    // Configure CB to store read-back args
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t cb_size = tt::align(total_read_size, l1_alignment);
    CircularBufferConfig cb0(cb_size, {{tt::CBIndex::c_0, tt::DataFormat::Float32}});
    cb0.set_page_size(tt::CBIndex::c_0, cb_size);
    // Use this to write back from TRISC0 (compute kernel)
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

    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device, workload);

    auto& program_from_workload = workload.get_programs().at(device_range);

    std::vector<uint32_t> read_result;
    for (const auto& core : core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(
            device,
            core,
            program_from_workload.impl().get_circular_buffer(cb0_handle)->address(),
            total_read_size,
            read_result);
        // BRISC: rta_count and crta_count should be set to 0
        EXPECT_EQ(read_result[0], 0);
        EXPECT_EQ(read_result[1], 0);

        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, compute_scratch_addr, total_read_size, read_result);
        // TRISC0: rta_count and crta_count should be set to 0
        EXPECT_EQ(read_result[0], 0);
        EXPECT_EQ(read_result[1], 0);
    }
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
        // RTA tests on TRISC0
        RTAAssertTestParams{
            "CRTA_TRISC0", false, "common runtime arg index out of bounds", HalProcessorClassType::COMPUTE}),
    [](const ::testing::TestParamInfo<RTAAssertTestParams>& info) { return info.param.test_name; });
}  // namespace tt::tt_metal
