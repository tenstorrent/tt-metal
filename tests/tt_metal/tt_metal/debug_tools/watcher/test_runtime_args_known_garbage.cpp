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

// Test if RTA payload region initialized by HostMemDeviceCommand
// is filled with known garbage (with 0xBEEF####) for cores with unset RTAs
// This test is useful only when watcher asserts are disabled
TEST_F(MeshWatcherFixture, WatcherKnownGarbageRTAs) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_enabled = !tt::tt_metal::MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_enabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are disabled";
    }

    // First run the program with the initial runtime args
    CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
    CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const std::vector<uint32_t> rtas = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};

    // Configure CB to store read-back args
    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t cb_size = tt::align(rtas.size() * sizeof(uint32_t), l1_alignment);
    CircularBufferConfig cb_config =
        CircularBufferConfig(cb_size, {{0, tt::DataFormat::Float32}}).set_page_size(0, cb_size);

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_runtime_args_prefill.cpp",
        core_range_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // SetRuntimeArgs only for first_core_range ; second_core_range has unset RTAs and should read known garbage back
    SetRuntimeArgs(program, kernel, first_core_range, rtas);

    workload.add_program(device_range, std::move(program));
    EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false));
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<uint32_t> read_result;
    // Verify each core in first_core_range (valid data)
    for (const auto& core : first_core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, cb_addr, rtas.size() * sizeof(uint32_t), read_result);
        EXPECT_EQ(read_result.size(), rtas.size());
        for (uint32_t i = 0; i < read_result.size(); i++) {
            EXPECT_EQ(read_result[i], rtas[i]);
        }
    }

    // Verify each core in second_core_range (expect known garbage: 0xBEEF####)
    for (const auto& core : second_core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, cb_addr, rtas.size() * sizeof(uint32_t), read_result);
        EXPECT_EQ(read_result.size(), rtas.size());
        for (const auto& result : read_result) {
            EXPECT_EQ(result & 0xFFFF0000, 0xBEEF0000);
        }
    }
}

// This test reads back RTA and CRTA counts and verifies all RTA and CRTAs payload as dispatched.
// There should be no watcher asserts here as MAX_RTA_IDX/MAX_CRTA_IDX accessed on device side kernel
// are within the allocated RTA/CRTA bounds
TEST_F(MeshWatcherFixture, WatcherArgCountCheck) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_disabled = tt::tt_metal::MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_disabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are enabled";
    }

    // First run the program with the initial runtime args
    CoreRange core_range(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const std::vector<uint32_t> rtas = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};
    const std::vector<uint32_t> crtas = {0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777};
    const uint32_t total_read_size = (2 + rtas.size() + crtas.size()) * sizeof(uint32_t);

    // Configure CB to store read-back args
    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t cb_size = tt::align(total_read_size, l1_alignment);
    CircularBufferConfig cb_config =
        CircularBufferConfig(cb_size, {{0, tt::DataFormat::Float32}}).set_page_size(0, cb_size);

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    // The max index set to access is within bounds of RTA and CRTA
    const uint32_t max_rta_idx = rtas.size() - 1;
    const uint32_t max_crta_idx = crtas.size() - 1;
    std::map<std::string, std::string> defines = {
        {"MAX_RTA_IDX", std::to_string(max_rta_idx)}, {"MAX_CRTA_IDX", std::to_string(max_crta_idx)}};

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    SetRuntimeArgs(program, kernel, core_range, rtas);
    SetCommonRuntimeArgs(program, kernel, crtas);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<uint32_t> read_result;
    for (const auto& core : core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, cb_addr, total_read_size, read_result);
        // First check RTA and CRTA counts match as expected
        EXPECT_EQ(read_result[0], rtas.size());
        EXPECT_EQ(read_result[1], crtas.size());
        // Second check if the RTA and CRTA payloads match as expected
        for (uint32_t i = 0; i < rtas.size(); i++) {
            EXPECT_EQ(read_result[i + 2], rtas[i]);
        }
        for (uint32_t i = 0; i < crtas.size(); i++) {
            EXPECT_EQ(read_result[i + rtas.size() + 2], crtas[i]);
        }
    }
}

// This test sets MAX_RTA_IDX == size of RTA payload. This should trigger dev_msgs::DebugAssertRtaOutOfBounds
// as we're accessing an index beyond the dispatched RTA bounds
TEST_F(MeshWatcherFixture, WatcherRTACountAsserts) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_disabled = tt::tt_metal::MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_disabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are enabled";
    }

    // First run the program with the initial runtime args
    CoreRange core_range(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const std::vector<uint32_t> rtas = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};
    const std::vector<uint32_t> crtas = {0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777};
    const uint32_t total_read_size = (2 + rtas.size() + crtas.size()) * sizeof(uint32_t);

    // Configure CB to store read-back args
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t cb_size = tt::align(total_read_size, l1_alignment);
    CircularBufferConfig cb_config =
        CircularBufferConfig(cb_size, {{0, tt::DataFormat::Float32}}).set_page_size(0, cb_size);

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    // The max index to access is within bounds of RTA
    const uint32_t max_rta_idx = rtas.size();
    std::map<std::string, std::string> defines = {{"MAX_RTA_IDX", std::to_string(max_rta_idx)}};

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    SetRuntimeArgs(program, kernel, core_range, rtas);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    std::string exception;
    while (exception.empty()) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(exception.find("unique runtime arg index out of bounds") != std::string::npos) << exception;

    auto& base_cq = mesh_device->mesh_command_queue();
    auto* fd_cq = dynamic_cast<tt::tt_metal::distributed::FDMeshCommandQueue*>(&base_cq);
    if (fd_cq != nullptr) {
        tt::tt_metal::tt_dispatch_tests::Common::FDMeshCQTestAccessor::force_abort_for_watcher_kill(*fd_cq);
    }
}

// This test sets MAX_CRTA_IDX == size of CRTA payload. This should trigger dev_msgs::DebugAssertCrtaOutOfBounds
// as we're accessing an index out of dispatched CRTA bounds
TEST_F(MeshWatcherFixture, WatcherCRTACountAsserts) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_disabled = tt::tt_metal::MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_disabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are enabled";
    }

    // First run the program with the initial runtime args
    CoreRange core_range(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const std::vector<uint32_t> rtas = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};
    const std::vector<uint32_t> crtas = {0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777};
    const uint32_t total_read_size = (2 + rtas.size() + crtas.size()) * sizeof(uint32_t);

    // Configure CB to store read-back args
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t cb_size = tt::align(total_read_size, l1_alignment);
    CircularBufferConfig cb_config =
        CircularBufferConfig(cb_size, {{0, tt::DataFormat::Float32}}).set_page_size(0, cb_size);

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    // The max index to access is within bounds of CRTA
    const uint32_t max_crta_idx = crtas.size();
    std::map<std::string, std::string> defines = {{"MAX_CRTA_IDX", std::to_string(max_crta_idx)}};

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    SetCommonRuntimeArgs(program, kernel, crtas);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    std::string exception;
    while (exception.empty()) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(exception.find("common runtime arg index out of bounds") != std::string::npos) << exception;

    auto& base_cq = mesh_device->mesh_command_queue();
    auto* fd_cq = dynamic_cast<tt::tt_metal::distributed::FDMeshCommandQueue*>(&base_cq);
    if (fd_cq != nullptr) {
        tt::tt_metal::tt_dispatch_tests::Common::FDMeshCQTestAccessor::force_abort_for_watcher_kill(*fd_cq);
    }
}

// In this test no RTA or CRTA are set, so counts read back should be zero
// This is an edge case since the dispatcher doesn't dispatch anything, but
// Tests run on RISCV_0 and TRISC0
TEST_F(MeshWatcherFixture, WatcherZeroArgCheck) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    bool watcher_assert_disabled = tt::tt_metal::MetalContext::instance().rtoptions().watcher_assert_disabled();
    if (watcher_assert_disabled) {
        GTEST_SKIP() << "This test can only be run when watcher assert checks (RTA/CRTA) are enabled";
    }

    // First run the program with the initial runtime args
    CoreRange core_range(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const uint32_t total_read_size = (2) * sizeof(uint32_t);

    // Configure CB to store read-back args
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
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
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

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
        // RISCV_0: rta_count and crta_count should be set to 0
        EXPECT_EQ(read_result[0], 0);
        EXPECT_EQ(read_result[1], 0);

        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, compute_scratch_addr, total_read_size, read_result);
        // TRISC0: rta_count and crta_count should be set to 0
        EXPECT_EQ(read_result[0], 0);
        EXPECT_EQ(read_result[1], 0);
    }
}
}  // namespace tt::tt_metal
