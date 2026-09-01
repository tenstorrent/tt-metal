// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/global_circular_buffer.hpp>  // for the RemoteCircularBuffer scoped-lock tests below
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"

// Metal 2.0 host API (DFB tests)
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>
#include <vector>

#include "noc_debugging_fixture.hpp"

namespace tt::tt_metal {

// Test two cores: one locks and writes, another writes to the same region
// Both kernels synchronize using semaphores at start and end to ensure
// locks are held concurrently for the profiler to capture overlapping accesses
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 2) {
            GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
        }

        const CoreCoord locker_core = {0, 0};
        const CoreCoord writer_core = {1, 0};
        Program program = CreateProgram();
        distributed::MeshWorkload workload;

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);

        uint32_t locker_buffer_addr = unreserved_addr;
        uint32_t writer_buffer_addr = unreserved_addr + (alignment * 32);
        uint32_t num_elements = 8;

        auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);

        uint32_t locker_sem_id = CreateSemaphore(program, locker_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        std::vector<uint32_t> locker_args = {
            locker_buffer_addr,
            num_elements,
            locker_sem_id,
            writer_sem_id,
            writer_virtual_core.x,
            writer_virtual_core.y};

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, locker_kernel, locker_core, locker_args);

        std::vector<uint32_t> writer_args = {
            writer_buffer_addr,
            num_elements,
            locker_virtual_core.x,
            locker_virtual_core.y,
            locker_buffer_addr,
            writer_sem_id,
            locker_sem_id,
            locker_virtual_core.x,
            locker_virtual_core.y};

        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, writer_kernel, writer_core, writer_args);

        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());

        ReadMeshDeviceProfilerResults(*mesh_device);

        // Writer core (source of the NOC writes) should have been flagged for writing to a locked buffer
        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = this->get_write_to_locked_issues(device->id(), writer_virtual_core, 0);
            locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
        }
        ASSERT_FALSE(locked_issues.empty())
            << "Expected write-to-locked-buffer issue on writer core (1,0); NOC debug did not report the violation.";

        uint32_t expected_write_size = num_elements * sizeof(uint32_t);
        for (const auto& issue : locked_issues) {
            EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM);
            EXPECT_EQ(issue.issue_address, locker_buffer_addr);
            EXPECT_EQ(issue.issue_size, expected_write_size);
            EXPECT_EQ(issue.src_x, writer_virtual_core.x);
            EXPECT_EQ(issue.src_y, writer_virtual_core.y);
            EXPECT_EQ(issue.dst_x, locker_virtual_core.x);
            EXPECT_EQ(issue.dst_y, locker_virtual_core.y);
        }
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockMultipleL1Issues) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 2) {
            GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
        }

        const CoreCoord locker_core = {0, 0};
        const CoreCoord writer_core = {1, 0};
        Program program = CreateProgram();
        distributed::MeshWorkload workload;

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);

        uint32_t buffer_addr_a = unreserved_addr;
        uint32_t num_elements_a = 8;
        uint32_t buffer_addr_b = unreserved_addr + (alignment * 16);
        uint32_t num_elements_b = 16;
        uint32_t writer_buffer_addr = unreserved_addr + (alignment * 48);

        auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);

        uint32_t locker_sem_id = CreateSemaphore(program, locker_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_multi.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel_multi.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(
            program,
            locker_kernel,
            locker_core,
            {buffer_addr_a,
             num_elements_a,
             buffer_addr_b,
             num_elements_b,
             locker_sem_id,
             writer_sem_id,
             writer_virtual_core.x,
             writer_virtual_core.y});

        uint32_t write_size_a = num_elements_a * sizeof(uint32_t);
        uint32_t write_size_b = num_elements_b * sizeof(uint32_t);
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {writer_buffer_addr,
             write_size_a,
             write_size_b,
             locker_virtual_core.x,
             locker_virtual_core.y,
             buffer_addr_a,
             buffer_addr_b,
             writer_sem_id,
             locker_sem_id,
             locker_virtual_core.x,
             locker_virtual_core.y});

        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());

        ReadMeshDeviceProfilerResults(*mesh_device);

        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = this->get_write_to_locked_issues(device->id(), writer_virtual_core, 0);
            locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
        }
        ASSERT_GE(locked_issues.size(), 2u)
            << "Expected at least 2 write-to-locked-buffer issues (one per locked region)";

        bool found_issue_a = false;
        bool found_issue_b = false;
        for (const auto& issue : locked_issues) {
            EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM);
            EXPECT_EQ(issue.src_x, writer_virtual_core.x);
            EXPECT_EQ(issue.src_y, writer_virtual_core.y);
            EXPECT_EQ(issue.dst_x, locker_virtual_core.x);
            EXPECT_EQ(issue.dst_y, locker_virtual_core.y);

            if (issue.issue_address == buffer_addr_a && issue.issue_size == write_size_a) {
                found_issue_a = true;
            }
            if (issue.issue_address == buffer_addr_b && issue.issue_size == write_size_b) {
                found_issue_b = true;
            }
        }
        EXPECT_TRUE(found_issue_a) << "Missing write-to-locked issue for buffer A at addr 0x" << std::hex
                                   << buffer_addr_a;
        EXPECT_TRUE(found_issue_b) << "Missing write-to-locked issue for buffer B at addr 0x" << std::hex
                                   << buffer_addr_b;
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessNoIssue) {
    // inverted version of the test above
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 2) {
            GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
        }

        const CoreCoord locker_core = {0, 0};
        const CoreCoord writer_core = {1, 0};
        Program program = CreateProgram();
        distributed::MeshWorkload workload;

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);

        uint32_t locker_buffer_addr = unreserved_addr;
        uint32_t writer_buffer_addr = unreserved_addr + (alignment * 32);
        uint32_t num_elements = 8;

        auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);

        uint32_t locker_sem_id = CreateSemaphore(program, locker_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        std::vector<uint32_t> locker_args = {
            locker_buffer_addr,
            num_elements,
            locker_sem_id,
            writer_sem_id,
            writer_virtual_core.x,
            writer_virtual_core.y};

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_no_issue.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, locker_kernel, locker_core, locker_args);

        std::vector<uint32_t> writer_args = {
            writer_buffer_addr,
            num_elements,
            locker_virtual_core.x,
            locker_virtual_core.y,
            locker_buffer_addr,
            writer_sem_id,
            locker_sem_id,
            locker_virtual_core.x,
            locker_virtual_core.y};

        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel_no_issue.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, writer_kernel, writer_core, writer_args);

        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());

        ReadMeshDeviceProfilerResults(*mesh_device);

        // No write-to-locked issues should be reported (writes happen only when buffer is not locked)
        for (IDevice* device : mesh_device->get_devices()) {
            ChipId chip_id = device->id();
            EXPECT_FALSE(this->has_write_to_locked_issue(chip_id, writer_virtual_core, 0))
                << "Unexpected write-to-locked-buffer issue on writer core; writes were outside lock scope.";
            EXPECT_FALSE(this->has_write_to_locked_issue(chip_id, locker_virtual_core, 0))
                << "Unexpected write-to-locked-buffer issue on locker core.";
        }
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessCBIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 2) {
            GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
        }

        const CoreCoord locker_core = {0, 0};
        const CoreCoord writer_core = {1, 0};
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        workload.add_program(device_range, CreateProgram());

        Program& program = workload.get_programs().at(device_range);
        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
        uint32_t writer_buffer_addr = unreserved_addr + (alignment * 32);

        constexpr uint8_t cb_buffer_index = 0;
        uint32_t cb_page_size = 32;
        uint32_t cb_total_size = cb_page_size * 2;
        CircularBufferConfig cb_config =
            CircularBufferConfig(cb_total_size, {{cb_buffer_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_buffer_index, cb_page_size);
        CreateCircularBuffer(program, locker_core, cb_config);

        auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);
        uint32_t locker_sem_id = CreateSemaphore(program, locker_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        uint32_t write_size = alignment;
        // The locker stages its locked CB base here and NOCs it to the same offset on the writer,
        // so the writer can target the locked region directly.
        uint32_t scratch_addr = unreserved_addr + 0x20000;

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_cb.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(
            program,
            locker_kernel,
            locker_core,
            {static_cast<uint32_t>(cb_buffer_index),
             locker_sem_id,
             writer_sem_id,
             writer_virtual_core.x,
             writer_virtual_core.y,
             scratch_addr,
             scratch_addr});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {writer_buffer_addr,
             write_size,
             locker_virtual_core.x,
             locker_virtual_core.y,
             scratch_addr,
             writer_sem_id,
             locker_sem_id,
             locker_virtual_core.x,
             locker_virtual_core.y});

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        ReadMeshDeviceProfilerResults(*mesh_device);

        std::vector<uint32_t> published;
        detail::ReadFromDeviceL1(mesh_device->get_devices()[0], locker_core, scratch_addr, sizeof(uint32_t), published);
        ASSERT_FALSE(published.empty());
        const uint32_t cb_base = published[0];
        ASSERT_GT(cb_base, 0u) << "locker did not publish its locked CB base";

        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = this->get_write_to_locked_issues(device->id(), writer_virtual_core, 0);
            locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
        }
        ASSERT_FALSE(locked_issues.empty())
            << "Expected write-to-locked-CB issue on writer core; NOC debug did not report the violation.";

        for (const auto& issue : locked_issues) {
            EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB);
            EXPECT_GE(issue.issue_size, write_size);
            EXPECT_EQ(issue.issue_address, cb_base);
            EXPECT_EQ(issue.src_x, writer_virtual_core.x);
            EXPECT_EQ(issue.src_y, writer_virtual_core.y);
            EXPECT_EQ(issue.dst_x, locker_virtual_core.x);
            EXPECT_EQ(issue.dst_y, locker_virtual_core.y);
        }
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessCBNoIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 2) {
            GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
        }

        const CoreCoord locker_core = {0, 0};
        const CoreCoord writer_core = {1, 0};
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        workload.add_program(device_range, CreateProgram());

        Program& program = workload.get_programs().at(device_range);
        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
        uint32_t writer_buffer_addr = unreserved_addr + (alignment * 32);

        constexpr uint8_t cb_buffer_index = 0;
        uint32_t cb_page_size = 32;
        uint32_t cb_total_size = cb_page_size * 2;
        CircularBufferConfig cb_config =
            CircularBufferConfig(cb_total_size, {{cb_buffer_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_buffer_index, cb_page_size);
        CreateCircularBuffer(program, locker_core, cb_config);

        auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);
        uint32_t locker_sem_id = CreateSemaphore(program, locker_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        uint32_t write_size = alignment;
        uint32_t scratch_addr = unreserved_addr + 0x20000;

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_cb_no_issue.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(
            program,
            locker_kernel,
            locker_core,
            {static_cast<uint32_t>(cb_buffer_index),
             locker_sem_id,
             writer_sem_id,
             writer_virtual_core.x,
             writer_virtual_core.y,
             scratch_addr,
             scratch_addr});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {writer_buffer_addr,
             write_size,
             locker_virtual_core.x,
             locker_virtual_core.y,
             scratch_addr,
             writer_sem_id,
             locker_sem_id,
             locker_virtual_core.x,
             locker_virtual_core.y});

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        ReadMeshDeviceProfilerResults(*mesh_device);

        for (IDevice* device : mesh_device->get_devices()) {
            ChipId chip_id = device->id();
            EXPECT_FALSE(this->has_write_to_locked_issue(chip_id, writer_virtual_core, 0))
                << "Unexpected write-to-locked-CB issue on writer core; writes were outside lock scope.";
            EXPECT_FALSE(this->has_write_to_locked_issue(chip_id, locker_virtual_core, 0))
                << "Unexpected write-to-locked-CB issue on locker core.";
        }
    }
}

// Writing to a region locked by yourself should not trigger an issue.
TEST_F(NOCDebuggingFixture, ScopedLockSelfWriteToOwnLockNoIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());

        const CoreCoord core = {0, 0};
        Program program = CreateProgram();
        distributed::MeshWorkload workload;

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);

        uint32_t lock_addr = unreserved_addr;
        uint32_t num_elements = 8;
        uint32_t src_buffer_addr = unreserved_addr + (alignment * 32);
        uint32_t write_target_addr = lock_addr;
        uint32_t write_size = num_elements * sizeof(uint32_t);

        auto virtual_core = mesh_device->worker_core_from_logical_core(core);

        KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_self_write_kernel.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(
            program,
            kernel,
            core,
            {lock_addr, num_elements, src_buffer_addr, write_target_addr, write_size, virtual_core.x, virtual_core.y});

        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());

        ReadMeshDeviceProfilerResults(*mesh_device);

        for (IDevice* device : mesh_device->get_devices()) {
            EXPECT_FALSE(this->has_write_to_locked_issue(device->id(), virtual_core, 0))
                << "Writing into your OWN locked region is legitimate (ownership-aware); must not flag.";
        }
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockSelfWriteToUnlockedNoIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());

        const CoreCoord core = {0, 0};
        Program program = CreateProgram();
        distributed::MeshWorkload workload;

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);

        uint32_t lock_addr = unreserved_addr;
        uint32_t num_elements = 8;
        uint32_t src_buffer_addr = unreserved_addr + (alignment * 32);
        uint32_t write_target_addr = unreserved_addr + (alignment * 16);
        uint32_t write_size = num_elements * sizeof(uint32_t);

        auto virtual_core = mesh_device->worker_core_from_logical_core(core);

        KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_self_write_kernel.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(
            program,
            kernel,
            core,
            {lock_addr, num_elements, src_buffer_addr, write_target_addr, write_size, virtual_core.x, virtual_core.y});

        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());

        ReadMeshDeviceProfilerResults(*mesh_device);

        for (IDevice* device : mesh_device->get_devices()) {
            EXPECT_FALSE(this->has_write_to_locked_issue(device->id(), virtual_core, 0))
                << "Unexpected write-to-locked issue; NOC write targeted an unlocked region.";
        }
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockNoWritesNoIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());

        const CoreCoord core = {0, 0};
        Program program = CreateProgram();
        distributed::MeshWorkload workload;

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

        uint32_t lock_addr = unreserved_addr;
        uint32_t num_elements = 8;

        auto virtual_core = mesh_device->worker_core_from_logical_core(core);

        KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_only_kernel.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, kernel, core, {lock_addr, num_elements});

        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());

        ReadMeshDeviceProfilerResults(*mesh_device);

        for (IDevice* device : mesh_device->get_devices()) {
            EXPECT_FALSE(this->has_write_to_locked_issue(device->id(), virtual_core, 0))
                << "Unexpected write-to-locked issue; kernel only locked and unlocked with no NOC writes.";
        }
    }
}

namespace {

// None: no issue, wrote to its own locked region or outside the DFB ring
// Locked: wrote to a region another thread locked
// Unlocked: wrote to a region without holding the lock
enum class ExpectedDfbIssue { None, Locked, Unlocked };

void run_dfb_scoped_lock_test(
    NOCDebuggingFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t target_entry_index,
    bool write_after_unlock,
    ExpectedDfbIssue expected,
    bool skip_lock = false,
    NOC producer_noc = NOC::NOC_0,
    DataMovementProcessor producer_processor = DataMovementProcessor::RISCV_0,
    uint32_t publish_ring_base_addr = 0) {
    const experimental::NodeCoord core = {0, 0};
    auto virtual_core = mesh_device->worker_core_from_logical_core(core);

    const bool producer_is_rv0 = (producer_processor == DataMovementProcessor::RISCV_0);
    const int producer_proc_id = producer_is_rv0 ? 0 : 1;
    const DataMovementProcessor consumer_processor =
        producer_is_rv0 ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0;

    auto& mc = MetalContext::instance();
    uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
    uint32_t unreserved_addr =
        mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    uint32_t entry_size = alignment * 2;  // bytes per DFB entry
    uint32_t num_entries = 4;
    uint32_t write_size = alignment;
    uint32_t src_buffer_addr = unreserved_addr + 0x10000;  // producer's own L1 source, clear of the DFB
    uint32_t target_entry_offset = target_entry_index * entry_size;

    const experimental::DFBSpecName DFB_NAME{"lock_dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};

    experimental::DataflowBufferSpec dfb_spec{
        .unique_id = DFB_NAME,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // The two DM kernels claim different NOCs, so the consumer takes whichever one the producer did not.
    const NOC consumer_noc = (producer_noc == NOC::NOC_0) ? NOC::NOC_1 : NOC::NOC_0;
    const experimental::DataMovementHardwareConfig dm_producer_cfg = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = producer_processor, .noc = producer_noc}};
    const experimental::DataMovementHardwareConfig dm_consumer_cfg = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = consumer_processor, .noc = consumer_noc}};

    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_producer.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(DFB_NAME, "out")},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src_buffer_addr",
                  "write_size",
                  "self_noc_x",
                  "self_noc_y",
                  "target_entry_offset",
                  "write_after_unlock",
                  "skip_lock",
                  "publish_ring_base_addr"}},
        .hw_config = dm_producer_cfg,
    };
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_consumer.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(DFB_NAME, "in")},
        .hw_config = dm_consumer_cfg,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = core,
    };
    experimental::ProgramSpec spec{
        .name = "dfb_scoped_lock",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    experimental::ProgramRunArgs::KernelRunArgs producer_params{};
    producer_params.kernel = PRODUCER;
    producer_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        core,
        {{"src_buffer_addr", src_buffer_addr},
         {"write_size", write_size},
         {"self_noc_x", static_cast<uint32_t>(virtual_core.x)},
         {"self_noc_y", static_cast<uint32_t>(virtual_core.y)},
         {"target_entry_offset", target_entry_offset},
         {"write_after_unlock", static_cast<uint32_t>(write_after_unlock)},
         {"skip_lock", static_cast<uint32_t>(skip_lock)},
         {"publish_ring_base_addr", publish_ring_base_addr}});
    experimental::ProgramRunArgs::KernelRunArgs consumer_params{};
    consumer_params.kernel = CONSUMER;  // no runtime args
    run_args.kernel_run_args = {producer_params, consumer_params};
    experimental::SetProgramRunArgs(program, run_args);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    ReadMeshDeviceProfilerResults(*mesh_device);

    if (expected == ExpectedDfbIssue::Locked) {
        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = fixture->get_write_to_locked_issues(device->id(), virtual_core, producer_proc_id);
            locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
        }
        ASSERT_FALSE(locked_issues.empty()) << "Expected WRITE_TO_LOCKED_DFB; NOC debug did not report the violation.";
        for (const auto& issue : locked_issues) {
            EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_DFB);
            EXPECT_GE(issue.issue_size, write_size);  // recorded NOC size may round up past write_size (as in CB)
            EXPECT_GT(issue.issue_address, 0u);       // device-derived (get_write_ptr); exact addr not host-known
            EXPECT_EQ(issue.src_x, virtual_core.x);   // self-write: src == dst == producer core
            EXPECT_EQ(issue.src_y, virtual_core.y);
            EXPECT_EQ(issue.dst_x, virtual_core.x);
            EXPECT_EQ(issue.dst_y, virtual_core.y);
        }
    } else if (expected == ExpectedDfbIssue::Unlocked) {
        std::vector<NOCDebugIssueType> unlocked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = fixture->get_write_to_unlocked_dfb_issues(device->id(), virtual_core, producer_proc_id);
            unlocked_issues.insert(unlocked_issues.end(), issues.begin(), issues.end());
        }
        ASSERT_FALSE(unlocked_issues.empty())
            << "Expected WRITE_TO_UNLOCKED_DFB; a DFB write with no lock held was not flagged.";
        for (const auto& issue : unlocked_issues) {
            EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_UNLOCKED_DFB);
            EXPECT_GE(issue.issue_size, write_size);
            EXPECT_GT(issue.issue_address, 0u);
            EXPECT_EQ(issue.src_x, virtual_core.x);  // self-write: src == dst == producer core
            EXPECT_EQ(issue.src_y, virtual_core.y);
            EXPECT_EQ(issue.dst_x, virtual_core.x);
            EXPECT_EQ(issue.dst_y, virtual_core.y);
        }
    } else {
        for (IDevice* device : mesh_device->get_devices()) {
            EXPECT_FALSE(fixture->has_write_to_locked_issue(device->id(), virtual_core, producer_proc_id))
                << "Unexpected write-to-locked-DFB issue.";
            EXPECT_FALSE(fixture->has_write_to_unlocked_dfb_issue(device->id(), virtual_core, producer_proc_id))
                << "Unexpected write-to-unlocked-DFB issue.";
        }
    }
}

// A DFB's L1 extent must stop being tracked once the kernel that declared it exits, otherwise a later
// program that reuses the same L1 for something else gets false WRITE_TO_UNLOCKED_DFB reports.
void run_dfb_region_cleared_between_launches_test(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const experimental::NodeCoord core = {0, 0};
    auto virtual_core = mesh_device->worker_core_from_logical_core(core);

    auto& mc = MetalContext::instance();
    const uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
    const uint32_t unreserved_addr =
        mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t src_buffer_addr = unreserved_addr + 0x10000;  // matches run_dfb_scoped_lock_test
    const uint32_t publish_addr = unreserved_addr + 0x20000;     // scratch, clear of the ring
    const uint32_t write_size = alignment;

    // Launch 1. skip_lock -> writes entry 1 of its own ring with no lock held, so it must be flagged.
    run_dfb_scoped_lock_test(
        fixture,
        mesh_device,
        /*target_entry_index=*/1,
        /*write_after_unlock=*/false,
        ExpectedDfbIssue::Unlocked,
        /*skip_lock=*/true,
        /*producer_noc=*/NOC::NOC_0,
        /*producer_processor=*/DataMovementProcessor::RISCV_0,
        /*publish_ring_base_addr=*/publish_addr);

    IDevice* device = mesh_device->get_devices()[0];
    std::vector<uint32_t> published;
    detail::ReadFromDeviceL1(device, CoreCoord{core.x, core.y}, publish_addr, sizeof(uint32_t), published);
    ASSERT_FALSE(published.empty());
    const uint32_t ring_base = published[0];
    ASSERT_GT(ring_base, 0u) << "launch 1 did not publish its DFB ring base";

    // Launch 2: a program with NO DFB binding, writing entry 0.
    const experimental::KernelSpecName WRITER{"writer"};
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_l1_writer.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {.runtime_arg_names = {"src_buffer_addr", "write_size", "self_noc_x", "self_noc_y", "target_addr"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_specific =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0}},
    };
    experimental::WorkUnitSpec writer_wu{.name = "main", .kernels = {WRITER}, .target_nodes = core};
    experimental::ProgramSpec writer_program_spec{
        .name = "dfb_free_l1_write", .kernels = {writer_spec}, .work_units = {writer_wu}};

    Program writer_program = experimental::MakeProgramFromSpec(*mesh_device, writer_program_spec);
    experimental::ProgramRunArgs writer_run_args;
    experimental::ProgramRunArgs::KernelRunArgs writer_params{};
    writer_params.kernel = WRITER;
    writer_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        core,
        {{"src_buffer_addr", src_buffer_addr},
         {"write_size", write_size},
         {"self_noc_x", static_cast<uint32_t>(virtual_core.x)},
         {"self_noc_y", static_cast<uint32_t>(virtual_core.y)},
         {"target_addr", ring_base}});
    writer_run_args.kernel_run_args = {writer_params};
    experimental::SetProgramRunArgs(writer_program, writer_run_args);

    distributed::MeshWorkload writer_workload;
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    writer_workload.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(writer_program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), writer_workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    ReadMeshDeviceProfilerResults(*mesh_device);

    std::vector<NOCDebugIssueType> unlocked_issues;
    for (IDevice* dev : mesh_device->get_devices()) {
        auto issues = fixture->get_write_to_unlocked_dfb_issues(dev->id(), virtual_core, 0);
        unlocked_issues.insert(unlocked_issues.end(), issues.begin(), issues.end());
    }
    EXPECT_EQ(unlocked_issues.size(), 1u)
        << "Expected exactly one WRITE_TO_UNLOCKED_DFB (launch 1's). Got " << unlocked_issues.size() << ".";
}

// Cross-core variant: a WRITER on a different core NOC-writes into the locker's locked DFB ring
void run_dfb_scoped_lock_xcore_test(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const experimental::NodeCoord locker_core = {0, 0};
    const experimental::NodeCoord writer_core = {1, 0};
    auto locker_vc = mesh_device->worker_core_from_logical_core(locker_core);
    auto writer_vc = mesh_device->worker_core_from_logical_core(writer_core);

    auto& mc = MetalContext::instance();
    uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
    uint32_t unreserved_addr =
        mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t entry_size = alignment * 2;
    uint32_t num_entries = 4;
    uint32_t write_size = alignment;
    uint32_t src_buffer_addr = unreserved_addr + 0x10000;  // writer's payload source (on the writer core)
    uint32_t scratch_addr = unreserved_addr + 0x20000;     // locker stages / writer receives the entry addr

    const experimental::DFBSpecName DFB_NAME{"lock_dfb"};
    const experimental::KernelSpecName LOCKER{"locker"};
    const experimental::KernelSpecName CONSUMER{"consumer"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::SemaphoreSpecName SEM_LOCKED{"sem_locked"};
    const experimental::SemaphoreSpecName SEM_WRITTEN{"sem_written"};

    experimental::DataflowBufferSpec dfb_spec{
        .unique_id = DFB_NAME,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::SemaphoreSpec sem_locked{
        .unique_id = SEM_LOCKED, .target_nodes = experimental::NodeRange{locker_core, writer_core}};
    experimental::SemaphoreSpec sem_written{
        .unique_id = SEM_WRITTEN, .target_nodes = experimental::NodeRange{locker_core, writer_core}};

    const experimental::DataMovementHardwareConfig dm_rv0 = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_0}};
    const experimental::DataMovementHardwareConfig dm_rv1 = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1}};

    experimental::KernelSpec locker_spec{
        .unique_id = LOCKER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_xcore_locker.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(DFB_NAME, "out")},
        .semaphore_bindings =
            {{.semaphore_spec_name = SEM_LOCKED, .accessor_name = "locked"},
             {.semaphore_spec_name = SEM_WRITTEN, .accessor_name = "written"}},
        .runtime_arg_schema = {.runtime_arg_names = {"writer_noc_x", "writer_noc_y", "local_scratch", "writer_inbox"}},
        .hw_config = dm_rv0,
    };
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_consumer.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(DFB_NAME, "in")},
        .hw_config = dm_rv1,
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_xcore_writer.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = SEM_LOCKED, .accessor_name = "locked"},
             {.semaphore_spec_name = SEM_WRITTEN, .accessor_name = "written"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src_buffer_addr", "write_size", "target_noc_x", "target_noc_y", "inbox"}},
        .hw_config = dm_rv0,
    };

    experimental::WorkUnitSpec wu_locker{
        .name = "locker_wu",
        .kernels = {LOCKER, CONSUMER},
        .target_nodes = locker_core,
    };
    experimental::WorkUnitSpec wu_writer{
        .name = "writer_wu",
        .kernels = {WRITER},
        .target_nodes = writer_core,
    };
    experimental::ProgramSpec spec{
        .name = "dfb_scoped_lock_xcore",
        .kernels = {locker_spec, consumer_spec, writer_spec},
        .dataflow_buffers = {dfb_spec},
        .semaphores = {sem_locked, sem_written},
        .work_units = {wu_locker, wu_writer},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    experimental::ProgramRunArgs::KernelRunArgs locker_params{};
    locker_params.kernel = LOCKER;
    locker_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        locker_core,
        {{"writer_noc_x", static_cast<uint32_t>(writer_vc.x)},
         {"writer_noc_y", static_cast<uint32_t>(writer_vc.y)},
         {"local_scratch", scratch_addr},   // this-core (locker) word to stage the entry addr
         {"writer_inbox", scratch_addr}});  // writer-core word to publish the entry addr into
    experimental::ProgramRunArgs::KernelRunArgs writer_params{};
    writer_params.kernel = WRITER;
    writer_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        writer_core,
        {{"src_buffer_addr", src_buffer_addr},
         {"write_size", write_size},
         {"target_noc_x", static_cast<uint32_t>(locker_vc.x)},
         {"target_noc_y", static_cast<uint32_t>(locker_vc.y)},
         {"inbox", scratch_addr}});  // local word the locker published the entry addr into
    experimental::ProgramRunArgs::KernelRunArgs consumer_params{};
    consumer_params.kernel = CONSUMER;  // no runtime args
    run_args.kernel_run_args = {locker_params, consumer_params, writer_params};
    experimental::SetProgramRunArgs(program, run_args);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    ReadMeshDeviceProfilerResults(*mesh_device);

    // The violation is recorded on the WRITER core (source of the NOC writes), with src = writer, dst = locker.
    std::vector<NOCDebugIssueType> locked_issues;
    for (IDevice* device : mesh_device->get_devices()) {
        auto issues = fixture->get_write_to_locked_issues(device->id(), writer_vc, 0);
        locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
    }
    ASSERT_FALSE(locked_issues.empty())
        << "Expected cross-core WRITE_TO_LOCKED_DFB; NOC debug did not report the violation.";
    for (const auto& issue : locked_issues) {
        EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_DFB);
        EXPECT_GE(issue.issue_size, write_size);
        EXPECT_GT(issue.issue_address, 0u);
        EXPECT_EQ(issue.src_x, writer_vc.x);  // cross-core: write sourced from the writer core
        EXPECT_EQ(issue.src_y, writer_vc.y);
        EXPECT_EQ(issue.dst_x, locker_vc.x);  // landed on the locker core (the DFB's L1)
        EXPECT_EQ(issue.dst_y, locker_vc.y);
    }
}

// Loopback multicast: a producer multicasts into its own unlocked DFB ring, with the producer sitting in the
// interior of the mcast rectangle. Only by iterating the whole rectangle (and matching the source core within
// it) can the tracker attribute the self-write and flag WRITE_TO_UNLOCKED_DFB.
void run_dfb_mcast_loopback_unlocked_test(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const experimental::NodeCoord producer_core = {1, 0};  // middle of the row -> never a rectangle corner
    const experimental::NodeCoord rect_lo = {0, 0};
    const experimental::NodeCoord rect_hi = {2, 0};
    auto producer_vc = mesh_device->worker_core_from_logical_core(producer_core);
    auto vc_lo = mesh_device->worker_core_from_logical_core(rect_lo);
    auto vc_hi = mesh_device->worker_core_from_logical_core(rect_hi);
    uint32_t mcast_x_start = std::min<uint32_t>(vc_lo.x, vc_hi.x);
    uint32_t mcast_x_end = std::max<uint32_t>(vc_lo.x, vc_hi.x);
    uint32_t mcast_y = producer_vc.y;
    uint32_t num_dests = 3;  // whole row, including the source (MCAST_INCL_SRC)

    auto& mc = MetalContext::instance();
    uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
    uint32_t unreserved_addr =
        mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t entry_size = alignment * 2;
    uint32_t num_entries = 4;
    uint32_t write_size = alignment;
    uint32_t src_buffer_addr = unreserved_addr + 0x10000;  // producer's own L1 source, clear of the DFB

    const experimental::DFBSpecName DFB_NAME{"lock_dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};

    experimental::DataflowBufferSpec dfb_spec{
        .unique_id = DFB_NAME,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const experimental::DataMovementHardwareConfig dm_producer_cfg = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_0}};
    const experimental::DataMovementHardwareConfig dm_consumer_cfg = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1}};

    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_mcast_loopback_producer.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(DFB_NAME, "out")},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src_buffer_addr",
                  "write_size",
                  "mcast_noc_x_start",
                  "mcast_noc_y_start",
                  "mcast_noc_x_end",
                  "mcast_noc_y_end",
                  "num_dests"}},
        .hw_config = dm_producer_cfg,
    };
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_consumer.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(DFB_NAME, "in")},
        .hw_config = dm_consumer_cfg,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = producer_core,
    };
    experimental::ProgramSpec spec{
        .name = "dfb_mcast_loopback",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    experimental::ProgramRunArgs::KernelRunArgs producer_params{};
    producer_params.kernel = PRODUCER;
    producer_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        producer_core,
        {{"src_buffer_addr", src_buffer_addr},
         {"write_size", write_size},
         {"mcast_noc_x_start", mcast_x_start},
         {"mcast_noc_y_start", mcast_y},
         {"mcast_noc_x_end", mcast_x_end},
         {"mcast_noc_y_end", mcast_y},
         {"num_dests", num_dests}});
    experimental::ProgramRunArgs::KernelRunArgs consumer_params{};
    consumer_params.kernel = CONSUMER;
    run_args.kernel_run_args = {producer_params, consumer_params};
    experimental::SetProgramRunArgs(program, run_args);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    ReadMeshDeviceProfilerResults(*mesh_device);

    // The violation is recorded on the producer core (source == destination for a loopback write).
    std::vector<NOCDebugIssueType> unlocked_issues;
    for (IDevice* device : mesh_device->get_devices()) {
        auto issues = fixture->get_write_to_unlocked_dfb_issues(device->id(), producer_vc, 0);
        unlocked_issues.insert(unlocked_issues.end(), issues.begin(), issues.end());
    }
    ASSERT_FALSE(unlocked_issues.empty())
        << "Expected loopback-mcast WRITE_TO_UNLOCKED_DFB; the interior source core was not flagged.";
    for (const auto& issue : unlocked_issues) {
        EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_UNLOCKED_DFB);
        EXPECT_GE(issue.issue_size, write_size);
        EXPECT_GT(issue.issue_address, 0u);
        EXPECT_EQ(issue.src_x, producer_vc.x);  // loopback: src == dst == producer core
        EXPECT_EQ(issue.src_y, producer_vc.y);
        EXPECT_EQ(issue.dst_x, producer_vc.x);
        EXPECT_EQ(issue.dst_y, producer_vc.y);
    }
}

// Cross-core multicast: a writer on a different row multicasts across a 3-core row whose interior
// core holds a locked DFB. The locker is not the mcast start/end corner, so only iterating the whole
// rectangle flags WRITE_TO_LOCKED_DFB.
void run_dfb_mcast_xcore_locked_test(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const experimental::NodeCoord locker_core = {1, 0};  // middle of the row -> never a rectangle corner
    const experimental::NodeCoord writer_core = {1, 1};  // different row -> outside the mcast rectangle
    const experimental::NodeCoord rect_lo = {0, 0};
    const experimental::NodeCoord rect_hi = {2, 0};
    auto locker_vc = mesh_device->worker_core_from_logical_core(locker_core);
    auto writer_vc = mesh_device->worker_core_from_logical_core(writer_core);
    auto vc_lo = mesh_device->worker_core_from_logical_core(rect_lo);
    auto vc_hi = mesh_device->worker_core_from_logical_core(rect_hi);
    uint32_t mcast_x_start = std::min<uint32_t>(vc_lo.x, vc_hi.x);
    uint32_t mcast_x_end = std::max<uint32_t>(vc_lo.x, vc_hi.x);
    uint32_t mcast_y = locker_vc.y;
    uint32_t num_dests = 3;  // whole row-0 rectangle (writer is on row 1, outside it)

    auto& mc = MetalContext::instance();
    uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
    uint32_t unreserved_addr =
        mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t entry_size = alignment * 2;
    uint32_t num_entries = 4;
    uint32_t write_size = alignment;
    uint32_t src_buffer_addr = unreserved_addr + 0x10000;  // writer's payload source (on the writer core)
    uint32_t scratch_addr = unreserved_addr + 0x20000;     // locker stages / writer receives the entry addr

    const experimental::DFBSpecName DFB_NAME{"lock_dfb"};
    const experimental::KernelSpecName LOCKER{"locker"};
    const experimental::KernelSpecName CONSUMER{"consumer"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::SemaphoreSpecName SEM_LOCKED{"sem_locked"};
    const experimental::SemaphoreSpecName SEM_WRITTEN{"sem_written"};

    experimental::DataflowBufferSpec dfb_spec{
        .unique_id = DFB_NAME,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::SemaphoreSpec sem_locked{
        .unique_id = SEM_LOCKED, .target_nodes = experimental::NodeRange{locker_core, writer_core}};
    experimental::SemaphoreSpec sem_written{
        .unique_id = SEM_WRITTEN, .target_nodes = experimental::NodeRange{locker_core, writer_core}};

    const experimental::DataMovementHardwareConfig dm_rv0 = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_0}};
    const experimental::DataMovementHardwareConfig dm_rv1 = experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1}};

    experimental::KernelSpec locker_spec{
        .unique_id = LOCKER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_xcore_locker.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(DFB_NAME, "out")},
        .semaphore_bindings =
            {{.semaphore_spec_name = SEM_LOCKED, .accessor_name = "locked"},
             {.semaphore_spec_name = SEM_WRITTEN, .accessor_name = "written"}},
        .runtime_arg_schema = {.runtime_arg_names = {"writer_noc_x", "writer_noc_y", "local_scratch", "writer_inbox"}},
        .hw_config = dm_rv0,
    };
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_consumer.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(DFB_NAME, "in")},
        .hw_config = dm_rv1,
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_dfb_mcast_xcore_writer.cpp",
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = SEM_LOCKED, .accessor_name = "locked"},
             {.semaphore_spec_name = SEM_WRITTEN, .accessor_name = "written"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src_buffer_addr",
                  "write_size",
                  "locker_noc_x",
                  "locker_noc_y",
                  "mcast_noc_x_start",
                  "mcast_noc_y_start",
                  "mcast_noc_x_end",
                  "mcast_noc_y_end",
                  "num_dests",
                  "inbox"}},
        .hw_config = dm_rv0,
    };

    experimental::WorkUnitSpec wu_locker{
        .name = "locker_wu",
        .kernels = {LOCKER, CONSUMER},
        .target_nodes = locker_core,
    };
    experimental::WorkUnitSpec wu_writer{
        .name = "writer_wu",
        .kernels = {WRITER},
        .target_nodes = writer_core,
    };
    experimental::ProgramSpec spec{
        .name = "dfb_mcast_xcore",
        .kernels = {locker_spec, consumer_spec, writer_spec},
        .dataflow_buffers = {dfb_spec},
        .semaphores = {sem_locked, sem_written},
        .work_units = {wu_locker, wu_writer},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs run_args;
    experimental::ProgramRunArgs::KernelRunArgs locker_params{};
    locker_params.kernel = LOCKER;
    locker_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        locker_core,
        {{"writer_noc_x", static_cast<uint32_t>(writer_vc.x)},
         {"writer_noc_y", static_cast<uint32_t>(writer_vc.y)},
         {"local_scratch", scratch_addr},
         {"writer_inbox", scratch_addr}});
    experimental::ProgramRunArgs::KernelRunArgs writer_params{};
    writer_params.kernel = WRITER;
    writer_params.runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
        writer_core,
        {{"src_buffer_addr", src_buffer_addr},
         {"write_size", write_size},
         {"locker_noc_x", static_cast<uint32_t>(locker_vc.x)},
         {"locker_noc_y", static_cast<uint32_t>(locker_vc.y)},
         {"mcast_noc_x_start", mcast_x_start},
         {"mcast_noc_y_start", mcast_y},
         {"mcast_noc_x_end", mcast_x_end},
         {"mcast_noc_y_end", mcast_y},
         {"num_dests", num_dests},
         {"inbox", scratch_addr}});
    experimental::ProgramRunArgs::KernelRunArgs consumer_params{};
    consumer_params.kernel = CONSUMER;  // no runtime args
    run_args.kernel_run_args = {locker_params, consumer_params, writer_params};
    experimental::SetProgramRunArgs(program, run_args);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());
    ReadMeshDeviceProfilerResults(*mesh_device);

    // The violation is recorded on the WRITER core (source of the mcast), with src = writer, dst = locker.
    std::vector<NOCDebugIssueType> locked_issues;
    for (IDevice* device : mesh_device->get_devices()) {
        auto issues = fixture->get_write_to_locked_issues(device->id(), writer_vc, 0);
        locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
    }
    ASSERT_FALSE(locked_issues.empty())
        << "Expected mcast cross-core WRITE_TO_LOCKED_DFB; the interior locked core was not flagged.";
    for (const auto& issue : locked_issues) {
        EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_DFB);
        EXPECT_GE(issue.issue_size, write_size);
        EXPECT_GT(issue.issue_address, 0u);
        EXPECT_EQ(issue.src_x, writer_vc.x);  // cross-core: write sourced from the writer core
        EXPECT_EQ(issue.src_y, writer_vc.y);
        EXPECT_EQ(issue.dst_x, locker_vc.x);  // landed on the interior locker core (the DFB's L1)
        EXPECT_EQ(issue.dst_y, locker_vc.y);
    }
}

}  // namespace

// Writing into an entry of the ring while holding its lock -> no issue.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBWriteInOwnLockNoIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this, mesh_device, /*target_entry_index=*/0, /*write_after_unlock=*/false, ExpectedDfbIssue::None);
    }
}

// The write-lock covers only entry 0, so writing a different (unlocked) entry of the DFB region ->
// WRITE_TO_UNLOCKED_DFB.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBWriteUnlockedEntryIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this, mesh_device, /*target_entry_index=*/2, /*write_after_unlock=*/false, ExpectedDfbIssue::Unlocked);
    }
}

// Same case as above, but the producer drives NOC_1 instead of NOC_0.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBWriteUnlockedEntryIssueNoc1) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this,
            mesh_device,
            /*target_entry_index=*/2,
            /*write_after_unlock=*/false,
            ExpectedDfbIssue::Unlocked,
            /*skip_lock=*/false,
            /*producer_noc=*/NOC::NOC_1);
    }
}

// Same case again, but the DFB producer runs on NCRISC instead of BRISC, and the issue is
// asserted on NCRISC.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBWriteUnlockedEntryIssueNcrisc) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this,
            mesh_device,
            /*target_entry_index=*/2,
            /*write_after_unlock=*/false,
            ExpectedDfbIssue::Unlocked,
            /*skip_lock=*/false,
            /*producer_noc=*/NOC::NOC_0,
            /*producer_processor=*/DataMovementProcessor::RISCV_1);
    }
}

// Writing into the entry you yourself locked, from NCRISC.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBWriteInOwnLockNoIssueNcrisc) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this,
            mesh_device,
            /*target_entry_index=*/0,
            /*write_after_unlock=*/false,
            ExpectedDfbIssue::None,
            /*skip_lock=*/false,
            /*producer_noc=*/NOC::NOC_0,
            /*producer_processor=*/DataMovementProcessor::RISCV_1);
    }
}

// Lock held (one entry), but the write targets just PAST the ring -> no issue (outside the DFB region).
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBNoIssueSpatial) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this, mesh_device, /*target_entry_index=*/4, /*write_after_unlock=*/false, ExpectedDfbIssue::None);
    }
}

// Write into the ring after releasing the lock -> WRITE_TO_UNLOCKED_DFB.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBWriteAfterUnlockIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this, mesh_device, /*target_entry_index=*/0, /*write_after_unlock=*/true, ExpectedDfbIssue::Unlocked);
    }
}

// A producer that writes into the ring without locking it -> WRITE_TO_UNLOCKED_DFB.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBWriteNeverLockedIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this,
            mesh_device,
            /*target_entry_index=*/1,
            /*write_after_unlock=*/false,
            ExpectedDfbIssue::Unlocked,
            /*skip_lock=*/true);
    }
}

// A DFB's L1 extent must stop being tracked when the kernel that declared it exits, so a later program
// reusing that L1 is not falsely flagged.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBRegionClearedBetweenLaunches) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_region_cleared_between_launches_test(this, mesh_device);
    }
}

// Cross-core: a writer on a DIFFERENT core NOC-writes into the locker's locked DFB ring -> WRITE_TO_LOCKED_DFB
// with src = writer core, dst = locker core.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBIssueCrossCore) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 2) {
            GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
        }
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_xcore_test(this, mesh_device);
    }
}

// Loopback multicast into the producer's own unlocked DFB, with the producer in the interior of the mcast
// rectangle. Verifies the tracker iterates the whole rectangle and attributes the self-write ->
// WRITE_TO_UNLOCKED_DFB.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBLoopbackMcastUnlockedIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 3) {
            GTEST_SKIP() << "Test requires at least 3 cores in x dimension";
        }
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_mcast_loopback_unlocked_test(this, mesh_device);
    }
}

// Cross-core multicast that lands on another core's locked DFB, where that core is the interior of the mcast
// rectangle (never the start/end corner). Verifies the tracker checks every core the mcast covers ->
// WRITE_TO_LOCKED_DFB.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBMcastCrossCoreInteriorIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 3 || grid_size.y < 2) {
            GTEST_SKIP() << "Test requires at least a 3x2 core grid";
        }
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_mcast_xcore_locked_test(this, mesh_device);
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessRemoteCBIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 3) {
            GTEST_SKIP() << "Test requires at least 3 cores in x dimension";
        }

        const CoreCoord sender_core = {0, 0};
        const CoreCoord receiver_core = {1, 0};  // locker
        const CoreCoord writer_core = {2, 0};
        CoreRangeSet receiver_cores = CoreRangeSet(CoreRange(receiver_core));

        constexpr uint32_t gcb_page_size = 32;
        constexpr uint32_t gcb_size = gcb_page_size * 100;  // 3200 bytes
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{sender_core, receiver_cores}};
        auto global_cb = experimental::CreateGlobalCircularBuffer(
            mesh_device.get(), sender_receiver_core_mapping, gcb_size, BufferType::L1);

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        workload.add_program(device_range, CreateProgram());
        Program& program = workload.get_programs().at(device_range);

        constexpr uint32_t remote_cb_index = 31;
        CircularBufferConfig gcb_config = CircularBufferConfig(gcb_page_size);
        gcb_config.remote_index(remote_cb_index)
            .set_page_size(gcb_page_size)
            .set_data_format(tt::DataFormat::Float16_b);
        experimental::CreateCircularBuffer(program, receiver_cores, gcb_config, global_cb);

        auto receiver_virtual_core = mesh_device->worker_core_from_logical_core(receiver_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);
        uint32_t receiver_sem_id = CreateSemaphore(program, receiver_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        auto& mc = MetalContext::instance();
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        // NOC records sub-32B writes rounded up, so use a 32B-aligned size for an exact issue_size check.
        uint32_t write_size = 2 * alignment;
        // The remote CB's L1 region base is the global CB buffer address (== fifo_start_addr on the
        // receiver). A single write there lands exactly at the start of the locked region, so the flagged
        // issue address is known exactly on the host.
        uint32_t gcb_addr = static_cast<uint32_t>(global_cb.buffer_address());

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_remote_cb_locker.cpp",
            receiver_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        // The locker stages the locked remote-CB base here and NOCs it to the same offset on the writer, so the
        // writer can target the locked region directly (same protocol as the CB tests above).
        uint32_t scratch_addr = unreserved_addr + 0x20000;

        SetRuntimeArgs(
            program,
            locker_kernel,
            receiver_core,
            {remote_cb_index,
             receiver_sem_id,
             writer_sem_id,
             writer_virtual_core.x,
             writer_virtual_core.y,
             scratch_addr,
             scratch_addr});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {unreserved_addr,
             write_size,
             receiver_virtual_core.x,
             receiver_virtual_core.y,
             scratch_addr,
             writer_sem_id,
             receiver_sem_id,
             receiver_virtual_core.x,
             receiver_virtual_core.y});

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        ReadMeshDeviceProfilerResults(*mesh_device);

        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = this->get_write_to_locked_issues(device->id(), writer_virtual_core, 0);
            locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
        }
        ASSERT_FALSE(locked_issues.empty())
            << "Expected write-to-locked-CB issue on writer core; NOC debug did not report the "
               "RemoteCircularBuffer lock violation.";
        for (const auto& issue : locked_issues) {
            EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB);
            EXPECT_EQ(issue.issue_address, gcb_addr);  // exact: the remote CB's L1 region base
            EXPECT_EQ(issue.issue_size, write_size);
            EXPECT_EQ(issue.src_x, writer_virtual_core.x);
            EXPECT_EQ(issue.src_y, writer_virtual_core.y);
            EXPECT_EQ(issue.dst_x, receiver_virtual_core.x);
            EXPECT_EQ(issue.dst_y, receiver_virtual_core.y);
        }
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessRemoteCBNoIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 3) {
            GTEST_SKIP() << "Test requires at least 3 cores in x dimension";
        }

        const CoreCoord sender_core = {0, 0};
        const CoreCoord receiver_core = {1, 0};  // locker
        const CoreCoord writer_core = {2, 0};
        CoreRangeSet receiver_cores = CoreRangeSet(CoreRange(receiver_core));

        constexpr uint32_t gcb_page_size = 32;
        constexpr uint32_t gcb_size = gcb_page_size * 100;  // 3200 bytes
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{sender_core, receiver_cores}};
        auto global_cb = experimental::CreateGlobalCircularBuffer(
            mesh_device.get(), sender_receiver_core_mapping, gcb_size, BufferType::L1);

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        workload.add_program(device_range, CreateProgram());
        Program& program = workload.get_programs().at(device_range);

        constexpr uint32_t remote_cb_index = 31;
        CircularBufferConfig gcb_config = CircularBufferConfig(gcb_page_size);
        gcb_config.remote_index(remote_cb_index)
            .set_page_size(gcb_page_size)
            .set_data_format(tt::DataFormat::Float16_b);
        experimental::CreateCircularBuffer(program, receiver_cores, gcb_config, global_cb);

        auto receiver_virtual_core = mesh_device->worker_core_from_logical_core(receiver_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);
        uint32_t receiver_sem_id = CreateSemaphore(program, receiver_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        auto& mc = MetalContext::instance();
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        // NOC records sub-32B writes rounded up, so use a 32B-aligned size for an exact issue_size check.
        uint32_t write_size = 2 * alignment;

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_remote_cb_locker_no_issue.cpp",
            receiver_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        // Same publish/inbox protocol as the Issue variant; here the lock is only taken while the writer is idle.
        uint32_t scratch_addr = unreserved_addr + 0x20000;

        SetRuntimeArgs(
            program,
            locker_kernel,
            receiver_core,
            {remote_cb_index,
             receiver_sem_id,
             writer_sem_id,
             writer_virtual_core.x,
             writer_virtual_core.y,
             scratch_addr,
             scratch_addr});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {unreserved_addr,
             write_size,
             receiver_virtual_core.x,
             receiver_virtual_core.y,
             scratch_addr,
             writer_sem_id,
             receiver_sem_id,
             receiver_virtual_core.x,
             receiver_virtual_core.y});

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        ReadMeshDeviceProfilerResults(*mesh_device);

        for (IDevice* device : mesh_device->get_devices()) {
            ChipId chip_id = device->id();
            EXPECT_FALSE(this->has_write_to_locked_issue(chip_id, writer_virtual_core, 0))
                << "Unexpected write-to-locked-CB issue on writer core; writes were outside the "
                   "RemoteCircularBuffer lock scope.";
            EXPECT_FALSE(this->has_write_to_locked_issue(chip_id, receiver_virtual_core, 0))
                << "Unexpected write-to-locked-CB issue on receiver core.";
        }
    }
}

// Same locker/writer setup as ScopedLockConcurrentAccessIssue, but the writer uses STATEFUL writes. This is the
// end-to-end check that the destination a kernel actually sends is what the host captures: the hardware programs the
// destination core once in the set-state call and each write supplies only the address word, so the recorded write
// events carry a placeholder (0,0) core. The host must correlate with the set-state event to recover the real
// destination. Asserting the reported coordinates, address and size therefore validates the whole device -> host
// pipeline (device emission, event encoding, and host reconstruction) rather than just the host state machine.
void RunScopedLockStatefulWriteTest(
    NOCDebuggingFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_trid) {
    auto grid_size = mesh_device->compute_with_storage_grid_size();
    if (grid_size.x < 2) {
        GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
    }

    const CoreCoord locker_core = {0, 0};
    const CoreCoord writer_core = {1, 0};
    Program program = CreateProgram();
    distributed::MeshWorkload workload;

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    auto& mc = MetalContext::instance();
    uint32_t unreserved_addr =
        mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);

    // Put the destination well away from the writer's own buffer so a mixed-up src/dst cannot coincidentally match.
    // Note this is an ordinary L1 address (~17 bits): L1 is 1.5MB, so no address reachable here comes close to
    // exercising the width of the event's address field.
    uint32_t locker_buffer_addr = unreserved_addr + (alignment * 64);
    uint32_t writer_buffer_addr = unreserved_addr;
    uint32_t num_elements = 8;

    auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);
    auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);

    uint32_t locker_sem_id = CreateSemaphore(program, locker_core, 0);
    uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

    std::vector<uint32_t> locker_args = {
        locker_buffer_addr, num_elements, locker_sem_id, writer_sem_id, writer_virtual_core.x, writer_virtual_core.y};

    KernelHandle locker_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel.cpp",
        locker_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

    SetRuntimeArgs(program, locker_kernel, locker_core, locker_args);

    std::vector<uint32_t> writer_args = {
        writer_buffer_addr,
        num_elements,
        locker_virtual_core.x,
        locker_virtual_core.y,
        locker_buffer_addr,
        writer_sem_id,
        locker_sem_id,
        locker_virtual_core.x,
        locker_virtual_core.y};

    std::map<std::string, std::string> writer_defines;
    if (use_trid) {
        writer_defines["USE_TRID"] = "1";
    }

    KernelHandle writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_stateful_writer_kernel.cpp",
        writer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .defines = writer_defines});

    SetRuntimeArgs(program, writer_kernel, writer_core, writer_args);

    workload.add_program(device_range, std::move(program));

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    ReadMeshDeviceProfilerResults(*mesh_device);

    std::vector<NOCDebugIssueType> locked_issues;
    for (IDevice* device : mesh_device->get_devices()) {
        auto issues = fixture->get_write_to_locked_issues(device->id(), writer_virtual_core, 0);
        locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
    }
    ASSERT_FALSE(locked_issues.empty())
        << "Expected a write-to-locked-buffer issue for a stateful write into the locked region. Either the "
           "set-state event was not captured or the destination was not reconstructed from it.";

    uint32_t expected_write_size = num_elements * sizeof(uint32_t);
    for (const auto& issue : locked_issues) {
        EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM);
        // The address travels on the write event itself (the set-state only carries the base/coords).
        EXPECT_EQ(issue.issue_address, locker_buffer_addr)
            << "captured destination address does not match the address the kernel wrote to";
        // For the non-trid variant the size comes from the set-state; for the trid variant from the write event.
        EXPECT_EQ(issue.issue_size, expected_write_size);
        EXPECT_EQ(issue.src_x, writer_virtual_core.x);
        EXPECT_EQ(issue.src_y, writer_virtual_core.y);
        // The critical assertion: the destination core came from the set-state, not the (0,0) placeholder.
        EXPECT_EQ(issue.dst_x, locker_virtual_core.x)
            << "destination core was not reconstructed from the set-state event";
        EXPECT_EQ(issue.dst_y, locker_virtual_core.y)
            << "destination core was not reconstructed from the set-state event";
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockStatefulWriteToLockedIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        RunScopedLockStatefulWriteTest(this, mesh_device, /*use_trid=*/false);
    }
}

TEST_F(NOCDebuggingFixture, ScopedLockStatefulTridWriteToLockedIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        RunScopedLockStatefulWriteTest(this, mesh_device, /*use_trid=*/true);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// SKIPPED / KNOWN-FAILING: a destination address wider than the event encoding.
//
// This test asserts the property we actually want -- that the destination address the host reports is *exactly* the
// address the kernel wrote to, for an address large enough to exceed the current event encoding. It cannot pass
// today and is skipped; enable it (delete the GTEST_SKIP below) once the device -> host debug pipeline is widened.
//
// It is skipped rather than merely DISABLED_ because the body cannot safely be run at all right now: a large
// destination address is not a valid NOC destination on a worker core, so the write is never accepted and the kernel
// blocks forever -- force-running this was verified to hang the device rather than fail cleanly. Whoever enables it
// must therefore also choose a destination that both exceeds 24 bits AND is a valid NOC target (DRAM is the obvious
// candidate, which in turn needs an observation hook other than the write-to-locked-buffer check, since only cores
// hold scoped locks). That open question is the reason this is a placeholder rather than a working test.
//
// Why it cannot pass today. A NOC address is a coordinate part plus a 36-bit local address
// (NOC_ADDR_LOCAL_BITS == 36 on Wormhole, Blackhole and Quasar). The local part reaches the host in
// KernelProfilerNocEventMetadata::LocalNocEventDstTrailer, which packs it as:
//
//     dst_addr_4b : 22   (address >> 2)      -> only the low 24 bits of the address survive (16MB)
//     dst_addr_offset : 4  (only 2 bits used, `addr & 0x3`)
//     src_addr_4b : 22, src_addr_offset : 4, counter_value : 12     -- 64 bits, fully packed
//
// On top of that, setDstAddr/getDstAddr, NocWriteEvent::dst_addr and NOCDebugIssueType::issue_address are all
// uint32_t, and the lock side has the same limit (NocDebuggingEventMetadata::locked_addr is 24 bits). So any
// destination at or above 16MB -- i.e. every DRAM address -- is currently recorded truncated, and the reported
// address cannot match what the kernel really targeted.
//
// What would make it pass. Widening the encoding so the full local address survives. Note the two *_addr_offset
// fields declare 4 bits but only use 2, so 4 bits are already spare; rebalancing (src_addr is always local L1 and
// needs at most 21 bits) gives dst 29+2 = 31 bits within the same 64-bit word, which covers DRAM but still not the
// full 36 bits. Reaching 36 bits needs a wider event record, and the uint32_t address fields on the host widened
// to match.
//
// Safety of the write itself. The destination address deliberately does not correspond to real memory on the target
// core, so the write is not expected to land anywhere meaningful -- only the *recorded* address matters here. It
// uses POSTED writes for exactly that reason: a non-posted write would wait forever for an acknowledgement from a
// destination that never responds. The lock side is safe unconditionally because scoped_lock only emits a debug
// event and never dereferences the address.
//
TEST_F(NOCDebuggingFixture, ScopedLockStatefulWriteLargeDestinationAddressRecordedExactly) {
    GTEST_SKIP() << "Known limitation: the NOC event encoding keeps only the low 24 bits of a destination address, "
                    "so a destination beyond 16MB cannot round-trip to the host. Remove this skip once the "
                    "device->host debug pipeline carries the full address; see the comment above this test, which "
                    "also covers why the body cannot be run as-is today.";

    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        auto grid_size = mesh_device->compute_with_storage_grid_size();
        if (grid_size.x < 2) {
            GTEST_SKIP() << "Test requires at least 2 cores in x dimension";
        }

        const CoreCoord locker_core = {0, 0};
        const CoreCoord writer_core = {1, 0};
        Program program = CreateProgram();
        distributed::MeshWorkload workload;

        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

        // 64MB: past the 24-bit window the trailer can represent, while still inside the 36-bit NOC local address
        // space that Wormhole, Blackhole and Quasar all share. If a particular architecture ever needs a different
        // value to stay clear of a special aperture, select it per arch here.
        constexpr uint32_t large_dst_addr = 0x0400'0000;
        static_assert(large_dst_addr > 0xFF'FFFF, "the point of this test is an address beyond the 24-bit field");

        uint32_t writer_buffer_addr = unreserved_addr;  // the source stays an ordinary L1 address
        uint32_t num_elements = 8;

        auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);
        auto writer_virtual_core = mesh_device->worker_core_from_logical_core(writer_core);

        uint32_t locker_sem_id = CreateSemaphore(program, locker_core, 0);
        uint32_t writer_sem_id = CreateSemaphore(program, writer_core, 0);

        // The locker "locks" the large region. scoped_lock only records an event, so no memory is touched.
        std::vector<uint32_t> locker_args = {
            large_dst_addr, num_elements, locker_sem_id, writer_sem_id, writer_virtual_core.x, writer_virtual_core.y};

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, locker_kernel, locker_core, locker_args);

        std::vector<uint32_t> writer_args = {
            writer_buffer_addr,
            num_elements,
            locker_virtual_core.x,
            locker_virtual_core.y,
            large_dst_addr,
            writer_sem_id,
            locker_sem_id,
            locker_virtual_core.x,
            locker_virtual_core.y};

        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_stateful_writer_kernel.cpp",
            writer_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .defines = {{"USE_POSTED", "1"}}});

        SetRuntimeArgs(program, writer_kernel, writer_core, writer_args);

        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());

        ReadMeshDeviceProfilerResults(*mesh_device);

        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = this->get_write_to_locked_issues(device->id(), writer_virtual_core, 0);
            locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
        }
        // Fails today: the recorded write address is truncated to its low 24 bits and the recorded lock address is
        // capped at 0xFFFFFF, so the two no longer overlap and no violation is reported at all.
        ASSERT_FALSE(locked_issues.empty())
            << "No write-to-locked issue reported for a large destination address. Expected while the event "
               "encoding truncates addresses to 24 bits; see the comment above this test.";

        for (const auto& issue : locked_issues) {
            // The assertion that matters: the address must survive the round trip to the host unchanged.
            EXPECT_EQ(issue.issue_address, large_dst_addr)
                << "host recorded 0x" << std::hex << issue.issue_address << " but the kernel wrote to 0x"
                << large_dst_addr << "; the destination address did not survive the event encoding";
            EXPECT_EQ(issue.dst_x, locker_virtual_core.x);
            EXPECT_EQ(issue.dst_y, locker_virtual_core.y);
        }
    }
}

}  // namespace tt::tt_metal
