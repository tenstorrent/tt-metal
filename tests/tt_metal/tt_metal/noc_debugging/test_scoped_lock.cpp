// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"

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
        uint32_t l1_size = mesh_device->l1_size_per_core();
        uint32_t stride = alignment * 64;

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
             writer_virtual_core.y});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {writer_buffer_addr,
             write_size,
             locker_virtual_core.x,
             locker_virtual_core.y,
             unreserved_addr,
             l1_size,
             stride,
             writer_sem_id,
             locker_sem_id,
             locker_virtual_core.x,
             locker_virtual_core.y});

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        ReadMeshDeviceProfilerResults(*mesh_device);

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
            EXPECT_GT(issue.issue_address, 0u);
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
        uint32_t l1_size = mesh_device->l1_size_per_core();
        uint32_t stride = alignment * 64;

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel_cb_no_issue.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel_no_issue.cpp",
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
             writer_virtual_core.y});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {writer_buffer_addr,
             write_size,
             locker_virtual_core.x,
             locker_virtual_core.y,
             unreserved_addr,
             l1_size,
             stride,
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

TEST_F(NOCDebuggingFixture, ScopedLockSelfWriteToLockedIssue) {
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

        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = this->get_write_to_locked_issues(device->id(), virtual_core, 0);
            locked_issues.insert(locked_issues.end(), issues.begin(), issues.end());
        }
        ASSERT_FALSE(locked_issues.empty())
            << "Expected write-to-locked issue when kernel writes to its own locked region";

        for (const auto& issue : locked_issues) {
            EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM);
            EXPECT_EQ(issue.issue_address, write_target_addr);
            EXPECT_EQ(issue.issue_size, write_size);
            EXPECT_EQ(issue.src_x, virtual_core.x);
            EXPECT_EQ(issue.src_y, virtual_core.y);
            EXPECT_EQ(issue.dst_x, virtual_core.x);
            EXPECT_EQ(issue.dst_y, virtual_core.y);
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

TEST_F(NOCDebuggingFixture, ScopedLockCBSelfWriteToLockedIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());

        const CoreCoord core = {0, 0};
        auto virtual_core = mesh_device->worker_core_from_logical_core(core);

        auto& mc = MetalContext::instance();
        uint32_t unreserved_addr =
            mc.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        uint32_t alignment = mc.hal().get_alignment(HalMemType::L1);

        constexpr uint8_t cb_index = 0;
        uint32_t cb_page_size = 2048;
        uint32_t cb_total_size = cb_page_size * 2;

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        workload.add_program(device_range, CreateProgram());
        Program& program = workload.get_programs().at(device_range);

        CircularBufferConfig cb_config = CircularBufferConfig(cb_total_size, {{cb_index, tt::DataFormat::Float16_b}})
                                             .set_page_size(cb_index, cb_page_size);
        CreateCircularBuffer(program, core, cb_config);

        // NOC records sub-32B writes rounded up, so use a 32B-aligned size for an exact issue_size check.
        uint32_t write_size = 2 * alignment;
        uint32_t src_buffer_addr = unreserved_addr;
        uint32_t scratch_addr = unreserved_addr + write_size;  // host reads the kernel's write target here

        KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_self_write_kernel.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        SetRuntimeArgs(
            program,
            kernel,
            core,
            {static_cast<uint32_t>(cb_index),
             src_buffer_addr,
             write_size,
             virtual_core.x,
             virtual_core.y,
             scratch_addr});

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        ReadMeshDeviceProfilerResults(*mesh_device);

        for (IDevice* device : mesh_device->get_devices()) {
            std::vector<uint32_t> scratch;
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core, scratch_addr, sizeof(uint32_t), scratch, CoreType::WORKER);
            ASSERT_FALSE(scratch.empty());
            uint32_t expected_addr = scratch[0];

            auto locked_issues = this->get_write_to_locked_issues(device->id(), virtual_core, 0);
            ASSERT_FALSE(locked_issues.empty())
                << "Expected write-to-locked-CB issue when the kernel writes into its own locked "
                   "CircularBuffer region. A 16x-inflated (<<4) lock region would sit outside L1 and miss this.";
            for (const auto& issue : locked_issues) {
                EXPECT_EQ(issue.base_type, NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB);
                EXPECT_EQ(issue.issue_address, expected_addr);
                EXPECT_EQ(issue.issue_size, write_size);
                EXPECT_EQ(issue.src_x, virtual_core.x);
                EXPECT_EQ(issue.src_y, virtual_core.y);
                EXPECT_EQ(issue.dst_x, virtual_core.x);
                EXPECT_EQ(issue.dst_y, virtual_core.y);
            }
        }
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

        SetRuntimeArgs(
            program,
            locker_kernel,
            receiver_core,
            {remote_cb_index, receiver_sem_id, writer_sem_id, writer_virtual_core.x, writer_virtual_core.y});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {unreserved_addr,
             write_size,
             receiver_virtual_core.x,
             receiver_virtual_core.y,
             gcb_addr,
             gcb_addr + write_size,
             write_size,
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
        uint32_t l1_size = mesh_device->l1_size_per_core();
        // Sweep the whole L1: the receiver's remote-CB region (gcb_size bytes) is much larger than the
        // stride, so the sweep is guaranteed to hit it wherever the allocator placed it.
        uint32_t stride = alignment * 64;

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_remote_cb_locker_no_issue.cpp",
            receiver_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_cb_writer_kernel_no_issue.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(
            program,
            locker_kernel,
            receiver_core,
            {remote_cb_index, receiver_sem_id, writer_sem_id, writer_virtual_core.x, writer_virtual_core.y});
        SetRuntimeArgs(
            program,
            writer_kernel,
            writer_core,
            {unreserved_addr,
             write_size,
             receiver_virtual_core.x,
             receiver_virtual_core.y,
             unreserved_addr,
             l1_size,
             stride,
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
