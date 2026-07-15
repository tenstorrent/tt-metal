// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
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

namespace {

void run_dfb_scoped_lock_test(
    NOCDebuggingFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t target_entry_index,
    bool write_after_unlock,
    bool expect_issue) {
    const experimental::NodeCoord core = {0, 0};
    auto virtual_core = mesh_device->worker_core_from_logical_core(core);

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

    const experimental::DataMovementHardwareConfig dm_producer_cfg =
        experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0};
    const experimental::DataMovementHardwareConfig dm_consumer_cfg =
        experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};

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
                  "write_after_unlock"}},
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
         {"write_after_unlock", static_cast<uint32_t>(write_after_unlock)}});
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

    if (expect_issue) {
        std::vector<NOCDebugIssueType> locked_issues;
        for (IDevice* device : mesh_device->get_devices()) {
            auto issues = fixture->get_write_to_locked_issues(device->id(), virtual_core, 0);
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
    } else {
        for (IDevice* device : mesh_device->get_devices()) {
            EXPECT_FALSE(fixture->has_write_to_locked_issue(device->id(), virtual_core, 0))
                << "Unexpected write-to-locked-DFB issue; the write did not overlap a held entry.";
        }
    }
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

    const experimental::DataMovementHardwareConfig dm_rv0 =
        experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0};
    const experimental::DataMovementHardwareConfig dm_rv1 =
        experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};

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

}  // namespace

// Write into a MIDDLE entry of the ring while the lock is held -> WRITE_TO_LOCKED_DFB.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBIssue) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this, mesh_device, /*target_entry_index=*/2, /*write_after_unlock=*/false, /*expect_issue=*/true);
    }
}

// Lock held (whole ring), but the write targets just PAST the ring -> no issue (ring boundary respected).
// target_entry_index == num_entries (4) lands at fifo_limit, the first byte outside the locked region.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBNoIssueSpatial) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this, mesh_device, /*target_entry_index=*/4, /*write_after_unlock=*/false, /*expect_issue=*/false);
    }
}

// Write into the locked entry after release -> no issue.
TEST_F(NOCDebuggingFixture, ScopedLockConcurrentAccessDFBNoIssueTemporal) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogMetal, "Running on mesh device {}", mesh_device->id());
        if (!this->dfb_scoped_lock_tracker_supported(mesh_device)) {
            GTEST_SKIP() << "DFB scoped-lock tracker not yet brought up on this arch (#45918)";
        }
        run_dfb_scoped_lock_test(
            this, mesh_device, /*target_entry_index=*/0, /*write_after_unlock=*/true, /*expect_issue=*/false);
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

}  // namespace tt::tt_metal
