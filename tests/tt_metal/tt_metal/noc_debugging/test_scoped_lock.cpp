// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
        bool found_write_to_locked = false;
        for (IDevice* device : mesh_device->get_devices()) {
            ChipId chip_id = device->id();
            if (this->has_write_to_locked_issue(chip_id, writer_virtual_core, 0)) {
                found_write_to_locked = true;
                break;
            }
        }
        EXPECT_TRUE(found_write_to_locked)
            << "Expected write-to-locked-buffer issue on writer core (1,0); NOC debug did not report the violation.";
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

        bool found_write_to_locked_cb = false;
        for (IDevice* device : mesh_device->get_devices()) {
            if (this->has_write_to_locked_issue(device->id(), writer_virtual_core, 0)) {
                found_write_to_locked_cb = true;
                break;
            }
        }
        EXPECT_TRUE(found_write_to_locked_cb)
            << "Expected write-to-locked-CB issue on writer core; NOC debug did not report the violation.";
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

}  // namespace tt::tt_metal
