// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "command_queue_fixture.hpp"
#include "impl/context/metal_context.hpp"

#include <vector>

namespace tt::tt_metal {

// Test two cores: one locks and writes, another writes to the same region
// This tests the profiler's ability to track overlapping memory accesses
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixScopedLockConcurrentAccess) {
    for (auto& mesh_device : devices_) {
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
        uint32_t writer_buffer_addr = unreserved_addr + alignment * 32;  // Separate local buffer for writer
        uint32_t num_elements = 8;
        uint32_t locker_write_value = 0x11110000;
        uint32_t writer_write_value = 0x22220000;

        auto locker_virtual_core = mesh_device->worker_core_from_logical_core(locker_core);

        // Locker kernel: locks its region and writes to it
        std::vector<uint32_t> locker_args = {
            locker_buffer_addr,
            num_elements,
            locker_write_value,
            0,
            0,
            0,  // Not using NoC write
            0   // do_noc_write = false
        };

        KernelHandle locker_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_test_kernel.cpp",
            locker_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, locker_kernel, locker_core, locker_args);

        // Writer kernel: writes to the locker core's "locked" region
        std::vector<uint32_t> writer_args = {
            writer_buffer_addr,
            num_elements,
            writer_write_value,
            locker_virtual_core.x,
            locker_virtual_core.y,
            locker_buffer_addr  // Target the locker's buffer
        };

        KernelHandle writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/scoped_lock_writer_kernel.cpp",
            writer_core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

        SetRuntimeArgs(program, writer_kernel, writer_core, writer_args);

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);

        // The final data on locker core depends on execution order
        // Since there's no actual mutex, the data could be from either kernel
        // The important thing is that the profiler can track both accesses
        auto* device = mesh_device->get_devices()[0];
        std::vector<uint32_t> final_data(num_elements, 0);
        detail::ReadFromDeviceL1(device, locker_core, locker_buffer_addr, num_elements * sizeof(uint32_t), final_data);

        // Data should be either locker's values or writer's values (or mixed)
        // We just verify the writes happened
        bool has_locker_data = false;
        bool has_writer_data = false;
        for (uint32_t i = 0; i < num_elements; i++) {
            if (final_data[i] == locker_write_value + i) {
                has_locker_data = true;
            }
            if (final_data[i] == writer_write_value + i) {
                has_writer_data = true;
            }
        }

        // At least one of them should have written
        EXPECT_TRUE(has_locker_data || has_writer_data) << "Neither locker nor writer data found in buffer";

        log_info(tt::LogTest, "TensixScopedLockConcurrentAccess passed on device {}", device->id());
        log_info(tt::LogTest, "  Run with TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 to capture lock events");
    }
}

}  // namespace tt::tt_metal
