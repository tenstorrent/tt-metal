// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <array>
#include <cstdint>
#include <exception>
#include <unordered_set>
#include <vector>

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "command_queue_fixture.hpp"
#include "dispatch_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt_stl/span.hpp>
#include "sub_device_test_utils.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal {

TEST_F(UnitMeshCQSingleCardTraceFixture, TensixTestSubDeviceTraceBasicPrograms) {
    auto mesh_device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    uint32_t num_iters = 5;
    auto sub_device_manager = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(mesh_device.get(), sub_device_1, sub_device_2);

    // Compile the programs
    distributed::MeshWorkload waiter_workload, syncer_workload, incrementer_workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::AddProgramToMeshWorkload(waiter_workload, std::move(waiter_program), device_range);
    distributed::AddProgramToMeshWorkload(syncer_workload, std::move(syncer_program), device_range);
    distributed::AddProgramToMeshWorkload(incrementer_workload, std::move(incrementer_program), device_range);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload, false);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});
    // Test blocking on one sub-device
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload, true);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload, false);
    mesh_device->reset_sub_device_stall_group();
    mesh_device->mesh_command_queue().finish();

    // Capture the trace
    auto tid_1 = distributed::BeginTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id());
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_1);

    auto tid_2 = distributed::BeginTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id());
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_2);

    // Capture trace on one sub-device while another sub-device is running a program
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload, false);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});
    auto tid_3 = distributed::BeginTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id());
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_3);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload, false);
    mesh_device->reset_sub_device_stall_group();
    mesh_device->mesh_command_queue().finish();

    for (uint32_t i = 0; i < num_iters; i++) {
        // Regular program execution
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload, false);
        // Test blocking on one sub-device
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload, true);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload, false);

        // Full trace execution
        distributed::ReplayTrace(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_1, false);

        // Partial trace execution
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload, false);
        distributed::ReplayTrace(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_2, false);

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload, false);
        distributed::ReplayTrace(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_3, false);
    }
    mesh_device->mesh_command_queue().finish();
    ReadMeshDeviceProfilerResults(*mesh_device, ProfilerReadState::NORMAL);
}

TEST_F(UnitMeshCQSingleCardTraceFixture, TensixTestSubDeviceIllegalOperations) {
    auto mesh_device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});

    // Assert no idle eth cores specified
    EXPECT_THROW(
        SubDevice sub_device_3(std::array{
            CoreRangeSet(CoreRange({3, 3}, {3, 3})),
            CoreRangeSet(CoreRange({4, 4}, {4, 4})),
            CoreRangeSet(CoreRange({5, 5}, {5, 5}))}),
        std::exception);
    auto sub_device_manager_1 = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    auto sub_device_manager_2 = mesh_device->create_sub_device_manager({sub_device_2, sub_device_1}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager_1);

    auto [waiter_program_1, syncer_program_1, incrementer_program_1, global_sem_1] =
        create_basic_sync_program(mesh_device.get(), sub_device_1, sub_device_2);

    // Compile the programs
    distributed::MeshWorkload waiter_workload_1, syncer_workload_1, incrementer_workload_1;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::AddProgramToMeshWorkload(waiter_workload_1, std::move(waiter_program_1), device_range);
    distributed::AddProgramToMeshWorkload(syncer_workload_1, std::move(syncer_program_1), device_range);
    distributed::AddProgramToMeshWorkload(incrementer_workload_1, std::move(incrementer_program_1), device_range);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload_1, false);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});
    // Test blocking on one sub-device
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload_1, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload_1, false);
    mesh_device->reset_sub_device_stall_group();
    mesh_device->mesh_command_queue().finish();

    // Capture the trace
    auto tid_1 = distributed::BeginTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id());
    // Can not load a sub-device manager while tracing
    EXPECT_THROW(mesh_device->load_sub_device_manager(sub_device_manager_2), std::exception);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload_1, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload_1, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload_1, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_1);

    mesh_device->load_sub_device_manager(sub_device_manager_2);
    auto [waiter_program_2, syncer_program_2, incrementer_program_2, global_sem_2] =
        create_basic_sync_program(mesh_device.get(), sub_device_2, sub_device_1);

    distributed::MeshWorkload waiter_workload_2, syncer_workload_2, incrementer_workload_2;
    distributed::AddProgramToMeshWorkload(waiter_workload_2, std::move(waiter_program_2), device_range);
    distributed::AddProgramToMeshWorkload(syncer_workload_2, std::move(syncer_program_2), device_range);
    distributed::AddProgramToMeshWorkload(incrementer_workload_2, std::move(incrementer_program_2), device_range);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload_2, false);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload_2, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload_2, false);
    mesh_device->reset_sub_device_stall_group();
    mesh_device->mesh_command_queue().finish();

    auto tid_2 = distributed::BeginTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id());
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_workload_2, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_workload_2, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_workload_2, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_2);

    // Full trace execution
    distributed::ReplayTrace(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_2, false);

    // Can not replay a trace on a different sub-device manager
    EXPECT_THROW(
        distributed::ReplayTrace(mesh_device.get(), mesh_device->mesh_command_queue().id(), tid_1, false),
        std::exception);

    mesh_device->mesh_command_queue().finish();

    mesh_device->remove_sub_device_manager(sub_device_manager_1);
    EXPECT_THROW(mesh_device->load_sub_device_manager(sub_device_manager_1), std::exception);
}

}  // namespace tt::tt_metal
