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

TEST_F(CommandQueueSingleCardTraceFixture, TensixTestSubDeviceTraceBasicPrograms) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    uint32_t num_iters = 5;
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    device->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(device, sub_device_1, sub_device_2);

    // Compile the programs
    EnqueueProgram(device->command_queue(), waiter_program, false);
    device->set_sub_device_stall_group({SubDeviceId{0}});
    // Test blocking on one sub-device
    EnqueueProgram(device->command_queue(), syncer_program, true);
    EnqueueProgram(device->command_queue(), incrementer_program, false);
    device->reset_sub_device_stall_group();
    Synchronize(device);

    // Capture the trace
    auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
    EnqueueProgram(device->command_queue(), waiter_program, false);
    EnqueueProgram(device->command_queue(), syncer_program, false);
    EnqueueProgram(device->command_queue(), incrementer_program, false);
    EndTraceCapture(device, device->command_queue().id(), tid_1);

    auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
    EnqueueProgram(device->command_queue(), syncer_program, false);
    EnqueueProgram(device->command_queue(), incrementer_program, false);
    EndTraceCapture(device, device->command_queue().id(), tid_2);

    // Capture trace on one sub-device while another sub-device is running a program
    EnqueueProgram(device->command_queue(), waiter_program, false);
    device->set_sub_device_stall_group({SubDeviceId{0}});
    auto tid_3 = BeginTraceCapture(device, device->command_queue().id());
    EnqueueProgram(device->command_queue(), syncer_program, false);
    EnqueueProgram(device->command_queue(), incrementer_program, false);
    EndTraceCapture(device, device->command_queue().id(), tid_3);
    EnqueueProgram(device->command_queue(), syncer_program, false);
    EnqueueProgram(device->command_queue(), incrementer_program, false);
    device->reset_sub_device_stall_group();
    Synchronize(device);

    for (uint32_t i = 0; i < num_iters; i++) {
        // Regular program execution
        EnqueueProgram(device->command_queue(), waiter_program, false);
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);

        // Full trace execution
        ReplayTrace(device, device->command_queue().id(), tid_1, false);

        // Partial trace execution
        EnqueueProgram(device->command_queue(), waiter_program, false);
        ReplayTrace(device, device->command_queue().id(), tid_2, false);

        EnqueueProgram(device->command_queue(), waiter_program, false);
        ReplayTrace(device, device->command_queue().id(), tid_3, false);
    }
    Synchronize(device);
    detail::DumpDeviceProfileResults(device);
}

TEST_F(CommandQueueSingleCardTraceFixture, TensixActiveEthTestSubDeviceTraceBasicEthPrograms) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    uint32_t num_iters = 5;
    if (!does_device_have_active_eth_cores(device)) {
        GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
    }
    auto eth_core = *device->get_active_ethernet_cores(true).begin();
    SubDevice sub_device_2(std::array{
        CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})}),
        CoreRangeSet(CoreRange(eth_core, eth_core))});
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    device->load_sub_device_manager(sub_device_manager);
    const auto erisc_count =
        tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(HalProgrammableCoreType::ACTIVE_ETH);
    for (int erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
        log_info(tt::LogTest, "Test active ethernet DM{}", erisc_idx);
        auto [waiter_program, syncer_program, incrementer_program, global_sem] = create_basic_eth_sync_program(
            device, sub_device_1, sub_device_2, static_cast<DataMovementProcessor>(erisc_idx));

        // Compile the programs
        EnqueueProgram(device->command_queue(), waiter_program, false);
        device->set_sub_device_stall_group({SubDeviceId{0}});
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        device->reset_sub_device_stall_group();
        Synchronize(device);

        // Capture the trace
        auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), waiter_program, false);
        EnqueueProgram(device->command_queue(), syncer_program, false);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        EndTraceCapture(device, device->command_queue().id(), tid_1);

        auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), syncer_program, false);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        EndTraceCapture(device, device->command_queue().id(), tid_2);

        for (uint32_t i = 0; i < num_iters; i++) {
            // Regular program execution
            EnqueueProgram(device->command_queue(), waiter_program, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program, true);
            EnqueueProgram(device->command_queue(), incrementer_program, false);

            // Full trace execution
            ReplayTrace(device, device->command_queue().id(), tid_1, false);

            // Partial trace execution
            EnqueueProgram(device->command_queue(), waiter_program, false);
            ReplayTrace(device, device->command_queue().id(), tid_2, false);
        }
        Synchronize(device);
    }
}

TEST_F(CommandQueueSingleCardTraceFixture, TensixActiveEthTestSubDeviceTraceProgramsReconfigureSubDevices) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::array{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    SubDevice sub_device_3(std::array{CoreRangeSet(std::array{CoreRange({2, 4}, {3, 4}), CoreRange({5, 1}, {6, 3})})});
    uint32_t num_iters = 5;
    if (!does_device_have_active_eth_cores(device)) {
        GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
    }
    auto eth_core = *device->get_active_ethernet_cores(true).begin();
    SubDevice sub_device_4(std::array{
        CoreRangeSet(std::array{CoreRange({2, 1}, {2, 2}), CoreRange({1, 5}, {5, 5})}),
        CoreRangeSet(CoreRange(eth_core, eth_core))});

    auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_3, sub_device_4}, 3200);

    device->load_sub_device_manager(sub_device_manager_1);

    auto [waiter_program_1, syncer_program_1, incrementer_program_1, global_sem_1] =
        create_basic_sync_program(device, sub_device_1, sub_device_2);

    // Compile the programs
    EnqueueProgram(device->command_queue(), waiter_program_1, false);
    device->set_sub_device_stall_group({SubDeviceId{0}});
    EnqueueProgram(device->command_queue(), syncer_program_1, false);
    EnqueueProgram(device->command_queue(), incrementer_program_1, false);
    device->reset_sub_device_stall_group();
    Synchronize(device);

    // Capture the trace
    auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
    EnqueueProgram(device->command_queue(), waiter_program_1, false);
    EnqueueProgram(device->command_queue(), syncer_program_1, false);
    EnqueueProgram(device->command_queue(), incrementer_program_1, false);
    EndTraceCapture(device, device->command_queue().id(), tid_1);

    auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
    EnqueueProgram(device->command_queue(), syncer_program_1, false);
    EnqueueProgram(device->command_queue(), incrementer_program_1, false);
    EndTraceCapture(device, device->command_queue().id(), tid_2);

    device->load_sub_device_manager(sub_device_manager_2);

    const auto erisc_count =
        tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(HalProgrammableCoreType::ACTIVE_ETH);
    for (int erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
        log_info(tt::LogTest, "Test active ethernet DM{}", erisc_idx);
        auto [waiter_program_2, syncer_program_2, incrementer_program_2, global_sem_2] = create_basic_eth_sync_program(
            device, sub_device_3, sub_device_4, static_cast<DataMovementProcessor>(erisc_idx));

        // Compile the programs
        EnqueueProgram(device->command_queue(), waiter_program_2, false);
        device->set_sub_device_stall_group({SubDeviceId{0}});
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        device->reset_sub_device_stall_group();
        Synchronize(device);

        // Capture the trace
        auto tid_3 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), waiter_program_2, false);
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        EndTraceCapture(device, device->command_queue().id(), tid_3);

        auto tid_4 = BeginTraceCapture(device, device->command_queue().id());
        EnqueueProgram(device->command_queue(), syncer_program_2, false);
        EnqueueProgram(device->command_queue(), incrementer_program_2, false);
        EndTraceCapture(device, device->command_queue().id(), tid_4);

        for (uint32_t i = 0; i < num_iters; i++) {
            device->load_sub_device_manager(sub_device_manager_1);
            // Regular program execution
            EnqueueProgram(device->command_queue(), waiter_program_1, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program_1, false);
            EnqueueProgram(device->command_queue(), incrementer_program_1, false);

            // Full trace execution
            ReplayTrace(device, device->command_queue().id(), tid_1, false);

            // Partial trace execution
            EnqueueProgram(device->command_queue(), waiter_program_1, false);
            ReplayTrace(device, device->command_queue().id(), tid_2, false);

            device->load_sub_device_manager(sub_device_manager_2);
            // Regular program execution
            EnqueueProgram(device->command_queue(), waiter_program_2, false);
            // Test blocking on one sub-device
            EnqueueProgram(device->command_queue(), syncer_program_2, false);
            EnqueueProgram(device->command_queue(), incrementer_program_2, false);

            // Full trace execution
            ReplayTrace(device, device->command_queue().id(), tid_3, false);

            // Partial trace execution
            EnqueueProgram(device->command_queue(), waiter_program_2, false);
            ReplayTrace(device, device->command_queue().id(), tid_4, false);
        }
    }
    Synchronize(device);
}

TEST_F(CommandQueueSingleCardTraceFixture, TensixTestSubDeviceIllegalOperations) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});

    // Assert no idle eth cores specified
    EXPECT_THROW(
        SubDevice sub_device_3(std::array{
            CoreRangeSet(CoreRange({3, 3}, {3, 3})),
            CoreRangeSet(CoreRange({4, 4}, {4, 4})),
            CoreRangeSet(CoreRange({5, 5}, {5, 5}))}),
        std::exception);
    auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_2, sub_device_1}, 3200);
    device->load_sub_device_manager(sub_device_manager_1);

    auto [waiter_program_1, syncer_program_1, incrementer_program_1, global_sem_1] =
        create_basic_sync_program(device, sub_device_1, sub_device_2);

    // Compile the programs
    EnqueueProgram(device->command_queue(), waiter_program_1, false);
    device->set_sub_device_stall_group({SubDeviceId{0}});
    // Test blocking on one sub-device
    EnqueueProgram(device->command_queue(), syncer_program_1, false);
    EnqueueProgram(device->command_queue(), incrementer_program_1, false);
    device->reset_sub_device_stall_group();
    Synchronize(device);

    // Capture the trace
    auto tid_1 = BeginTraceCapture(device, device->command_queue().id());
    // Can not load a sub-device manager while tracing
    EXPECT_THROW(device->load_sub_device_manager(sub_device_manager_2), std::exception);
    EnqueueProgram(device->command_queue(), waiter_program_1, false);
    EnqueueProgram(device->command_queue(), syncer_program_1, false);
    EnqueueProgram(device->command_queue(), incrementer_program_1, false);
    EndTraceCapture(device, device->command_queue().id(), tid_1);

    device->load_sub_device_manager(sub_device_manager_2);
    auto [waiter_program_2, syncer_program_2, incrementer_program_2, global_sem_2] =
        create_basic_sync_program(device, sub_device_2, sub_device_1);

    EnqueueProgram(device->command_queue(), waiter_program_2, false);
    device->set_sub_device_stall_group({SubDeviceId{0}});
    EnqueueProgram(device->command_queue(), syncer_program_2, false);
    EnqueueProgram(device->command_queue(), incrementer_program_2, false);
    device->reset_sub_device_stall_group();
    Synchronize(device);

    auto tid_2 = BeginTraceCapture(device, device->command_queue().id());
    EnqueueProgram(device->command_queue(), waiter_program_2, false);
    EnqueueProgram(device->command_queue(), syncer_program_2, false);
    EnqueueProgram(device->command_queue(), incrementer_program_2, false);
    EndTraceCapture(device, device->command_queue().id(), tid_2);

    // Full trace execution
    ReplayTrace(device, device->command_queue().id(), tid_2, false);

    // Can not replay a trace on a different sub-device manager
    EXPECT_THROW(ReplayTrace(device, device->command_queue().id(), tid_1, false), std::exception);

    Synchronize(device);

    device->remove_sub_device_manager(sub_device_manager_1);
    EXPECT_THROW(device->load_sub_device_manager(sub_device_manager_1), std::exception);
}

}  // namespace tt::tt_metal
