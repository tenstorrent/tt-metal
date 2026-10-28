// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Queue fences (tt-metalium/experimental/queue_fence.hpp): a queue waits for exactly the programs a fence names.
// Programs are held on a semaphore until the host releases them, so every check is deterministic.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/queue_fence.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::GlobalSemaphore;
using tt::tt_metal::SubDevice;
using tt::tt_metal::SubDeviceId;
using tt::tt_metal::distributed::MeshCommandQueue;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshEvent;
using tt::tt_metal::distributed::MeshWorkload;
using tt::tt_metal::experimental::EnqueueWaitForQueueFence;
using tt::tt_metal::experimental::GetQueueFence;

// Sub-device i is the single core {i, 0}.
constexpr uint32_t k_num_sub_devices = 3;

class QueueFenceTest : public MultiCommandQueueSingleDeviceFixture {
protected:
    void SetUp() override {
        MultiCommandQueueSingleDeviceFixture::SetUp();
        if (device_ == nullptr) {
            return;
        }
        std::vector<SubDevice> sub_devices;
        for (uint32_t i = 0; i < k_num_sub_devices; ++i) {
            sub_devices.push_back(SubDevice(std::array{CoreRangeSet(CoreRange(core(i), core(i)))}));
        }
        manager_ = device_->create_sub_device_manager(sub_devices, 0);
        device_->load_sub_device_manager(manager_);
    }

    void TearDown() override {
        if (device_ != nullptr) {
            device_->clear_loaded_sub_device_manager();
            device_->remove_sub_device_manager(manager_);
        }
        MultiCommandQueueSingleDeviceFixture::TearDown();
    }

    static CoreCoord core(uint32_t sub_device) { return CoreCoord(sub_device, 0); }

    MeshCommandQueue& queue(uint8_t id) { return device_->mesh_command_queue(id); }

    MeshWorkload workload(tt::tt_metal::Program program) {
        MeshWorkload workload;
        workload.add_program(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)), std::move(program));
        return workload;
    }

    // A program on the sub-device's core that runs until release(semaphore).
    MeshWorkload held(uint32_t sub_device, const GlobalSemaphore& semaphore) {
        auto program = tt::tt_metal::CreateProgram();
        auto kernel = tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
            CoreRangeSet(CoreRange(core(sub_device), core(sub_device))),
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        tt::tt_metal::SetRuntimeArgs(program, kernel, core(sub_device), {static_cast<uint32_t>(semaphore.address())});
        return workload(std::move(program));
    }

    MeshWorkload blank(uint32_t sub_device) {
        auto program = tt::tt_metal::CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            CoreRangeSet(CoreRange(core(sub_device), core(sub_device))),
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        return workload(std::move(program));
    }

    GlobalSemaphore semaphore(uint32_t sub_device) {
        return tt::tt_metal::CreateGlobalSemaphore(
            *device_, CoreRangeSet(CoreRange(core(sub_device), core(sub_device))), 0);
    }

    void release(uint32_t sub_device, const GlobalSemaphore& semaphore) {
        std::vector<uint32_t> one = {1};
        tt::tt_metal::detail::WriteToDeviceL1(
            device_->get_devices()[0], core(sub_device), static_cast<uint32_t>(semaphore.address()), one);
    }

    // A program's first launch uploads its binaries with a buffer write, which waits for the queue's programs on every
    // sub-device of the stall group. Launching each program once beforehand keeps that wait out of the sequence under
    // test. A held program's semaphore is released in advance, so the warm-up launch returns at once.
    void warm_up(uint8_t queue_id, MeshWorkload& workload) {
        EnqueueMeshWorkload(queue(queue_id), workload, false);
        tt::tt_metal::distributed::Finish(queue(queue_id));
    }
    void warm_up(uint8_t queue_id, MeshWorkload& workload, uint32_t sub_device, const GlobalSemaphore& semaphore) {
        release(sub_device, semaphore);
        warm_up(queue_id, workload);
    }

    // Whether the event completes within the time a running device needs for a few blank programs.
    static bool completes(const MeshEvent& event) {
        for (int i = 0; i < 100; ++i) {
            if (tt::tt_metal::distributed::EventQuery(event)) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }

    tt::tt_metal::SubDeviceManagerId manager_{0};
};

// The hazard fences exist for: a queue sends a program on one sub-device ahead of an earlier one still running on
// another.
TEST_F(QueueFenceTest, OneQueueDoesNotOrderItsSubDevices) {
    auto held_semaphore = semaphore(1);
    auto producer = held(1, held_semaphore);
    auto consumer = blank(0);
    warm_up(0, producer, 1, held_semaphore);
    warm_up(0, consumer);
    EnqueueMeshWorkload(queue(0), producer, false);
    EnqueueMeshWorkload(queue(0), consumer, false);
    auto consumer_done = queue(0).enqueue_record_event_to_host(std::vector{SubDeviceId{0}});
    EXPECT_TRUE(completes(consumer_done));
    release(1, held_semaphore);
    tt::tt_metal::distributed::Finish(queue(0));
}

TEST_F(QueueFenceTest, AFenceOnTheSameQueueWaitsForItsSubDeviceOnly) {
    auto p_semaphore = semaphore(1);
    auto q_semaphore = semaphore(2);
    auto p = held(1, p_semaphore);
    auto q = held(2, q_semaphore);
    auto consumer = blank(0);
    warm_up(0, p, 1, p_semaphore);
    warm_up(0, q, 2, q_semaphore);
    warm_up(0, consumer);
    EnqueueMeshWorkload(queue(0), p, false);
    auto fence = GetQueueFence(queue(0), SubDeviceId{1});
    EnqueueMeshWorkload(queue(0), q, false);
    EnqueueWaitForQueueFence(queue(0), fence);
    EnqueueMeshWorkload(queue(0), consumer, false);
    auto consumer_done = queue(0).enqueue_record_event_to_host(std::vector{SubDeviceId{0}});
    auto q_done = queue(0).enqueue_record_event_to_host(std::vector{SubDeviceId{2}});

    EXPECT_FALSE(completes(consumer_done));  // P is held
    release(1, p_semaphore);
    EXPECT_TRUE(completes(consumer_done));  // while Q is still held
    EXPECT_FALSE(tt::tt_metal::distributed::EventQuery(q_done));
    release(2, q_semaphore);
    tt::tt_metal::distributed::EventSynchronize(q_done);
}

TEST_F(QueueFenceTest, AFenceFromAnotherQueueWaitsForItsProgramsOnly) {
    auto p_semaphore = semaphore(1);
    auto q_semaphore = semaphore(1);
    auto p = held(1, p_semaphore);
    auto q = held(1, q_semaphore);
    auto consumer = blank(0);
    warm_up(0, p, 1, p_semaphore);
    warm_up(0, q, 1, q_semaphore);
    warm_up(1, consumer);
    EnqueueMeshWorkload(queue(0), p, false);
    auto fence = GetQueueFence(queue(0), SubDeviceId{1});
    EnqueueMeshWorkload(queue(0), q, false);  // after P on the same sub-device: an event recorded now covers it too
    EnqueueWaitForQueueFence(queue(1), fence);
    EnqueueMeshWorkload(queue(1), consumer, false);
    auto consumer_done = queue(1).enqueue_record_event_to_host(std::vector{SubDeviceId{0}});
    auto q_done = queue(0).enqueue_record_event_to_host(std::vector{SubDeviceId{1}});

    EXPECT_FALSE(completes(consumer_done));  // P is held
    release(1, p_semaphore);
    EXPECT_TRUE(completes(consumer_done));  // while Q is still held
    EXPECT_FALSE(tt::tt_metal::distributed::EventQuery(q_done));
    release(1, q_semaphore);
    tt::tt_metal::distributed::EventSynchronize(q_done);
    tt::tt_metal::distributed::Finish(queue(1));
}

// Reloading the sub-device manager restarts the worker counters; waiting for a fence from before must not wait for its
// old count again, which the restarted counter would never reach.
TEST_F(QueueFenceTest, AFenceFromBeforeTheCountersRestartedHasCompleted) {
    auto program = blank(1);
    for (int i = 0; i < 4; ++i) {
        EnqueueMeshWorkload(queue(0), program, false);
    }
    auto fence = GetQueueFence(queue(0), SubDeviceId{1});
    ASSERT_GT(fence.count, 0u);
    tt::tt_metal::distributed::Finish(queue(0));
    device_->clear_loaded_sub_device_manager();
    device_->load_sub_device_manager(manager_);

    EnqueueWaitForQueueFence(queue(0), fence);
    EnqueueWaitForQueueFence(queue(1), fence);
    auto done0 = queue(0).enqueue_record_event_to_host();
    auto done1 = queue(1).enqueue_record_event_to_host();
    EXPECT_TRUE(completes(done0));
    EXPECT_TRUE(completes(done1));
}

}  // namespace
}  // namespace ttnn
