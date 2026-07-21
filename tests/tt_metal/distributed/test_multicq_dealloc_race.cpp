// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Integration test for the multi-CQ deallocation race (#43725).
//
// Background:
//   When multiple command queues are in use, a MeshBuffer's pending-event
//   deallocation path must tolerate quiesced CQs. Specifically:
//     - CQ0 and CQ1 record completion events on a buffer via add_pending_event.
//     - CQ1 is quiesced (finish_and_reset_in_use drains and resets counters)
//       while its pending-event slot remains populated.
//     - Buffer deallocation calls wait_for_pending_events(), which must not
//       spin forever on the quiesced CQ.
//
//   Without the fix, EventSynchronize/EventQuery could spin on a CQ whose
//   event counters were reset to 0 by quiesce, causing a hang.

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <thread>
#include <unistd.h>  // alarm()
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "device.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "tt_metal/distributed/mesh_command_queue_base.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

// Fixture: mesh device with 2 CQs.
// A 30-second SIGALRM watchdog is installed per-test (see SetUp/TearDown) so
// that a hang causes the process to abort rather than block the CI runner.
class MultiCQDeallocRaceFixture : public MeshDeviceFixtureBase {
protected:
    MultiCQDeallocRaceFixture() : MeshDeviceFixtureBase(Config{.num_cqs = 2}) {}

    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        // Arm a 30-second watchdog: if the test hangs the process is killed.
        alarm(30);
    }

    void TearDown() override {
        alarm(0);  // Disarm watchdog on clean exit.
        MeshDeviceFixtureBase::TearDown();
    }
};

// ---------------------------------------------------------------------------
// TestMultiCQDeallocRace
//
// Steps:
//   1. Allocate a replicated MeshBuffer.
//   2. CQ0 and CQ1: write data to the buffer (async), record host-visible events,
//      and register both as pending events on the buffer.
//   3. CQ1: quiesce via finish_and_reset_in_use() — this drains CQ1, resets
//      its event counters, and leaves a stale pending event for deallocation.
//   4. Deallocate the buffer — wait_for_pending_events() must complete
//      without spinning forever on the quiesced CQ.
//
// Pass = completes in <30s.
// Fail = hang (watchdog kills at 30s) or crash.
// ---------------------------------------------------------------------------
TEST_F(MultiCQDeallocRaceFixture, TestMultiCQDeallocRace) {
    const uint32_t page_size = 1024;  // 1 KB
    const uint32_t num_pages = 64;

    // Step 1: Allocate a replicated buffer.
    DeviceLocalBufferConfig per_device_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    ReplicatedBufferConfig global_config = {
        .size = num_pages * page_size,
    };

    auto mesh_buf = MeshBuffer::create(global_config, per_device_config, mesh_device_.get());
    log_info(tt::LogTest, "[MultiCQDeallocRace] Buffer allocated");

    // Step 2: CQ0 — write data and record a completion event.
    {
        auto& cq0 = mesh_device_->mesh_command_queue(0);
        auto registration = mesh_buf->try_acquire_pending_event_registration();
        ASSERT_TRUE(registration.has_value());
        std::vector<uint32_t> src(num_pages * page_size / sizeof(uint32_t));
        std::iota(src.begin(), src.end(), 0xBEEF0000u);

        EnqueueWriteMeshBuffer(cq0, mesh_buf, src, /*blocking=*/false);

        // Record an event on CQ0 and attach it to the buffer as a pending event.
        // This is the event that wait_for_pending_events() will need to synchronize.
        auto event = cq0.enqueue_record_event_to_host();
        registration->publish(event);
        log_info(tt::LogTest, "[MultiCQDeallocRace] CQ0: write + event recorded (event_id={})", event.id());
    }

    // Step 3: CQ1 — record a pending event, then quiesce. This drains CQ1 and
    // leaves wait_for_pending_events() with a non-empty stale CQ1 slot.
    {
        auto& cq1 = mesh_device_->impl().mesh_command_queue_base(1);
        auto registration = mesh_buf->try_acquire_pending_event_registration();
        ASSERT_TRUE(registration.has_value());
        std::vector<uint32_t> dummy(num_pages * page_size / sizeof(uint32_t), 0xDEAD);
        EnqueueWriteMeshBuffer(cq1, mesh_buf, dummy, /*blocking=*/false);
        auto event = cq1.enqueue_record_event_to_host();
        registration->publish(event);
        log_info(tt::LogTest, "[MultiCQDeallocRace] CQ1: write + event recorded (event_id={})", event.id());

        // Quiesce CQ1: drain outstanding work and reset event counters.
        cq1.wait_for_completion(/*reset_launch_msg_state=*/false);
        cq1.finish_and_reset_in_use();
        for (auto* device : mesh_device_->get_view().get_devices()) {
            device->sysmem_manager().set_current_and_last_completed_event(
                /*cq_id=*/1, /*current=*/0, /*last_completed=*/0);
        }
        log_info(tt::LogTest, "[MultiCQDeallocRace] CQ1: quiesced with stale pending event");
    }

    // Step 4: Deallocate the buffer.
    // wait_for_pending_events() will:
    //   - See CQ0's pending event and synchronize it (CQ0 is still in-use).
    //   - See CQ1's pending event but skip it because CQ1 was quiesced.
    // Without the fix, this would hang if EventSynchronize/EventQuery spun on
    // a quiesced CQ whose counters were reset.
    ASSERT_TRUE(mesh_buf->has_pending_events()) << "Buffer should have pending events from CQ0 and CQ1";
    log_info(tt::LogTest, "[MultiCQDeallocRace] Deallocating buffer with pending events...");
    mesh_buf.reset();  // Trigger destructor -> deallocate() -> wait_for_pending_events()
    log_info(tt::LogTest, "[MultiCQDeallocRace] Buffer deallocated — no hang, test passed");
}

// ---------------------------------------------------------------------------
// TestEventQueryQuiesced
//
// Verify that EventQuery returns true for events on a quiesced CQ
// **specifically because of the quiesced short-circuit**, not because
// last_completed_event >= event_id happens to be true.
//
// After quiescing, finish_and_reset_in_use sets last_completed = UINT32_MAX,
// which satisfies the counter check for any event_id.  To eliminate this
// false-positive path, we manually reset last_completed to 0 after quiescing
// (simulating the start of a new cycle with no completed events).  Now:
//   - quiesced=true → EventQuery short-circuits to true (the path we test)
//   - Without quiesced check: last_completed(0) >= event_id → false (fail)
// ---------------------------------------------------------------------------
TEST_F(MultiCQDeallocRaceFixture, TestEventQueryQuiesced) {
    const uint32_t page_size = 1024;
    const uint32_t num_pages = 16;

    DeviceLocalBufferConfig per_device_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    ReplicatedBufferConfig global_config = {
        .size = num_pages * page_size,
    };

    auto mesh_buf = MeshBuffer::create(global_config, per_device_config, mesh_device_.get());

    // Record an event on CQ0.
    auto& cq0 = mesh_device_->impl().mesh_command_queue_base(0);
    std::vector<uint32_t> src(num_pages * page_size / sizeof(uint32_t), 0xCAFE);
    EnqueueWriteMeshBuffer(cq0, mesh_buf, src, /*blocking=*/false);
    auto event = cq0.enqueue_record_event_to_host();
    log_info(tt::LogTest, "[EventQueryQuiesced] CQ0: event recorded (id={})", event.id());
    ASSERT_GT(event.id(), 0u) << "Event ID should be > 0 for this test to be meaningful";

    // Wait for the event to complete, then quiesce CQ0.
    EventSynchronize(event);
    cq0.wait_for_completion(/*reset_launch_msg_state=*/false);
    cq0.finish_and_reset_in_use();
    log_info(tt::LogTest, "[EventQueryQuiesced] CQ0: quiesced (last_completed=UINT32_MAX)");

    // Manually reset last_completed_event to 0 on all devices for CQ0.
    // This simulates the counter state at the START of a new cycle (after quiesce
    // reset counters, before any new events complete).  The quiesced flag remains true
    // because we haven't called mark_in_use().
    //
    // Without this reset, last_completed=UINT32_MAX always satisfies >= event_id,
    // making the test pass even if the quiesced check were removed (false positive).
    const uint8_t cq_id = 0;
    for (auto* device : mesh_device_->get_view().get_devices()) {
        device->sysmem_manager().set_current_and_last_completed_event(cq_id, /*current=*/0, /*last_completed=*/0);
    }
    log_info(tt::LogTest, "[EventQueryQuiesced] Manually reset last_completed to 0 — counter check alone would fail");

    // EventQuery on the old event should return true — the quiesced flag is
    // the ONLY reason this succeeds.  If the quiesced check were removed,
    // last_completed(0) >= event_id(>0) would be false.
    bool completed = EventQuery(event);
    EXPECT_TRUE(completed) << "EventQuery should return true via quiesced short-circuit, not counter check";
    log_info(tt::LogTest, "[EventQueryQuiesced] EventQuery returned {} — expected true (via quiesced path)", completed);
}

// ---------------------------------------------------------------------------
// TestEventSynchronizeEpochShortCircuit
//
// Verify that EventSynchronize does NOT hang for a stale (pre-quiesce) event
// AFTER the CQ has been reactivated via mark_in_use() — i.e. once is_quiesced
// has been cleared and the only thing protecting the waiter is the
// quiesce_epoch comparison.
//
// Sequence:
//   1. Record event E on CQ0 (quiesce_epoch captured = 0).
//   2. Wait for E, then quiesce CQ0 (epoch -> 1, is_quiesced=true, counters reset).
//   3. Reactivate CQ0 by submitting new work (mark_in_use clears is_quiesced,
//      but quiesce_epoch stays 1 — only finish_and_reset_in_use advances it).
//   4. Reset last_completed_event to 0 so the counter check alone would spin
//      forever on E's stale event_id.
//   5. EventSynchronize(E) must return promptly: the epoch short-circuit
//      (E.quiesce_epoch()=0 < cq.quiesce_epoch()=1) is the ONLY thing that
//      prevents an infinite spin here. Without the epoch check, the reactivated
//      CQ (is_quiesced=false) with last_completed=0 < event_id spins forever.
//
// Pass = completes in <30s. Fail = hang (watchdog aborts at 30s).
// ---------------------------------------------------------------------------
TEST_F(MultiCQDeallocRaceFixture, TestEventSynchronizeEpochShortCircuit) {
    const uint32_t page_size = 1024;
    const uint32_t num_pages = 16;

    DeviceLocalBufferConfig per_device_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    ReplicatedBufferConfig global_config = {
        .size = num_pages * page_size,
    };

    auto mesh_buf = MeshBuffer::create(global_config, per_device_config, mesh_device_.get());
    auto& cq0 = mesh_device_->impl().mesh_command_queue_base(0);

    // Step 1: record event E (epoch 0).
    std::vector<uint32_t> src(num_pages * page_size / sizeof(uint32_t), 0xF00D);
    EnqueueWriteMeshBuffer(cq0, mesh_buf, src, /*blocking=*/false);
    auto event = cq0.enqueue_record_event_to_host();
    log_info(tt::LogTest, "[EpochShortCircuit] E recorded (id={}, epoch={})", event.id(), event.quiesce_epoch());
    ASSERT_GT(event.id(), 0u) << "Event ID must be > 0 for the counter path to spin";

    // Step 2: complete + quiesce CQ0.
    EventSynchronize(event);
    cq0.wait_for_completion(/*reset_launch_msg_state=*/false);
    cq0.finish_and_reset_in_use();
    log_info(tt::LogTest, "[EpochShortCircuit] CQ0 quiesced (epoch now {})", cq0.quiesce_epoch());

    // Step 3: reactivate CQ0 by submitting new work — mark_in_use() clears
    // is_quiesced but does NOT advance quiesce_epoch.
    std::vector<uint32_t> next(num_pages * page_size / sizeof(uint32_t), 0xBAAD);
    EnqueueWriteMeshBuffer(cq0, mesh_buf, next, /*blocking=*/false);
    auto reactivating_event = cq0.enqueue_record_event_to_host();
    EventSynchronize(reactivating_event);
    cq0.wait_for_completion(/*reset_launch_msg_state=*/false);
    log_info(
        tt::LogTest, "[EpochShortCircuit] CQ0 reactivated (is_quiesced cleared, epoch still {})", cq0.quiesce_epoch());

    // Step 4: reset last_completed to 0 so the counter path cannot satisfy E.
    const uint8_t cq_id = 0;
    for (auto* device : mesh_device_->get_view().get_devices()) {
        device->sysmem_manager().set_current_and_last_completed_event(cq_id, /*current=*/0, /*last_completed=*/0);
    }

    // Step 5: EventSynchronize(E) must return via the epoch short-circuit.
    // is_quiesced is false (reactivated), last_completed is 0 (< event_id), so
    // ONLY event.quiesce_epoch()(0) < cq.quiesce_epoch()(1) saves us from a hang.
    EventSynchronize(event);
    EXPECT_TRUE(EventQuery(event)) << "Epoch short-circuit should report stale pre-quiesce event complete";
    log_info(tt::LogTest, "[EpochShortCircuit] EventSynchronize/Query on stale E completed via epoch path");
}

// A publisher acquired before dispatch must keep explicit deallocation blocked
// until its completion event is published. This deterministically covers both
// workload-enqueue→event-record and event-record→publication windows.
TEST_F(MultiCQDeallocRaceFixture, TestDeallocationWaitsForEventPublisher) {
    constexpr uint32_t page_size = 1024;
    constexpr uint32_t num_pages = 16;
    DeviceLocalBufferConfig per_device_config{
        .page_size = page_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    ReplicatedBufferConfig global_config = {
        .size = num_pages * page_size,
    };

    auto mesh_buf = MeshBuffer::create(global_config, per_device_config, mesh_device_.get());
    auto registration = mesh_buf->try_acquire_pending_event_registration();
    ASSERT_TRUE(registration.has_value());

    auto& cq0 = mesh_device_->mesh_command_queue(0);
    std::vector<uint32_t> src(num_pages * page_size / sizeof(uint32_t), 0xACED);
    EnqueueWriteMeshBuffer(cq0, mesh_buf, src, /*blocking=*/false);

    std::atomic<bool> deallocation_finished = false;
    std::thread deallocator([&] {
        mesh_buf->deallocate();
        deallocation_finished.store(true, std::memory_order_release);
    });

    bool observed_closed_gate = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        auto probe = mesh_buf->try_acquire_pending_event_registration();
        if (!probe.has_value()) {
            observed_closed_gate = true;
            break;
        }
        std::this_thread::yield();
    }

    EXPECT_TRUE(observed_closed_gate) << "Deallocation did not close the event-registration gate";
    EXPECT_FALSE(deallocation_finished.load(std::memory_order_acquire))
        << "Deallocation released the buffer while an event publisher was active";

    auto event = cq0.enqueue_record_event_to_host();
    registration->publish(event);
    deallocator.join();

    EXPECT_TRUE(deallocation_finished.load(std::memory_order_acquire));
    EXPECT_FALSE(mesh_buf->is_allocated());
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
