// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include <cstdint>
#include <memory>
#include <numeric>
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
        std::vector<uint32_t> src(num_pages * page_size / sizeof(uint32_t));
        std::iota(src.begin(), src.end(), 0xBEEF0000u);

        EnqueueWriteMeshBuffer(cq0, mesh_buf, src, /*blocking=*/false);

        // Record an event on CQ0 and attach it to the buffer as a pending event.
        // This is the event that wait_for_pending_events() will need to synchronize.
        auto event = cq0.enqueue_record_event_to_host();
        mesh_buf->add_pending_event(event);
        log_info(tt::LogTest, "[MultiCQDeallocRace] CQ0: write + event recorded (event_id={})", event.id());
    }

    // Step 3: CQ1 — record a pending event, then quiesce. This drains CQ1 and
    // leaves wait_for_pending_events() with a non-empty stale CQ1 slot.
    {
        auto& cq1 = mesh_device_->mesh_command_queue(1);
        std::vector<uint32_t> dummy(num_pages * page_size / sizeof(uint32_t), 0xDEAD);
        EnqueueWriteMeshBuffer(cq1, mesh_buf, dummy, /*blocking=*/false);
        auto event = cq1.enqueue_record_event_to_host();
        mesh_buf->add_pending_event(event);
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
    auto& cq0 = mesh_device_->mesh_command_queue(0);
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

}  // namespace
}  // namespace tt::tt_metal::distributed::test
