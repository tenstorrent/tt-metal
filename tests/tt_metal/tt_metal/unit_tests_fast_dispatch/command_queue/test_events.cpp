// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

TEST_F(CommandQueueFixture, TestEventsWrittenToCompletionQueueInOrder) {
    size_t num_buffers = 100;
    uint32_t page_size = 2048;
    vector<uint32_t> page(page_size / sizeof(uint32_t));
    uint32_t completion_queue_base = this->device_->sysmem_manager().get_completion_queue_read_ptr(0);
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
    constexpr uint32_t completion_queue_event_alignment = 32;
    for (size_t i = 0; i < num_buffers; i++) {
        Buffer buf(this->device_, page_size, page_size, BufferType::DRAM);
        EnqueueWriteBuffer(*this->cmd_queue, buf, page, false);
    }
    Finish(*this->cmd_queue);

    // Read completion queue and ensure we see events 0-99 inclusive in order
    uint32_t event;
    for (size_t i = 0; i < num_buffers; i++) {
        uint32_t host_addr = completion_queue_base + i * completion_queue_event_alignment;
        tt::Cluster::instance().read_sysmem(&event, 4, host_addr, mmio_device_id, channel);
        EXPECT_EQ(event, i);
    }
}

// Basic test, record events, check that Event struct was updated. Enough commands to trigger issue queue wrap.
TEST_F(CommandQueueFixture, TestEventsEnqueueRecordEventIssueQueueWrap) {
    size_t num_events = 100000; // Enough to wrap issue queue. 768MB and cmds are 22KB each, so 35k cmds.
    for (size_t i = 0; i < num_events; i++) {
        Event event;
        EnqueueQueueRecordEvent(*this->cmd_queue, event);
        EXPECT_EQ(event.event_id, i);
        EXPECT_EQ(event.cq_id, this->cmd_queue->id());
    }
    Finish(*this->cmd_queue);
}

// Test where Host synchronously waits for event to be completed.
TEST_F(CommandQueueFixture, TestEventsEnqueueRecordEventAndSynchronize) {
    size_t num_events = 100;
    size_t num_events_between_sync = 10;
    std::vector<Event> sync_events;

    // A bunch of events recorded, occasionally will sync from host.
    for (size_t i = 0; i < num_events; i++) {
        auto &event = sync_events.emplace_back(Event());

        // Uncomment this to ensure syncing on uninitialized event would assert.
        // EventSynchronize(event);

        EnqueueQueueRecordEvent(*this->cmd_queue, event);
        log_debug(tt::LogTest, "Done recording event. Got Event(event_id: {} cq_id: {})", event.event_id, event.cq_id);
        EXPECT_EQ(event.event_id, i);
        EXPECT_EQ(event.cq_id, this->cmd_queue->id());

        // Host synchronize every N number of events.
        if (i > 0 && ((i % num_events_between_sync) == 0)) {
            EventSynchronize(event);
        }
    }

    // A bunch of bonus syncs where event_id is mod on earlier ID's.
    EventSynchronize(sync_events.at(2));
    EventSynchronize(sync_events.at(sync_events.size() - 2));
    EventSynchronize(sync_events.at(5));

    // Uncomment this to confirm future events not yet seen would hang.
    // log_debug(tt::LogTest, "The next event is not yet seen, would hang");
    // Event future_event = sync_events.at(0);
    // future_event.event_id = num_events + 2;
    // EventSynchronize(future_event);

    Finish(*this->cmd_queue);
}
