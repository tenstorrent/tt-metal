// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/host_api.hpp>
#include <future>
#include <initializer_list>
#include <memory>
#include <thread>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/device.hpp>
#include "impl/dispatch/dispatch_settings.hpp"
#include "impl/dispatch/system_memory_manager.hpp"
#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"

namespace tt::tt_metal {

using std::vector;

constexpr uint32_t completion_queue_event_offset = sizeof(CQDispatchCmd);
constexpr uint32_t completion_queue_page_size = DispatchSettings::TRANSFER_PAGE_SIZE;

enum class DataMovementMode : uint8_t { WRITE = 0, READ = 1 };

TEST_F(UnitMeshCQEventFixture, TestEventsDataMovementWrittenToCompletionQueueInOrder) {
    size_t num_buffers = 100;
    uint32_t page_size = 2048;
    vector<uint32_t> page(page_size / sizeof(uint32_t));
    uint32_t expected_event_id = 0;
    uint32_t last_read_address = 0;
    auto mesh_device = this->devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device = mesh_device->get_devices()[0];

    for (const DataMovementMode data_movement_mode : {DataMovementMode::READ, DataMovementMode::WRITE}) {
        auto start = std::chrono::system_clock::now();

        uint32_t completion_queue_base = device->sysmem_manager().get_completion_queue_read_ptr(0);
        chip_id_t mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());

        vector<std::shared_ptr<distributed::MeshBuffer>> buffers;
        for (size_t i = 0; i < num_buffers; i++) {
            distributed::DeviceLocalBufferConfig local_config{.page_size = page_size, .buffer_type = BufferType::DRAM};
            distributed::ReplicatedBufferConfig buffer_config{.size = page_size};
            buffers.push_back(distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get()));

            if (data_movement_mode == DataMovementMode::WRITE) {
                distributed::WriteShard(cq, buffers.back(), page, distributed::MeshCoordinate(0, 0), true);
            } else if (data_movement_mode == DataMovementMode::READ) {
                distributed::ReadShard(cq, page, buffers.back(), distributed::MeshCoordinate(0, 0), true);
            }
        }
        distributed::Finish(cq);

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);

        // Read completion queue and ensure we see events 0-99 inclusive in order
        uint32_t event;
        if (data_movement_mode == DataMovementMode::WRITE) {
            for (size_t i = 0; i < num_buffers; i++) {
                uint32_t host_addr = last_read_address + i * completion_queue_page_size + completion_queue_event_offset;
                tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
                    &event, 4, host_addr, mmio_device_id, channel);
                EXPECT_EQ(event, ++expected_event_id);  // Event ids start at 1
            }
        } else if (data_movement_mode == DataMovementMode::READ) {
            for (size_t i = 0; i < num_buffers; i++) {
                // Extra entry in the completion queue is from the buffer read data.
                uint32_t host_addr =
                    completion_queue_base + (2 * i + 1) * completion_queue_page_size + completion_queue_event_offset;
                tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
                    &event, 4, host_addr, mmio_device_id, channel);
                EXPECT_EQ(event, ++expected_event_id);  // Event ids start at 1
                last_read_address = host_addr - completion_queue_event_offset + completion_queue_page_size;
            }
        }
    }
}

// Basic test, record events, check that Event struct was updated. Enough commands to trigger issue queue wrap.
TEST_F(UnitMeshCQEventFixture, TestEventsEnqueueRecordEventIssueQueueWrap) {
    auto mesh_device = this->devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    const size_t num_events = 100000;  // Enough to wrap issue queue. 768MB and cmds are 22KB each, so 35k cmds.
    uint32_t cmds_issued_per_cq = 0;

    auto start = std::chrono::system_clock::now();

    for (size_t i = 0; i < num_events; i++) {
        auto event = distributed::EnqueueRecordEventToHost(cq);
        EXPECT_EQ(event.id(), cmds_issued_per_cq + 1);  // Event ids start at 1
        EXPECT_EQ(event.mesh_cq_id(), cq.id());
        cmds_issued_per_cq++;
    }
    distributed::Finish(cq);

    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
    log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
}

// Test where Host synchronously waits for event to be completed.
TEST_F(UnitMeshCQEventFixture, TestEventsEnqueueRecordEventAndSynchronize) {
    auto mesh_device = this->devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    const size_t num_events = 100;
    const size_t num_events_between_sync = 10;

    auto start = std::chrono::system_clock::now();

    std::vector<std::shared_ptr<distributed::MeshEvent>> sync_events;

    // A bunch of events recorded, occasionally will sync from host.
    for (size_t i = 0; i < num_events; i++) {
        auto event = sync_events.emplace_back(
            std::make_shared<distributed::MeshEvent>(distributed::EnqueueRecordEventToHost(cq)));
        // Host synchronize every N number of events.
        if (i > 0 && ((i % num_events_between_sync) == 0)) {
            distributed::EventSynchronize(*event);
        }
    }
    // A bunch of bonus syncs where event_id is mod on earlier ID's.
    distributed::EventSynchronize(*sync_events.at(2));
    distributed::EventSynchronize(*sync_events.at(sync_events.size() - 2));
    distributed::EventSynchronize(*sync_events.at(5));

    distributed::Finish(cq);

    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
    log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
}

// Device sync. Single CQ here, less interesting than 2CQ but still useful. Ensure no hangs.
TEST_F(UnitMeshCQEventFixture, TestEventsQueueWaitForEventBasic) {
    auto mesh_device = this->devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    const size_t num_events = 50;
    const size_t num_events_between_sync = 5;

    auto start = std::chrono::system_clock::now();
    std::vector<std::shared_ptr<distributed::MeshEvent>> sync_events;

    // A bunch of events recorded, occasionally will sync from device.
    for (size_t i = 0; i < num_events; i++) {
        auto event = sync_events.emplace_back(
            std::make_shared<distributed::MeshEvent>(distributed::EnqueueRecordEventToHost(cq)));

        // Device synchronize every N number of events.
        if (i > 0 && ((i % num_events_between_sync) == 0)) {
            log_debug(tt::LogTest, "Going to WaitForEvent for i: {}", i);
            distributed::EnqueueWaitForEvent(cq, *event);
        }
    }

    // A bunch of bonus syncs where event_id is mod on earlier ID's.
    distributed::EnqueueWaitForEvent(cq, *sync_events.at(0));
    distributed::EnqueueWaitForEvent(cq, *sync_events.at(sync_events.size() - 5));
    distributed::EnqueueWaitForEvent(cq, *sync_events.at(4));
    distributed::Finish(cq);

    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
    log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
}

// Device sync. Single CQ here, less interesting than 2CQ but still useful. Ensure no hangs.
TEST_F(UnitMeshCQEventFixture, TestEventsEventsQueryBasic) {
    auto mesh_device = this->devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    const size_t num_events = 50;
    const size_t num_events_between_query = 5;
    bool event_status;

    auto start = std::chrono::system_clock::now();
    std::vector<std::shared_ptr<distributed::MeshEvent>> sync_events;

    // Record many events, occasionally query from host, but cannot guarantee completion status.
    for (size_t i = 0; i < num_events; i++) {
        auto event = sync_events.emplace_back(
            std::make_shared<distributed::MeshEvent>(distributed::EnqueueRecordEventToHost(cq)));

        if (i > 0 && ((i % num_events_between_query) == 0)) {
            [[maybe_unused]] auto status = distributed::EventQuery(*event);
            log_trace(tt::LogTest, "EventQuery for i: {} - status: {}", i, status);
        }
    }

    // Wait until earlier events are finished, then ensure query says they are finished.
    auto& early_event_1 = sync_events.at(num_events - 10);
    distributed::EventSynchronize(*early_event_1);  // Block until this event is finished.
    event_status = distributed::EventQuery(*early_event_1);
    EXPECT_EQ(event_status, true);

    auto& early_event_2 = sync_events.at(num_events - 5);
    distributed::Finish(cq);  // Block until all events finished.
    event_status = distributed::EventQuery(*early_event_2);
    EXPECT_EQ(event_status, true);

    // Query a future event that hasn't completed and ensure it's not finished.
    auto future_event = std::make_shared<distributed::MeshEvent>(
        0xffff,
        mesh_device.get(),
        cq.id(),
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 0)));
    distributed::EnqueueRecordEvent(cq);
    event_status = distributed::EventQuery(*future_event);
    EXPECT_EQ(event_status, false);

    distributed::Finish(cq);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
    log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
}

// Mix of WritesBuffers, RecordEvent, WaitForEvent, EventSynchronize with some checking.
TEST_F(UnitMeshCQEventFixture, TestEventsMixedWriteBufferRecordWaitSynchronize) {
    auto mesh_device = this->devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device = mesh_device->get_devices()[0];
    const size_t num_buffers = 2;
    const uint32_t page_size = 2048;
    vector<uint32_t> page(page_size / sizeof(uint32_t));
    uint32_t events_issued_per_cq = 0;
    const uint32_t num_events_per_cq = 2;  // Record and blocking write
    uint32_t expected_event_id = 0;

    auto start = std::chrono::system_clock::now();

    uint32_t completion_queue_base = device->sysmem_manager().get_completion_queue_read_ptr(0);
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
    for (size_t i = 0; i < num_buffers; i++) {
        log_debug(tt::LogTest, "i: {} - Going to record event, write, wait, synchronize.", i);
        auto event = std::make_shared<distributed::MeshEvent>(distributed::EnqueueRecordEventToHost(cq));
        EXPECT_EQ(event->mesh_cq_id(), cq.id());
        EXPECT_EQ(event->id(), events_issued_per_cq + 1);  // Event ids start at 1

        distributed::DeviceLocalBufferConfig local_config{.page_size = page_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig buffer_config{.size = page_size};
        std::shared_ptr<distributed::MeshBuffer> buf =
            distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
        distributed::WriteShard(cq, buf, page, distributed::MeshCoordinate(0, 0), true);
        distributed::EnqueueWaitForEvent(cq, *event);

        if (i % 10 == 0) {
            distributed::EventSynchronize(*event);
        }
        events_issued_per_cq += num_events_per_cq;
    }
    distributed::Finish(cq);

    // Read completion queue and ensure we see expected event IDs
    uint32_t event_id;
    for (size_t i = 0; i < num_buffers * num_events_per_cq; i++) {
        uint32_t host_addr = completion_queue_base + i * completion_queue_page_size + completion_queue_event_offset;
        tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
            &event_id, 4, host_addr, mmio_device_id, channel);
        EXPECT_EQ(event_id, ++expected_event_id);
    }

    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
    log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
}

}  // namespace tt::tt_metal
