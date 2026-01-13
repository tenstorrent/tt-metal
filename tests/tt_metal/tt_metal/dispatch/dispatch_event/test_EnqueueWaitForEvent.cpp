// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <cstdint>
#include <sys/types.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <functional>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "impl/dispatch/command_queue.hpp"
#include "impl/dispatch/dispatch_settings.hpp"
#include "impl/dispatch/system_memory_manager.hpp"
#include "dispatch_test_utils.hpp"
#include "gtest/gtest.h"
#include "multi_command_queue_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal {

using std::vector;

namespace local_test_functions {

void FinishAllCqs(vector<std::reference_wrapper<distributed::MeshCommandQueue>>& cqs) {
    for (auto& cq : cqs) {
        distributed::Finish(cq);
    }
}
}  // namespace local_test_functions

namespace basic_tests {

// Simplest test to record Event per CQ and wait from host, and verify populated Event struct is correct (many events,
// wrap issue queue)
TEST_F(UnitMeshMultiCQMultiDeviceEventFixture, TestEventsEventSynchronizeSanity) {
    for (auto& mesh_device : devices_) {
        log_info(tt::LogTest, "Running On Device {}", mesh_device->get_devices()[0]->id());
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {
            mesh_device->mesh_command_queue(0), mesh_device->mesh_command_queue(1)};
        vector<uint32_t> cmds_issued_per_cq = {0, 0};

        ASSERT_EQ(cqs.size(), 2) << "Expected 2 command queues";
        const int num_cmds_per_cq = 1;

        auto start = std::chrono::system_clock::now();
        std::unordered_map<uint, std::vector<distributed::MeshEvent>> sync_events;
        const size_t num_events = 10;

        for (size_t j = 0; j < num_events; j++) {
            for (uint i = 0; i < cqs.size(); i++) {
                log_debug(
                    tt::LogTest, "j : {} Recording and Host Syncing on event for CQ ID: {}", j, cqs[i].get().id());
                auto event = sync_events[i].emplace_back(cqs[i].get().enqueue_record_event_to_host());
                distributed::EventSynchronize(event);
                // Can check events fields after prev sync w/ async CQ.
                EXPECT_EQ(event.mesh_cq_id(), cqs[i].get().id());
                EXPECT_EQ(event.id(), cmds_issued_per_cq[i] + 1);
                cmds_issued_per_cq[i] += num_cmds_per_cq;
            }
        }

        // Sync on earlier events again per CQ just to show it works.
        for (uint i = 0; i < cqs.size(); i++) {
            for (size_t j = 0; j < num_events; j++) {
                distributed::EventSynchronize(sync_events.at(i)[j]);
            }
        }

        local_test_functions::FinishAllCqs(cqs);
        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
    }
}

// Simplest test to record Event per CQ and wait from host, and verify populated Event struct is correct (many events,
// wrap issue queue)
TEST_F(UnitMeshMultiCQSingleDeviceEventFixture, TestEventsEventSynchronizeSanity) {
    vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {
        this->device_->mesh_command_queue(0), this->device_->mesh_command_queue(1)};
    vector<uint32_t> cmds_issued_per_cq = {0, 0};

    ASSERT_EQ(cqs.size(), 2) << "Expected 2 command queues";
    const int num_cmds_per_cq = 1;

    auto start = std::chrono::system_clock::now();
    std::unordered_map<uint, std::vector<distributed::MeshEvent>> sync_events;
    const size_t num_events = 10;

    for (size_t j = 0; j < num_events; j++) {
        for (uint i = 0; i < cqs.size(); i++) {
            log_debug(tt::LogTest, "j : {} Recording and Host Syncing on event for CQ ID: {}", j, cqs[i].get().id());
            auto event = sync_events[i].emplace_back(cqs[i].get().enqueue_record_event_to_host());
            distributed::EventSynchronize(event);
            // Can check events fields after prev sync w/ async CQ.
            EXPECT_EQ(event.mesh_cq_id(), cqs[i].get().id());
            EXPECT_EQ(event.id(), cmds_issued_per_cq[i] + 1);
            cmds_issued_per_cq[i] += num_cmds_per_cq;
        }
    }

    // Sync on earlier events again per CQ just to show it works.
    for (uint i = 0; i < cqs.size(); i++) {
        for (size_t j = 0; j < num_events; j++) {
            distributed::EventSynchronize(sync_events.at(i)[j]);
        }
    }

    local_test_functions::FinishAllCqs(cqs);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
    log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
}

// Simplest test to record and wait-for-events on same CQ.
TEST_F(UnitMeshMultiCQMultiDeviceEventFixture, TestEventsEnqueueWaitForEventSanity) {
    for (auto& mesh_device : devices_) {
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {
            mesh_device->mesh_command_queue(0), mesh_device->mesh_command_queue(1)};
        vector<uint32_t> events_issued_per_cq = {0, 0};
        size_t num_events = 10;

        ASSERT_EQ(cqs.size(), 2) << "Expected 2 command queues";
        const int num_events_per_cq = 1;

        auto start = std::chrono::system_clock::now();

        for (size_t j = 0; j < num_events; j++) {
            for (uint i = 0; i < cqs.size(); i++) {
                log_debug(
                    tt::LogTest, "j : {} Recording and Device Syncing on event for CQ ID: {}", j, cqs[i].get().id());
                auto event = cqs[i].get().enqueue_record_event();
                EXPECT_EQ(event.mesh_cq_id(), cqs[i].get().id());
                EXPECT_EQ(event.id(), events_issued_per_cq[i] + 1);
                cqs[i].get().enqueue_wait_for_event(event);
                events_issued_per_cq[i] += num_events_per_cq;
            }
        }
        local_test_functions::FinishAllCqs(cqs);
        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
    }
}

// Record event on one CQ, wait-for-that-event on another CQ. Then do the flip. Occasionally insert
// syncs from Host per CQ, and verify completion queues per CQ are correct.
TEST_F(UnitMeshMultiCQMultiDeviceEventFixture, TestEventsEnqueueWaitForEventCrossCQs) {
    for (auto& mesh_device : devices_) {
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {
            mesh_device->mesh_command_queue(0), mesh_device->mesh_command_queue(1)};
        vector<uint32_t> cmds_issued_per_cq = {0, 0};
        const size_t num_events_per_cq = 10;

        // Currently hardcoded for 2 CQ. For 3+ CQ, can extend to record for CQ0, Wait for CQ1,CQ2,etc.
        ASSERT_EQ(cqs.size(), 2) << "Expected 2 command queues";
        const int num_cmds_per_cq = 1;
        vector<uint32_t> expected_event_id = {0, 0};

        auto start = std::chrono::system_clock::now();

        // Issue a number of Event Record/Waits per CQ, with Record/Wait on alternate CQs
        for (size_t j = 0; j < num_events_per_cq; j++) {
            for (uint i = 0; i < cqs.size(); i++) {
                auto cq_idx_record = i;
                auto cq_idx_wait = (i + 1) % cqs.size();
                log_debug(
                    tt::LogTest,
                    "j : {} Recording event on CQ ID: {} and Device Syncing on CQ ID: {}",
                    j,
                    cqs[cq_idx_record].get().id(),
                    cqs[cq_idx_wait].get().id());
                auto event = cqs[cq_idx_record].get().enqueue_record_event();
                EXPECT_EQ(event.mesh_cq_id(), cqs[cq_idx_record].get().id());
                EXPECT_EQ(event.id(), cmds_issued_per_cq[i] + 1);
                cqs[cq_idx_wait].get().enqueue_wait_for_event(event);

                // Note: Removed host sync here since MeshCommandQueue::enqueue_record_event creates device-only events
                // that don't notify the host. Host sync would require MeshCommandQueue::enqueue_record_event_to_host.
                cmds_issued_per_cq[cq_idx_record] += num_cmds_per_cq;
                // cmds_issued_per_cq[cq_idx_wait] += num_cmds_per_cq; // wait_for_event no longer records an event on
                // host
            }
        }

        local_test_functions::FinishAllCqs(cqs);

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
    }
}

// Simple 2CQ test to mix reads, writes, record-event, wait-for-event in a basic way. It's simple because
// the write, record-event, wait-event, read-event are all on the same CQ, but cover both CQ's.
TEST_F(UnitMeshMultiCQMultiDeviceEventFixture, TestEventsReadWriteWithWaitForEventSameCQ) {
    for (auto& mesh_device : devices_) {
        TestBufferConfig config = {.num_pages = 1, .page_size = 256, .buftype = BufferType::DRAM};
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {
            mesh_device->mesh_command_queue(0), mesh_device->mesh_command_queue(1)};
        vector<uint32_t> cmds_issued_per_cq = {0, 0};

        size_t buf_size = config.num_pages * config.page_size;

        size_t num_buffers_per_cq = 10;
        bool pass = true;

        auto start = std::chrono::system_clock::now();

        std::unordered_map<uint, std::vector<distributed::MeshEvent>> sync_events;

        for (uint buf_idx = 0; buf_idx < num_buffers_per_cq; buf_idx++) {
            vector<std::shared_ptr<distributed::MeshBuffer>> buffers;
            vector<vector<uint32_t>> srcs;
            for (uint i = 0; i < cqs.size(); i++) {
                uint32_t wr_data_base = (buf_idx * 1000) + (i * 100);
                // Create MeshBuffer with proper config
                distributed::ReplicatedBufferConfig global_buffer_config{.size = buf_size};
                distributed::DeviceLocalBufferConfig device_local_config{
                    .page_size = config.page_size, .buffer_type = config.buftype};
                buffers.push_back(
                    distributed::MeshBuffer::create(global_buffer_config, device_local_config, mesh_device.get()));
                srcs.push_back(generate_arange_vector(buffers[i]->size(), wr_data_base));
                log_debug(tt::LogTest, "buf_idx: {} Doing Write to cq_id: {} of data: {}", buf_idx, i, srcs[i]);

                distributed::WriteShard(cqs[i], buffers[i], srcs[i], distributed::MeshCoordinate(0, 0), false);
                auto event = sync_events[i].emplace_back(cqs[i].get().enqueue_record_event());
            }

            for (uint i = 0; i < cqs.size(); i++) {
                auto event = sync_events[i][buf_idx];
                cqs[i].get().enqueue_wait_for_event(event);
                vector<uint32_t> result;
                distributed::ReadShard(cqs[i], result, buffers[i], zero_coord_, true);  // Blocking.
                bool local_pass = (srcs[i] == result);
                log_debug(
                    tt::LogTest,
                    "Checking buf_idx: {} cq_idx: {} local_pass: {} write_data: {} read_results: {}",
                    buf_idx,
                    i,
                    local_pass,
                    srcs[i],
                    result);
                pass &= local_pass;
            }
        }

        local_test_functions::FinishAllCqs(cqs);
        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);
        EXPECT_TRUE(pass);
    }
}

// More interesting test where Blocking ReadBuffer, Non-Blocking WriteBuffer are on alternate CQs,
// ordered via events. Do many loops, occasionally increasing size of buffers (page size, num pages).
// Ensure read back data is correct, data is different for each write.
TEST_F(UnitMeshMultiCQMultiDeviceEventFixture, TestEventsReadWriteWithWaitForEventCrossCQs) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP() << "Skipping for GS due to readback mismatch under debug Github issue #6281 ";
    }

    for (auto& mesh_device : devices_) {
        TestBufferConfig config = {.num_pages = 1, .page_size = 32, .buftype = BufferType::DRAM};
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {
            mesh_device->mesh_command_queue(0), mesh_device->mesh_command_queue(1)};
        vector<uint32_t> cmds_issued_per_cq = {0, 0};

        size_t num_buffers_per_cq = 50;
        bool pass = true;

        auto start = std::chrono::system_clock::now();

        for (uint buf_idx = 0; buf_idx < num_buffers_per_cq; buf_idx++) {
            // Increase number of pages and page size every 10 buffers, to change async timing betwen CQs.
            if (buf_idx > 0 && ((buf_idx % 10) == 0)) {
                config.page_size *= 2;
                config.num_pages *= 2;
            }

            vector<std::shared_ptr<distributed::MeshBuffer>> buffers;
            vector<vector<uint32_t>> srcs;
            size_t buf_size = config.num_pages * config.page_size;

            for (uint i = 0; i < cqs.size(); i++) {
                uint32_t wr_data_base = (buf_idx * 1000) + (i * 100);
                auto& cq_write = cqs[i];
                auto& cq_read = cqs[(i + 1) % cqs.size()];
                vector<uint32_t> result;

                // Create MeshBuffer with proper config
                distributed::ReplicatedBufferConfig global_buffer_config{.size = buf_size};
                distributed::DeviceLocalBufferConfig device_local_config{
                    .page_size = config.page_size, .buffer_type = config.buftype};
                buffers.push_back(
                    distributed::MeshBuffer::create(global_buffer_config, device_local_config, mesh_device.get()));
                srcs.push_back(generate_arange_vector(buffers[i]->size(), wr_data_base));

                // Blocking Read after Non-Blocking Write on alternate CQs, events ensure ordering.
                log_debug(
                    tt::LogTest,
                    "buf_idx: {} Doing Write (page_size: {} num_pages: {}) to cq_id: {}",
                    buf_idx,
                    config.page_size,
                    config.num_pages,
                    cq_write.get().id());

                distributed::WriteShard(cq_write, buffers[i], srcs[i], zero_coord_, false);
                auto event = cq_write.get().enqueue_record_event();
                cq_read.get().enqueue_wait_for_event(event);
                distributed::ReadShard(cq_read, result, buffers[i], zero_coord_, true);
                bool local_pass = (srcs[i] == result);
                log_debug(
                    tt::LogTest,
                    "Checking buf_idx: {} cq_idx: {} local_pass: {} write_data: {} read_results: {}",
                    buf_idx,
                    i,
                    local_pass,
                    srcs[i],
                    result);
                pass &= local_pass;
            }
        }

        local_test_functions::FinishAllCqs(cqs);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = (end - start);
        log_info(tt::LogTest, "Test Finished in {}us", elapsed_seconds.count() * 1000 * 1000);
        EXPECT_TRUE(pass);
    }
}

// 2 CQs with single Buffer, and a loop where each iteration has non-blocking Write to Buffer via CQ0 and non-blocking
// Read to Bufffer via CQ1. Ping-Pongs between Writes and Reads to same buffer. Use events to synchronze read after
// write and write after read before checking correct data read at the end after all cmds finished on device.
TEST_F(UnitMeshMultiCQMultiDeviceEventFixture, TestEventsReadWriteWithWaitForEventCrossCQsPingPong) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP() << "Skipping for GS due to readback mismatch under debug Github issue #6281 ";
    }

    for (auto& mesh_device : devices_) {
        TestBufferConfig config = {.num_pages = 1, .page_size = 16, .buftype = BufferType::DRAM};
        vector<std::reference_wrapper<distributed::MeshCommandQueue>> cqs = {
            mesh_device->mesh_command_queue(0), mesh_device->mesh_command_queue(1)};
        size_t buf_size = config.num_pages * config.page_size;

        bool pass = true;

        // Some configuration, eventually refactor and spawn more tests.
        int num_buffers = 20;
        int num_wr_rd_per_buf = 5;
        bool use_events = true;  // Set to false to see failures.

        ASSERT_EQ(cqs.size(), 2) << "Expected 2 command queues";

        auto start = std::chrono::system_clock::now();

        // Repeat test starting with different CQ ID. Could have placed this loop lower down.
        for (uint cq_idx = 0; cq_idx < cqs.size(); cq_idx++) {
            auto& cq_write = cqs[cq_idx];
            auto& cq_read = cqs[(cq_idx + 1) % cqs.size()];

            // Another loop for increased testing. Repeat test multiple times for different buffers.
            for (int i = 0; i < num_buffers; i++) {
                vector<vector<uint32_t>> write_data;
                vector<vector<uint32_t>> read_results;
                vector<std::shared_ptr<distributed::MeshBuffer>> buffers;

                // Create MeshBuffer with proper config
                distributed::ReplicatedBufferConfig global_buffer_config{.size = buf_size};
                distributed::DeviceLocalBufferConfig device_local_config{
                    .page_size = config.page_size, .buffer_type = config.buftype};
                buffers.push_back(
                    distributed::MeshBuffer::create(global_buffer_config, device_local_config, mesh_device.get()));

                // Number of write-read combos per buffer. Fewer make RAW race without events easier to hit.
                for (uint j = 0; j < num_wr_rd_per_buf; j++) {
                    // Add entry in resutls vector, and construct write data, unique per loop
                    read_results.emplace_back();
                    write_data.push_back(generate_arange_vector(buffers.back()->size(), j * 100));

                    // Issue non-blocking write via first CQ and record event to synchronize with read on other CQ.
                    log_debug(
                        tt::LogTest,
                        "cq_idx: {} Doing Write j: {} (page_size: {} num_pages: {}) to cq_id: {} write_data: {}",
                        cq_idx,
                        j,
                        config.page_size,
                        config.num_pages,
                        cq_write.get().id(),
                        write_data.back());

                    distributed::WriteShard(cq_write, buffers.back(), write_data.back(), zero_coord_, false);
                    if (use_events) {
                        distributed::MeshEvent event_sync_read_after_write = cq_write.get().enqueue_record_event();

                        // Issue wait for write to complete, and non-blocking read from the second CQ.
                        cq_read.get().enqueue_wait_for_event(event_sync_read_after_write);
                    }
                    distributed::ReadShard(cq_read, read_results.back(), buffers.back(), zero_coord_, false);
                    log_debug(
                        tt::LogTest,
                        "cq_idx: {} Issued Read for j: {} to cq_id: {} got data: {}",
                        cq_idx,
                        j,
                        cq_read.get().id(),
                        read_results.back());  // Data not ready since non-blocking.

                    // If more loops, Record Event on second CQ and wait for it to complete on first CQ before next
                    // loop's write.
                    if (use_events && j < num_wr_rd_per_buf - 1) {
                        distributed::MeshEvent event_sync_write_after_read = cq_read.get().enqueue_record_event();
                        cq_write.get().enqueue_wait_for_event(event_sync_write_after_read);
                    }
                }

                // Basically like Finish, but use host sync on event to ensure all read cmds are finished.
                if (use_events) {
                    auto event_done_reads = cq_read.get().enqueue_record_event_to_host();
                    distributed::EventSynchronize(event_done_reads);
                }

                ASSERT_EQ(write_data.size(), read_results.size());
                ASSERT_EQ(write_data.size(), num_wr_rd_per_buf);

                for (uint j = 0; j < num_wr_rd_per_buf; j++) {
                    // Make copy of read results, helpful for comparison without events, since vector may be updated
                    // between comparison and debug log.
                    auto read_results_snapshot = read_results[j];
                    bool local_pass = write_data[j] == read_results_snapshot;
                    if (!local_pass) {
                        log_warning(
                            tt::LogTest,
                            "cq_idx: {} Checking j: {} local_pass: {} write_data: {} read_results: {}",
                            cq_idx,
                            j,
                            local_pass,
                            write_data[j],
                            read_results_snapshot);
                    }
                    pass &= local_pass;
                }

                // Before starting test with another buffer, drain CQs. Without this, see segfaults after
                // adding num_buffers loop.
                local_test_functions::FinishAllCqs(cqs);

            }  // num_buffers

        }  // cqs

        local_test_functions::FinishAllCqs(cqs);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = (end - start);
        log_info(tt::LogTest, "Test Finished in {}us", elapsed_seconds.count() * 1000 * 1000);

        EXPECT_TRUE(pass);
    }
}

}  // end namespace basic_tests

}  // namespace tt::tt_metal
