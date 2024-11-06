// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/watcher_server.hpp"
#include "tt_metal/impl/event/event.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using std::vector;
using namespace tt::tt_metal;

enum class DataMovementMode: uint8_t {
    WRITE = 0,
    READ = 1
};


TEST_F(CommandQueueEventFixture, TestEventsDataMovementWrittenToCompletionQueueInOrder) {
    size_t num_buffers = 100;
    uint32_t page_size = 2048;
    vector<uint32_t> page(page_size / sizeof(uint32_t));
    uint32_t expected_event_id = 0;
    uint32_t last_read_address = 0;

    for (const DataMovementMode data_movement_mode: {DataMovementMode::READ, DataMovementMode::WRITE}) {

        auto start = std::chrono::system_clock::now();

        uint32_t completion_queue_base = this->device_->sysmem_manager().get_completion_queue_read_ptr(0);
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());

        vector<std::shared_ptr<Buffer>> buffers;
        for (size_t i = 0; i < num_buffers; i++) {
            buffers.push_back(Buffer::create(this->device_, page_size, page_size, BufferType::DRAM));

            if (data_movement_mode == DataMovementMode::WRITE) {
                EnqueueWriteBuffer(this->device_->command_queue(), buffers.back(), page, true);
            } else if (data_movement_mode == DataMovementMode::READ) {
                EnqueueReadBuffer(this->device_->command_queue(), buffers.back(), page, true);
            }
        }
        Finish(this->device_->command_queue());

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        tt::log_info(tt::LogTest, "Test Finished in {:.2f} us", elapsed_seconds.count() * 1000 * 1000);

        // Read completion queue and ensure we see events 0-99 inclusive in order
        uint32_t event;
        if (data_movement_mode == DataMovementMode::WRITE) {
            for (size_t i = 0; i < num_buffers; i++) {
                uint32_t host_addr = last_read_address + i*completion_queue_page_size + completion_queue_event_offset;
                tt::Cluster::instance().read_sysmem(&event, 4, host_addr, mmio_device_id, channel);
                EXPECT_EQ(event, ++expected_event_id); // Event ids start at 1
            }
        } else if (data_movement_mode == DataMovementMode::READ) {
            for (size_t i = 0; i < num_buffers; i++) {
                // Extra entry in the completion queue is from the buffer read data.
                uint32_t host_addr = completion_queue_base + (2*i + 1)*completion_queue_page_size + completion_queue_event_offset;
                tt::Cluster::instance().read_sysmem(&event, 4, host_addr, mmio_device_id, channel);
                EXPECT_EQ(event, ++expected_event_id); // Event ids start at 1
                last_read_address = host_addr - completion_queue_event_offset + completion_queue_page_size;
            }
        }
    }

}
