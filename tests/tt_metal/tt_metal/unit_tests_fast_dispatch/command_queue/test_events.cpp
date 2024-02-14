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

TEST_F(CommandQueueSingleCardFixture, TestEventsWrittenToCompletionQueueInOrder) {
    size_t num_buffers = 100;
    uint32_t page_size = 2048;
    vector<uint32_t> page(page_size / sizeof(uint32_t));
    for (Device *device : devices_) {
        uint32_t completion_queue_base = device->sysmem_manager().get_completion_queue_read_ptr(0);
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        constexpr uint32_t completion_queue_event_alignment = 32;
        for (size_t i = 0; i < num_buffers; i++) {
            Buffer buf(device, page_size, page_size, BufferType::DRAM);
            EnqueueWriteBuffer(device->command_queue(), buf, page, false);
        }
        Finish(device->command_queue());

        // Read completion queue and ensure we see events 0-99 inclusive in order
        uint32_t event;
        for (size_t i = 0; i < num_buffers; i++) {
            uint32_t host_addr = completion_queue_base + i * completion_queue_event_alignment;
            tt::Cluster::instance().read_sysmem(&event, 4, host_addr, mmio_device_id, channel);
            EXPECT_EQ(event, i);
        }
    }
}
