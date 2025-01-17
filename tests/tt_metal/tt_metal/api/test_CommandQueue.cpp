// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "command_queue_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include "tt_metal/common/scoped_timer.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt::tt_metal;

namespace host_tests {

TEST_F(CommandQueueMultiDeviceFixture, DISABLED_TestAccessCommandQueue) {
    for (unsigned int device_id = 0; device_id < num_devices_; device_id++) {
        EXPECT_NO_THROW(devices_[device_id]->command_queue());
    }
}

TEST_F(CommandQueueFixture, TestCannotAccessCommandQueueForClosedDevice) {
    EXPECT_NO_THROW(device_->command_queue());
    CloseDevice(device_);
    EXPECT_ANY_THROW(device_->command_queue());
}

TEST_F(CommandQueueFixture, DISABLED_TensixTestAsyncAssertForDeprecatedAPI) {
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    Program program;
    CoreCoord core = {0, 0};
    uint32_t buf_size = 4096;
    uint32_t page_size = 4096;
    auto dummy_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto src0 = Buffer::create(this->device_, buf_size, page_size, BufferType::DRAM);
    std::vector<uint32_t> runtime_args = {src0->address()};
    try {
        SetRuntimeArgs(program, dummy_kernel, core, runtime_args);
    } catch (std::runtime_error& e) {
        std::string expected =
            "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for "
            "Fast Dispatch.";
        const string error = string(e.what());
        EXPECT_TRUE(error.find(expected) != std::string::npos);
    }
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueProgramFixture, TensixTestAsyncCommandQueueSanityAndProfile) {
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    Program program;

    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});
    // Add an NCRISC blank manually, but in compile program, the BRISC blank will be
    // added separately
    auto dummy_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    // Use scoper timer to benchmark time for pushing 2 commands
    {
        tt::ScopedTimer timer("AsyncCommandQueue");
        EnqueueProgram(command_queue, program, false);
        Finish(command_queue);
    }
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueBufferFixture, DISABLED_TensixTestAsyncCBAllocation) {
    // Test asynchronous allocation of buffers and their assignment to CBs
    auto& command_queue = this->device_->command_queue();
    auto current_mode = CommandQueue::default_mode();
    command_queue.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    Program program;

    const uint32_t num_pages = 1;
    const uint32_t page_size = detail::TileSize(tt::DataFormat::Float16_b);
    const tt::DataFormat data_format = tt::DataFormat::Float16_b;

    auto buffer_size = page_size;
    tt::tt_metal::InterleavedBufferConfig buff_config{
        .device = this->device_,
        .size = buffer_size,
        .page_size = buffer_size,
        .buffer_type = tt::tt_metal::BufferType::L1};
    // Asynchronously allocate an L1 Buffer
    auto l1_buffer = CreateBuffer(buff_config);
    CoreRange cr({0, 0}, {0, 2});
    CoreRangeSet cr_set({cr});
    std::vector<uint8_t> buffer_indices = {16, 24};

    CircularBufferConfig config1 =
        CircularBufferConfig(
            page_size, {{buffer_indices[0], data_format}, {buffer_indices[1], data_format}}, *l1_buffer)
            .set_page_size(buffer_indices[0], page_size)
            .set_page_size(buffer_indices[1], page_size);
    // Asynchronously assign the L1 Buffer to the CB
    auto multi_core_cb = CreateCircularBuffer(program, cr_set, config1);
    auto cb_ptr = detail::GetCircularBuffer(program, multi_core_cb);
    Finish(this->device_->command_queue());
    // Addresses should match
    EXPECT_EQ(cb_ptr->address(), l1_buffer->address());
    // Asynchronously allocate a new L1 buffer
    auto l1_buffer_2 = CreateBuffer(buff_config);
    // Asynchronously update CB address to match new L1 buffer
    UpdateDynamicCircularBufferAddress(program, multi_core_cb, *l1_buffer_2);
    Finish(this->device_->command_queue());
    // Addresses should match
    EXPECT_EQ(cb_ptr->address(), l1_buffer_2->address());
    command_queue.set_mode(current_mode);
}

TEST_F(CommandQueueMultiDeviceFixture, DISABLED_TestDirectedLoopbackToUniqueHugepage) {
    std::unordered_map<chip_id_t, std::vector<uint32_t>> golden_data;

    const uint32_t byte_size = 2048 * 16;
    const uint64_t address = 0;

    for (chip_id_t device_id = 0; device_id < num_devices_; device_id++) {
        std::vector<uint32_t> data =
            tt::test_utils::generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));

        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        tt::Cluster::instance().write_sysmem(
            data.data(), data.size() * sizeof(uint32_t), address, mmio_device_id, channel);

        golden_data[device_id] = data;
    }

    std::vector<uint32_t> readback_data;
    readback_data.resize(byte_size / sizeof(uint32_t));
    for (chip_id_t device_id = 0; device_id < num_devices_; device_id++) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        tt::Cluster::instance().read_sysmem(readback_data.data(), byte_size, address, mmio_device_id, channel);
        EXPECT_EQ(readback_data, golden_data.at(device_id));
    }
}
}  // namespace host_tests
