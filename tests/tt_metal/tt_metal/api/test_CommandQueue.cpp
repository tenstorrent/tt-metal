// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt::tt_metal;

namespace host_tests {

namespace multi_device_tests {
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

TEST_F(CommandQueueProgramFixture, DISABLED_TensixTestAsyncAssertForDeprecatedAPI) {
    auto &command_queue = this->device_->command_queue();
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
    } catch (std::runtime_error &e) {
        std::string expected =
            "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for "
            "Fast Dispatch.";
        const string error = string(e.what());
        EXPECT_TRUE(error.find(expected) != std::string::npos);
    }
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
        tt::Cluster::instance().write_sysmem(data.data(), data.size() * sizeof(uint32_t), address, mmio_device_id, channel);

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
}




}   // namespace host_tests
