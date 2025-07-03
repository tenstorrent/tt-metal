// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <memory>
#include <vector>

#include "buffer_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/stimulus.hpp"

using tt::tt_metal::IDevice;
using namespace tt::test_utils;
using namespace tt::test::buffer::detail;

namespace tt::test::buffer::detail {
bool SimpleDramReadOnly(IDevice* device, size_t local_address, size_t byte_size) {
    std::vector<uint32_t> inputs =
        generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    std::vector<uint32_t> outputs;
    uint32_t dram_channel = device->allocator()->get_dram_channel_from_bank_id(0);
    writeDramBackdoor(device, dram_channel, local_address, inputs);
    readDramBackdoor(device, dram_channel, local_address, byte_size, outputs);
    bool pass = (inputs == outputs);
    if (not pass) {
        log_info(tt::LogTest, "Mismatch at Channel={}, Packet Size(in Bytes)={}", dram_channel, byte_size);
    }
    return pass;
}
bool SimpleDramWriteOnly(IDevice* device, size_t local_address, size_t byte_size) {
    std::vector<uint32_t> inputs =
        generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    std::vector<uint32_t> outputs;
    uint32_t dram_channel = device->allocator()->get_dram_channel_from_bank_id(0);
    writeDramBackdoor(device, dram_channel, local_address, inputs);
    readDramBackdoor(device, dram_channel, local_address, byte_size, outputs);
    bool pass = (inputs == outputs);
    if (not pass) {
        log_info(tt::LogTest, "Mismatch at Channel={}, Packet Size(in Bytes)={}", dram_channel, byte_size);
    }
    return pass;
}
}  // namespace tt::test::buffer::detail

namespace tt::tt_metal {

TEST_F(DeviceFixture, TestSimpleDramBufferReadOnlyLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::DRAM);
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), lo_address, 4));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), lo_address, 8));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), lo_address, 16));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), lo_address, 32));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), lo_address, 1024));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), lo_address, 16 * 1024));
    }
}
TEST_F(DeviceFixture, TestSimpleDramBufferReadOnlyHi) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t hi_address = this->devices_.at(id)->dram_size_per_channel() - (16 * 1024);
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), hi_address, 4));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), hi_address, 8));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), hi_address, 16));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), hi_address, 32));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), hi_address, 1024));
        ASSERT_TRUE(SimpleDramReadOnly(this->devices_.at(id), hi_address, 16 * 1024));
    }
}
TEST_F(DeviceFixture, TestSimpleDramBufferWriteOnlyLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address = devices_.at(id)->allocator()->get_base_allocator_addr(HalMemType::DRAM);
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), lo_address, 4));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), lo_address, 8));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), lo_address, 16));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), lo_address, 32));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), lo_address, 1024));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), lo_address, 16 * 1024));
    }
}
TEST_F(DeviceFixture, TestSimpleDramBufferWriteOnlyHi) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t hi_address = this->devices_.at(id)->dram_size_per_channel() - (16 * 1024);
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), hi_address, 4));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), hi_address, 8));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), hi_address, 16));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), hi_address, 32));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), hi_address, 1024));
        ASSERT_TRUE(SimpleDramWriteOnly(this->devices_.at(id), hi_address, 16 * 1024));
    }
}

}  // namespace tt::tt_metal
