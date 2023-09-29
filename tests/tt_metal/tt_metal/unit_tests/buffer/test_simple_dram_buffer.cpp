// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "single_device_fixture.hpp"
#include "gtest/gtest.h"
#include "test_buffer_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"


using tt::tt_metal::Device;
using namespace tt::test_utils;
using namespace tt::test::buffer::detail;

namespace tt::test::buffer::detail {
    bool SimpleDramReadOnly (Device* device, size_t local_address, size_t byte_size) {
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;
        uint32_t dram_channel = device->dram_channel_from_bank_id(0);
        writeDramBackdoor(device, dram_channel, local_address, inputs);
        readDramBackdoor(device, dram_channel, local_address, byte_size, outputs);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Channel={}, Packet Size(in Bytes)={}", dram_channel, byte_size);
        }
        return pass;
    }
    bool SimpleDramWriteOnly (Device* device, size_t local_address, size_t byte_size) {
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;
        uint32_t dram_channel = device->dram_channel_from_bank_id(0);
        writeDramBackdoor(device, dram_channel, local_address, inputs);
        readDramBackdoor(device, dram_channel, local_address, byte_size, outputs);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Channel={}, Packet Size(in Bytes)={}", dram_channel, byte_size);
        }
        return pass;
    }
}


TEST_F(SingleDeviceFixture, TestSimpleDramBufferReadOnlyLo) {
    size_t lo_address = DRAM_UNRESERVED_BASE;
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, lo_address, 4));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, lo_address, 8));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, lo_address, 16));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, lo_address, 32));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, lo_address, 1024));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, lo_address, 16*1024));
}
TEST_F(SingleDeviceFixture, TestSimpleDramBufferReadOnlyHi) {
    size_t hi_address = this->device_->dram_size_per_channel() - (16*1024);
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, hi_address, 4));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, hi_address, 8));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, hi_address, 16));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, hi_address, 32));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, hi_address, 1024));
    ASSERT_TRUE(SimpleDramReadOnly(this->device_, hi_address, 16*1024));
}
TEST_F(SingleDeviceFixture, TestSimpleDramBufferWriteOnlyLo) {
    size_t lo_address = DRAM_UNRESERVED_BASE;
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, lo_address, 4));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, lo_address, 8));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, lo_address, 16));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, lo_address, 32));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, lo_address, 1024));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, lo_address, 16*1024));
}
TEST_F(SingleDeviceFixture, TestSimpleDramBufferWriteOnlyHi) {
    size_t hi_address = this->device_->dram_size_per_channel() - (16*1024);
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, hi_address, 4));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, hi_address, 8));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, hi_address, 16));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, hi_address, 32));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, hi_address, 1024));
    ASSERT_TRUE(SimpleDramWriteOnly(this->device_, hi_address, 16*1024));
}
