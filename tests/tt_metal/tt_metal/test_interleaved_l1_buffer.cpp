// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

void test_interleaved_l1_buffer_impl(IDevice* dev, int num_pages_one, int num_pages_two, uint32_t page_size) {
    uint32_t buffer_size = num_pages_one * page_size;

    InterleavedBufferConfig buff_config_0{
        .device = dev, .size = buffer_size, .page_size = page_size, .buffer_type = BufferType::L1};
    auto interleaved_buffer = CreateBuffer(buff_config_0);

    std::vector<uint32_t> host_buffer =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    detail::WriteToBuffer(interleaved_buffer, host_buffer);

    std::vector<uint32_t> readback_buffer;
    detail::ReadFromBuffer(interleaved_buffer, readback_buffer);

    EXPECT_EQ(host_buffer, readback_buffer);

    uint32_t second_buffer_size = num_pages_two * page_size;

    InterleavedBufferConfig buff_config_1{
        .device = dev, .size = second_buffer_size, .page_size = page_size, .buffer_type = BufferType::L1};

    auto second_interleaved_buffer = CreateBuffer(buff_config_1);

    std::vector<uint32_t> second_host_buffer = create_random_vector_of_bfloat16(
        second_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    detail::WriteToBuffer(second_interleaved_buffer, second_host_buffer);

    std::vector<uint32_t> second_readback_buffer;
    detail::ReadFromBuffer(second_interleaved_buffer, second_readback_buffer);

    EXPECT_EQ(second_host_buffer, second_readback_buffer);
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, InterleavedL1Buffer) {
    uint32_t page_size = 2 * 1024;
    int num_bank_pages_one = 258;
    int num_bank_pages_two = 378;

    test_interleaved_l1_buffer_impl(devices_[0]->get_devices()[0], num_bank_pages_one, num_bank_pages_two, page_size);
}
