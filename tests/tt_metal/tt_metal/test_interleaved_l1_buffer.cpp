// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
using namespace tt;
using namespace tt::tt_metal;

namespace {

void test_interleaved_l1_buffer_impl(
    distributed::MeshDevice& device, int num_pages_one, int num_pages_two, uint32_t page_size) {
    uint32_t buffer_size = num_pages_one * page_size;

    auto interleaved_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_size},
        {.page_size = page_size, .buffer_type = BufferType::L1},
        &device);

    std::vector<uint32_t> host_buffer =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    slow_dispatch::WriteToBuffer(*interleaved_buffer, host_buffer);

    std::vector<uint32_t> readback_buffer;
    slow_dispatch::ReadFromBuffer(*interleaved_buffer, readback_buffer);

    EXPECT_EQ(host_buffer, readback_buffer);

    uint32_t second_buffer_size = num_pages_two * page_size;

    auto second_interleaved_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = second_buffer_size},
        {.page_size = page_size, .buffer_type = BufferType::L1},
        &device);

    std::vector<uint32_t> second_host_buffer = create_random_vector_of_bfloat16(
        second_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    slow_dispatch::WriteToBuffer(*second_interleaved_buffer, second_host_buffer);

    std::vector<uint32_t> second_readback_buffer;
    slow_dispatch::ReadFromBuffer(*second_interleaved_buffer, second_readback_buffer);

    EXPECT_EQ(second_host_buffer, second_readback_buffer);
}

}  // namespace

TEST_F(UnitMeshFixture, InterleavedL1Buffer) {
    uint32_t page_size = 2 * 1024;
    int num_bank_pages_one = 258;
    int num_bank_pages_two = 378;

    test_interleaved_l1_buffer_impl(this->device(), num_bank_pages_one, num_bank_pages_two, page_size);
}
