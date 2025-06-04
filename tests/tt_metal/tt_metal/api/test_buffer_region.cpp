// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <memory>

#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>

namespace tt::tt_metal {

TEST_F(DeviceSingleCardBufferFixture, TestInvalidBufferRegion) {
    const InterleavedBufferConfig& buffer_config{
        .device = this->device_, .size = 2048, .page_size = 32, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> buffer = CreateBuffer(buffer_config);

    const BufferRegion buffer_region1(512, 4096);
    EXPECT_FALSE(buffer.get()->is_valid_region(buffer_region1));

    const BufferRegion buffer_region2(3072, 4096);
    EXPECT_FALSE(buffer.get()->is_valid_region(buffer_region2));

    const BufferRegion buffer_region3(0, 4096);
    EXPECT_FALSE(buffer.get()->is_valid_region(buffer_region3));
}

TEST_F(DeviceSingleCardBufferFixture, TestValidBufferRegion) {
    const InterleavedBufferConfig& buffer_config{
        .device = this->device_, .size = 2048, .page_size = 32, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> buffer = CreateBuffer(buffer_config);

    const BufferRegion buffer_region1(1024, 1024);
    EXPECT_TRUE(buffer.get()->is_valid_region(buffer_region1));

    const BufferRegion buffer_region2(0, 2048);
    EXPECT_TRUE(buffer.get()->is_valid_region(buffer_region2));

    const BufferRegion buffer_region3(0, 512);
    EXPECT_TRUE(buffer.get()->is_valid_region(buffer_region3));

    const BufferRegion buffer_region4(512, 512);
    EXPECT_TRUE(buffer.get()->is_valid_region(buffer_region4));
}

TEST_F(DeviceSingleCardBufferFixture, TestPartialBufferRegion) {
    const InterleavedBufferConfig& buffer_config{
        .device = this->device_, .size = 2048, .page_size = 32, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> buffer = CreateBuffer(buffer_config);

    const BufferRegion buffer_region1(1024, 1024);
    EXPECT_TRUE(buffer.get()->is_valid_partial_region(buffer_region1));

    const BufferRegion buffer_region2(0, 1024);
    EXPECT_TRUE(buffer.get()->is_valid_partial_region(buffer_region2));

    const BufferRegion buffer_region3(512, 1024);
    EXPECT_TRUE(buffer.get()->is_valid_partial_region(buffer_region3));
}

TEST_F(DeviceSingleCardBufferFixture, TestFullBufferRegion) {
    const InterleavedBufferConfig& buffer_config{
        .device = this->device_, .size = 2048, .page_size = 32, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> buffer = CreateBuffer(buffer_config);

    const BufferRegion buffer_region(0, 2048);
    EXPECT_FALSE(buffer.get()->is_valid_partial_region(buffer_region));
}

}  // namespace tt::tt_metal
