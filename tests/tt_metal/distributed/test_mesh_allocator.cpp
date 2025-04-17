// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <mesh_device.hpp>
#include <stddef.h>
#include <memory>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

using MeshAllocatorTest = GenericMeshDeviceFixture;

TEST_F(MeshAllocatorTest, BasicAllocationSanityCheck) {
    const size_t allocation_size = 1024 * 8;  // 1KB
    const tt::tt_metal::BufferType buffer_type = tt::tt_metal::BufferType::L1;

    auto config = InterleavedBufferConfig{
        .device = mesh_device_.get(),
        .size = allocation_size,
        .page_size = 1024,
        .buffer_type = buffer_type,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED};

    auto buffer = CreateBuffer(config);

    EXPECT_TRUE(buffer->is_allocated());
    EXPECT_EQ(buffer->size(), allocation_size);
    EXPECT_EQ(buffer->buffer_type(), buffer_type);
}

}  // namespace tt::tt_metal::distributed::test
