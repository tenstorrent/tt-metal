// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <mesh_device.hpp>
#include <cstddef>
#include <memory>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {

using MeshAllocatorTest = GenericMeshDeviceFixture;

TEST_F(MeshAllocatorTest, BasicAllocationSanityCheck) {
    const size_t allocation_size = 1024 * 8;  // 1KB
    const tt::tt_metal::BufferType buffer_type = tt::tt_metal::BufferType::L1;

    auto buffer = MeshBuffer::create(
        ReplicatedBufferConfig{.size = allocation_size},
        {.page_size = 1024, .buffer_type = buffer_type},
        mesh_device_.get());

    EXPECT_TRUE(buffer->is_allocated());
    EXPECT_EQ(buffer->size(), allocation_size);
    EXPECT_EQ(buffer->device_local_config().buffer_type, buffer_type);
}

}  // namespace tt::tt_metal::distributed::test
