// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <tt_stl/span.hpp>

#include <functional>
#include <vector>

#include <tt-metalium/host_mesh_buffer.hpp>

namespace tt::tt_metal {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::SizeIs;

TEST(HostMeshBufferTest, Empty) {
    auto buffer = HostMeshBuffer::create_replicated(HostBuffer());

    EXPECT_FALSE(buffer.is_allocated());
}

TEST(HostMeshBufferTest, OwnedLifecycle) {
    auto buffer = HostMeshBuffer::create_replicated(HostBuffer(std::vector<int>{1, 2, 3}));

    EXPECT_TRUE(buffer.is_allocated());

    buffer.deallocate();

    EXPECT_FALSE(buffer.is_allocated());
}

TEST(HostMeshBufferTest, ShardedWithInvalidLocalShape) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(2, 4);  // 2x4 > 2x3
    distributed::MeshCoordinate local_offset(0, 0);

    EXPECT_ANY_THROW(HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset));
}

TEST(HostMeshBufferTest, ShardedWithInvalidLocalOffset) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(1, 2);
    distributed::MeshCoordinate local_offset(2, 0);  // Offset 2 in first dimension exceeds global shape

    EXPECT_ANY_THROW(HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset));
}

TEST(HostMeshBufferTest, ShardedWithInvalidCombination) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(1, 2);
    distributed::MeshCoordinate local_offset(1, 2);  // Offset + shape exceeds global shape

    EXPECT_ANY_THROW(HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset));
}

TEST(HostMeshBufferTest, ShardedWithDimensionMismatch) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(1, 1, 1);  // 3D shape vs 2D global shape
    distributed::MeshCoordinate local_offset(1, 0);

    EXPECT_ANY_THROW(HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset));
}

TEST(HostMeshBufferTest, ReplicatedGetBuffer) {
    auto buffer = HostMeshBuffer::create_replicated(HostBuffer(std::vector<int>{1, 2, 3}));

    auto optional_buffer = buffer.get_buffer(0);
    ASSERT_TRUE(optional_buffer.has_value());

    auto span = optional_buffer->view_as<int>();
    EXPECT_THAT(span, Pointwise(Eq(), {1, 2, 3}));

    optional_buffer = buffer.get_buffer(1);
    ASSERT_TRUE(optional_buffer.has_value());
    span = optional_buffer->view_as<int>();
    EXPECT_THAT(span, Pointwise(Eq(), {1, 2, 3}));
}

TEST(HostMeshBufferTest, ShardedGetBuffer) {
    auto buffer = HostMeshBuffer::create_sharded(3);

    buffer.emplace_buffer(0, HostBuffer(std::vector<int>{1, 2, 3}));
    buffer.emplace_buffer(1, HostBuffer(std::vector<int>{4, 5, 6}));
    buffer.emplace_buffer(2, HostBuffer(std::vector<int>{7, 8, 9}));

    auto optional_buffer = buffer.get_buffer(0);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    optional_buffer = buffer.get_buffer(1);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {4, 5, 6}));

    optional_buffer = buffer.get_buffer(2);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {7, 8, 9}));

    // Out of global bounds.
    EXPECT_ANY_THROW(buffer.get_buffer(3));
}

TEST(HostMeshBufferTest, ShardedWithMeshShape) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(2, 3);  // Full shape
    distributed::MeshCoordinate local_offset(0, 0);

    auto buffer = HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset);

    for (int i = 0; i < 6; ++i) {
        buffer.emplace_buffer(i, HostBuffer(std::vector<int>{i * 3 + 1, i * 3 + 2, i * 3 + 3}));
    }

    for (int i = 0; i < 6; ++i) {
        auto optional_buffer = buffer.get_buffer(i);
        ASSERT_TRUE(optional_buffer.has_value());
        EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {i * 3 + 1, i * 3 + 2, i * 3 + 3}));
    }

    // Out of global bounds.
    EXPECT_ANY_THROW(buffer.get_buffer(6));
}

TEST(HostMeshBufferTest, ShardedWithLocalMeshShape) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(1, 2);
    distributed::MeshCoordinate local_offset(1, 0);

    auto buffer = HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset);

    // Create and emplace all buffers
    for (int i = 0; i < 6; ++i) {
        // Try to emplace all buffers - only the ones in our local shard will succeed
        buffer.emplace_buffer(i, HostBuffer(std::vector<int>{i * 3 + 1, i * 3 + 2, i * 3 + 3}));
    }

    auto optional_buffer = buffer.get_buffer(0);
    EXPECT_FALSE(optional_buffer.has_value());

    optional_buffer = buffer.get_buffer(1);
    EXPECT_FALSE(optional_buffer.has_value());

    optional_buffer = buffer.get_buffer(2);
    EXPECT_FALSE(optional_buffer.has_value());

    optional_buffer = buffer.get_buffer(3);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {10, 11, 12}));

    optional_buffer = buffer.get_buffer(4);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {13, 14, 15}));

    optional_buffer = buffer.get_buffer(5);
    EXPECT_FALSE(optional_buffer.has_value());

    // Out of global bounds.
    EXPECT_ANY_THROW(buffer.get_buffer(6));
}

TEST(HostMeshBufferTest, ReplicatedTransform) {
    auto buffer = HostMeshBuffer::create_replicated(HostBuffer(std::vector<int>{1, 2, 3}));

    std::vector<size_t> indices;
    buffer.transform([&indices](const HostBuffer& buffer, size_t index) {
        indices.push_back(index);
        auto span = buffer.view_as<int>();
        std::vector<int> new_data(span.begin(), span.end());
        for (auto& val : new_data) {
            val *= 2;
        }
        return HostBuffer(std::move(new_data));
    });

    EXPECT_THAT(indices, ElementsAre(0));

    auto optional_buffer = buffer.get_buffer(0);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {2, 4, 6}));
}

TEST(HostMeshBufferTest, ShardedTransform) {
    auto buffer = HostMeshBuffer::create_sharded(2);

    buffer.emplace_buffer(0, HostBuffer(std::vector<int>{1, 2, 3}));
    buffer.emplace_buffer(1, HostBuffer(std::vector<int>{4, 5, 6}));

    std::vector<size_t> indices;
    buffer.transform([&indices](const HostBuffer& buffer, size_t index) {
        indices.push_back(index);
        auto span = buffer.view_as<int>();
        std::vector<int> new_data(span.begin(), span.end());
        for (auto& val : new_data) {
            val *= 2;
        }
        return HostBuffer(std::move(new_data));
    });

    EXPECT_THAT(indices, ElementsAre(0, 1));

    auto optional_buffer = buffer.get_buffer(0);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {2, 4, 6}));

    optional_buffer = buffer.get_buffer(1);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {8, 10, 12}));
}

TEST(HostMeshBufferTest, ShardedWithLocalShapeTransform) {
    distributed::MeshShape global_shape(3, 1);
    distributed::MeshShape local_shape(2, 1);
    distributed::MeshCoordinate local_offset(1, 0);

    auto buffer = HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset);

    // Emplace buffers - only indices 1 and 2 are within our local shard
    for (int i = 0; i < 3; ++i) {
        buffer.emplace_buffer(i, HostBuffer(std::vector<int>{i * 3 + 1, i * 3 + 2, i * 3 + 3}));
    }

    std::vector<size_t> indices;
    buffer.transform([&indices](const HostBuffer& buffer, size_t index) {
        indices.push_back(index);
        auto span = buffer.view_as<int>();
        std::vector<int> new_data(span.begin(), span.end());
        for (auto& val : new_data) {
            val *= 2;
        }
        return HostBuffer(std::move(new_data));
    });

    EXPECT_THAT(indices, ElementsAre(0, 1));  // Local indices, not global

    auto optional_buffer = buffer.get_buffer(1);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {8, 10, 12}));

    optional_buffer = buffer.get_buffer(2);
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {14, 16, 18}));
}

TEST(HostMeshBufferTest, ReplicatedApply) {
    auto buffer = HostMeshBuffer::create_replicated(HostBuffer(std::vector<int>{1, 2, 3}));

    std::vector<size_t> indices;
    std::vector<std::vector<int>> values;

    buffer.apply([&indices, &values](const HostBuffer& buffer, size_t index) {
        indices.push_back(index);
        auto span = buffer.view_as<int>();
        values.push_back(std::vector<int>(span.begin(), span.end()));
    });

    EXPECT_THAT(indices, ElementsAre(0));
    ASSERT_THAT(values, SizeIs(1));
    EXPECT_THAT(values[0], ElementsAre(1, 2, 3));
}

TEST(HostMeshBufferTest, ShardedApply) {
    auto buffer = HostMeshBuffer::create_sharded(2);

    buffer.emplace_buffer(0, HostBuffer(std::vector<int>{1, 2, 3}));
    buffer.emplace_buffer(1, HostBuffer(std::vector<int>{4, 5, 6}));

    std::vector<size_t> indices;
    std::vector<std::vector<int>> values;
    buffer.apply([&indices, &values](const HostBuffer& buffer, size_t index) {
        indices.push_back(index);
        auto span = buffer.view_as<int>();
        values.push_back(std::vector<int>(span.begin(), span.end()));
    });

    EXPECT_THAT(indices, ElementsAre(0, 1));
    ASSERT_THAT(values, SizeIs(2));
    EXPECT_THAT(values[0], ElementsAre(1, 2, 3));
    EXPECT_THAT(values[1], ElementsAre(4, 5, 6));
}

TEST(HostMeshBufferTest, ShardedWithLocalShapeApply) {
    distributed::MeshShape global_shape(3, 1);
    distributed::MeshShape local_shape(2, 1);
    distributed::MeshCoordinate local_offset(1, 0);

    auto buffer = HostMeshBuffer::create_sharded(global_shape, local_shape, local_offset);

    // Emplace buffers - only indices 1 and 2 are within our local shard
    for (int i = 0; i < 3; ++i) {
        buffer.emplace_buffer(i, HostBuffer(std::vector<int>{i * 3 + 1, i * 3 + 2, i * 3 + 3}));
    }

    std::vector<size_t> indices;
    std::vector<std::vector<int>> values;
    buffer.apply([&indices, &values](const HostBuffer& buffer, size_t index) {
        indices.push_back(index);
        auto span = buffer.view_as<int>();
        values.push_back(std::vector<int>(span.begin(), span.end()));
    });

    EXPECT_THAT(indices, ElementsAre(0, 1));  // Local indices
    ASSERT_THAT(values, SizeIs(2));
    EXPECT_THAT(values[0], ElementsAre(4, 5, 6));  // Global index 1
    EXPECT_THAT(values[1], ElementsAre(7, 8, 9));  // Global index 2
}

}  // namespace
}  // namespace tt::tt_metal
