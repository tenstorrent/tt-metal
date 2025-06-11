// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <tt_stl/span.hpp>

#include <functional>
#include <vector>

#include <tt-metalium/distributed_host_buffer.hpp>

namespace tt::tt_metal {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

TEST(DistributedHostBufferTest, InvalidLocalShape) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(2, 4);  // 2x4 > 2x3
    distributed::MeshCoordinate local_offset(0, 0);

    EXPECT_ANY_THROW(DistributedHostBuffer::create(global_shape, local_shape, local_offset));
}

TEST(DistributedHostBufferTest, InvalidLocalOffset) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(1, 2);
    distributed::MeshCoordinate local_offset(2, 0);  // Offset 2 in first dimension exceeds global shape

    EXPECT_ANY_THROW(DistributedHostBuffer::create(global_shape, local_shape, local_offset));
}

TEST(DistributedHostBufferTest, InvalidCombination) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(1, 2);
    distributed::MeshCoordinate local_offset(1, 2);  // Offset + shape exceeds global shape

    EXPECT_ANY_THROW(DistributedHostBuffer::create(global_shape, local_shape, local_offset));
}

TEST(DistributedHostBufferTest, DimensionMismatch) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(1, 1, 1);  // 3D shape vs 2D global shape
    distributed::MeshCoordinate local_offset(1, 0);

    EXPECT_ANY_THROW(DistributedHostBuffer::create(global_shape, local_shape, local_offset));
}

TEST(DistributedHostBufferTest, EmplaceAndGetBuffer) {
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(3));

    buffer.emplace_shard(distributed::MeshCoordinate(0), []() { return HostBuffer(std::vector<int>{1, 2, 3}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });
    buffer.emplace_shard(distributed::MeshCoordinate(2), []() { return HostBuffer(std::vector<int>{7, 8, 9}); });

    auto optional_buffer = buffer.get_shard(distributed::MeshCoordinate(0));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(1));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {4, 5, 6}));

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(2));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {7, 8, 9}));

    // Out of global bounds.
    EXPECT_ANY_THROW(buffer.get_shard(distributed::MeshCoordinate(3)));
}

TEST(DistributedHostBufferTest, EmplaceAndGetBufferWithUnpopulatedLocalShards) {
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(3));

    buffer.emplace_shard(distributed::MeshCoordinate(1), []() { return HostBuffer(std::vector<int>{1, 2, 3}); });

    auto optional_buffer = buffer.get_shard(distributed::MeshCoordinate(0));
    EXPECT_FALSE(optional_buffer.has_value());

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(1));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(2));
    EXPECT_FALSE(optional_buffer.has_value());

    // Out of global bounds.
    EXPECT_ANY_THROW(buffer.get_shard(distributed::MeshCoordinate(3)));
}

TEST(DistributedHostBufferTest, EmplaceAndGetBufferWithLocalMeshShape) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(2, 1);
    distributed::MeshCoordinate local_offset(0, 1);

    auto buffer = DistributedHostBuffer::create(global_shape, local_shape, local_offset);
    EXPECT_EQ(buffer.shape().mesh_size(), 6);  // 2×3 = 6

    buffer.emplace_shard(distributed::MeshCoordinate(0, 0), []() { return HostBuffer(std::vector<int>{1, 2, 3}); });
    buffer.emplace_shard(distributed::MeshCoordinate(0, 1), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });
    buffer.emplace_shard(distributed::MeshCoordinate(0, 2), []() { return HostBuffer(std::vector<int>{7, 8, 9}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1, 0), []() { return HostBuffer(std::vector<int>{10, 11, 12}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1, 1), []() { return HostBuffer(std::vector<int>{13, 14, 15}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1, 2), []() { return HostBuffer(std::vector<int>{16, 17, 18}); });

    EXPECT_THAT(
        buffer.shard_coords(),
        UnorderedElementsAre(
            distributed::MeshCoordinate(0, 0),
            distributed::MeshCoordinate(0, 1),
            distributed::MeshCoordinate(0, 2),
            distributed::MeshCoordinate(1, 0),
            distributed::MeshCoordinate(1, 1),
            distributed::MeshCoordinate(1, 2)));

    auto optional_buffer = buffer.get_shard(distributed::MeshCoordinate(0, 0));
    EXPECT_FALSE(optional_buffer.has_value());

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(0, 1));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {4, 5, 6}));

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(0, 2));
    EXPECT_FALSE(optional_buffer.has_value());

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(1, 0));
    EXPECT_FALSE(optional_buffer.has_value());

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(1, 1));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {13, 14, 15}));

    optional_buffer = buffer.get_shard(distributed::MeshCoordinate(1, 2));
    EXPECT_FALSE(optional_buffer.has_value());

    // Out of global bounds.
    EXPECT_ANY_THROW(buffer.get_shard(distributed::MeshCoordinate(2, 0)));
}

TEST(DistributedHostBufferTest, Transform) {
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(2));
    EXPECT_EQ(buffer.shape().mesh_size(), 2);

    buffer.emplace_shard(distributed::MeshCoordinate(0), []() { return HostBuffer(std::vector<int>{1, 2, 3}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });

    auto transformed_buffer = buffer.transform([](const HostBuffer& buffer) {
        auto span = buffer.view_as<int>();
        std::vector<int> new_data(span.begin(), span.end());
        for (auto& val : new_data) {
            val *= 2;
        }
        return HostBuffer(std::move(new_data));
    });

    auto optional_buffer = transformed_buffer.get_shard(distributed::MeshCoordinate(0));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {2, 4, 6}));

    optional_buffer = transformed_buffer.get_shard(distributed::MeshCoordinate(1));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {8, 10, 12}));
}

TEST(DistributedHostBufferTest, TransformWithUnpopulatedLocalShards) {
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(3));

    buffer.emplace_shard(distributed::MeshCoordinate(1), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });

    auto transformed_buffer = buffer.transform([](const HostBuffer& buffer) {
        auto span = buffer.view_as<int>();
        std::vector<int> new_data(span.begin(), span.end());
        for (auto& val : new_data) {
            val *= 2;
        }
        return HostBuffer(std::move(new_data));
    });

    auto optional_buffer = transformed_buffer.get_shard(distributed::MeshCoordinate(0));
    ASSERT_FALSE(optional_buffer.has_value());

    optional_buffer = transformed_buffer.get_shard(distributed::MeshCoordinate(1));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {8, 10, 12}));

    optional_buffer = transformed_buffer.get_shard(distributed::MeshCoordinate(2));
    ASSERT_FALSE(optional_buffer.has_value());
}

TEST(DistributedHostBufferTest, TransformWithLocalShape) {
    distributed::MeshShape global_shape(3, 1);
    distributed::MeshShape local_shape(2, 1);
    distributed::MeshCoordinate local_offset(1, 0);

    auto buffer = DistributedHostBuffer::create(global_shape, local_shape, local_offset);
    EXPECT_EQ(buffer.shape().mesh_size(), 3);  // 3×1 = 3

    buffer.emplace_shard(distributed::MeshCoordinate(0, 0), []() { return HostBuffer(std::vector<int>{1, 2, 3}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1, 0), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });
    buffer.emplace_shard(distributed::MeshCoordinate(2, 0), []() { return HostBuffer(std::vector<int>{7, 8, 9}); });

    auto transformed_buffer = buffer.transform([](const HostBuffer& buffer) {
        auto span = buffer.view_as<int>();
        std::vector<int> new_data(span.begin(), span.end());
        for (auto& val : new_data) {
            val *= 2;
        }
        return HostBuffer(std::move(new_data));
    });

    auto optional_buffer = transformed_buffer.get_shard(distributed::MeshCoordinate(1, 0));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {8, 10, 12}));

    optional_buffer = transformed_buffer.get_shard(distributed::MeshCoordinate(2, 0));
    ASSERT_TRUE(optional_buffer.has_value());
    EXPECT_THAT(optional_buffer->view_as<int>(), Pointwise(Eq(), {14, 16, 18}));
}

TEST(DistributedHostBufferTest, Apply) {
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(2));
    EXPECT_EQ(buffer.shape().mesh_size(), 2);

    buffer.emplace_shard(distributed::MeshCoordinate(0), []() { return HostBuffer(std::vector<int>{1, 2, 3}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });

    std::vector<std::vector<int>> values;
    buffer.apply([&values](const HostBuffer& buffer) {
        auto span = buffer.view_as<int>();
        values.push_back(std::vector<int>(span.begin(), span.end()));
    });

    ASSERT_THAT(values, SizeIs(2));
    EXPECT_THAT(values[0], ElementsAre(1, 2, 3));
    EXPECT_THAT(values[1], ElementsAre(4, 5, 6));
}

TEST(DistributedHostBufferTest, ApplyWithUnpopulatedLocalShards) {
    auto buffer = DistributedHostBuffer::create(distributed::MeshShape(3));

    buffer.emplace_shard(distributed::MeshCoordinate(1), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });

    std::vector<std::vector<int>> values;
    buffer.apply([&values](const HostBuffer& buffer) {
        auto span = buffer.view_as<int>();
        values.push_back(std::vector<int>(span.begin(), span.end()));
    });

    ASSERT_THAT(values, SizeIs(1));
    EXPECT_THAT(values[0], ElementsAre(4, 5, 6));
}

TEST(DistributedHostBufferTest, ApplyWithLocalShape) {
    distributed::MeshShape global_shape(3, 1);
    distributed::MeshShape local_shape(2, 1);
    distributed::MeshCoordinate local_offset(1, 0);

    auto buffer = DistributedHostBuffer::create(global_shape, local_shape, local_offset);
    EXPECT_EQ(buffer.shape().mesh_size(), 3);  // 3×1 = 3

    buffer.emplace_shard(distributed::MeshCoordinate(0, 0), []() { return HostBuffer(std::vector<int>{1, 2, 3}); });
    buffer.emplace_shard(distributed::MeshCoordinate(1, 0), []() { return HostBuffer(std::vector<int>{4, 5, 6}); });
    buffer.emplace_shard(distributed::MeshCoordinate(2, 0), []() { return HostBuffer(std::vector<int>{7, 8, 9}); });

    std::vector<std::vector<int>> values;
    buffer.apply([&values](const HostBuffer& buffer) {
        auto span = buffer.view_as<int>();
        values.push_back(std::vector<int>(span.begin(), span.end()));
    });

    ASSERT_THAT(values, SizeIs(2));
    EXPECT_THAT(values[0], ElementsAre(4, 5, 6));  // Global index 1
    EXPECT_THAT(values[1], ElementsAre(7, 8, 9));  // Global index 2
}

}  // namespace
}  // namespace tt::tt_metal
