// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <tt_stl/span.hpp>

#include <functional>
#include <vector>

#include "ttnn/tensor/host_buffer/host_buffer.hpp"
#include "ttnn/tensor/host_buffer/memory_pin.hpp"

namespace tt::tt_metal {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Pointwise;

TEST(HostBufferTest, Empty) {
    HostBuffer buffer;

    EXPECT_FALSE(buffer.is_allocated());
    EXPECT_FALSE(buffer.is_borrowed());
    EXPECT_TRUE(buffer.view_bytes().empty());
}

TEST(HostBufferTest, OwnedLifecycle) {
    HostBuffer buffer(std::vector<int>{1, 2, 3});

    EXPECT_TRUE(buffer.is_allocated());
    EXPECT_FALSE(buffer.is_borrowed());
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    auto writable_view = buffer.view_as<int>();
    writable_view[1] = 5;

    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));

    buffer.deallocate();

    EXPECT_FALSE(buffer.is_allocated());
    EXPECT_TRUE(buffer.view_bytes().empty());
}

TEST(HostBufferTest, IncorrectCast) {
    HostBuffer buffer(std::vector<int>{1, 2, 3});
    EXPECT_ANY_THROW({ buffer.view_as<float>(); });
}

TEST(HostBufferTest, BorrowedLifecycle) {
    int num_increments = 0;
    int num_decrements = 0;
    std::vector<int> vec = {1, 2, 3};
    HostBuffer buffer(
        tt::stl::Span<int>(vec.data(), vec.size()),
        MemoryPin([&]() { num_increments++; }, [&]() { num_decrements++; }));

    EXPECT_EQ(num_increments, 1);
    EXPECT_EQ(num_decrements, 0);
    EXPECT_TRUE(buffer.is_allocated());
    EXPECT_TRUE(buffer.is_borrowed());
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    auto writable_view = buffer.view_as<int>();
    writable_view[1] = 5;
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));

    buffer.deallocate();

    EXPECT_EQ(num_increments, 1);
    EXPECT_EQ(num_decrements, 1);
    EXPECT_FALSE(buffer.is_allocated());
    EXPECT_FALSE(buffer.is_borrowed());
    EXPECT_TRUE(buffer.view_bytes().empty());
}

TEST(HostBufferTest, MakePinFromOwned) {
    HostBuffer buffer(std::vector<int>{1, 2, 3});
    HostBuffer borrowed(buffer.view_as<int>(), buffer.pin());

    EXPECT_FALSE(buffer.is_borrowed());
    EXPECT_TRUE(borrowed.is_borrowed());
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));
    EXPECT_THAT(borrowed.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    auto writable_view = buffer.view_as<int>();
    writable_view[1] = 5;
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));
    EXPECT_THAT(borrowed.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));

    buffer.deallocate();

    EXPECT_FALSE(buffer.is_allocated());
    EXPECT_TRUE(borrowed.is_allocated());
    EXPECT_THAT(borrowed.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));
}

TEST(HostBufferTest, MakePinFromBorrowed) {
    int num_increments = 0;
    int num_decrements = 0;
    std::vector<int> vec = {1, 2, 3};
    HostBuffer buffer(
        tt::stl::Span<int>(vec.data(), vec.size()),
        MemoryPin([&]() { num_increments++; }, [&]() { num_decrements++; }));

    EXPECT_EQ(num_increments, 1);
    EXPECT_EQ(num_decrements, 0);
    {
        HostBuffer borrowed(buffer.view_bytes(), buffer.pin());
        EXPECT_EQ(num_increments, 2);
        EXPECT_EQ(num_decrements, 0);
    }

    EXPECT_EQ(num_increments, 2);
    EXPECT_EQ(num_decrements, 1);
}
}  // namespace
}  // namespace tt::tt_metal
