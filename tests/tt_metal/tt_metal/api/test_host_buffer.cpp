// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <tt_stl/span.hpp>

#include <functional>
#include <vector>

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>

namespace tt::tt_metal {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Pointwise;

TEST(HostBufferTest, Empty) {
    HostBuffer buffer;
    EXPECT_TRUE(buffer.view_bytes().empty());
}

TEST(HostBufferTest, BasicOwned) {
    HostBuffer buffer(std::vector<int>{1, 2, 3});

    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    auto writable_view = buffer.view_as<int>();
    writable_view[1] = 5;

    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));
}

TEST(HostBufferTest, BasicBorrowed) {
    int num_increments = 0;
    int num_decrements = 0;
    std::vector<int> vec = {1, 2, 3};
    HostBuffer buffer(
        tt::stl::Span<int>(vec.data(), vec.size()),
        MemoryPin([&]() { num_increments++; }, [&]() { num_decrements++; }));

    EXPECT_EQ(num_increments, 1);
    EXPECT_EQ(num_decrements, 0);
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    auto writable_view = buffer.view_as<int>();
    writable_view[1] = 5;
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));
}

TEST(HostBufferTest, IncorrectCast) {
    HostBuffer buffer(std::vector<int>{1, 2, 3});
    EXPECT_ANY_THROW({ buffer.view_as<float>(); });
}

TEST(HostBufferTest, ShallowCopy) {
    HostBuffer buffer(std::vector<int>{1, 2, 3});
    HostBuffer copy = buffer;

    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));
    EXPECT_THAT(copy.view_as<int>(), Pointwise(Eq(), {1, 2, 3}));

    auto writable_view = buffer.view_as<int>();
    writable_view[1] = 5;
    EXPECT_THAT(buffer.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));
    EXPECT_THAT(copy.view_as<int>(), Pointwise(Eq(), {1, 5, 3}));
}

}  // namespace
}  // namespace tt::tt_metal
