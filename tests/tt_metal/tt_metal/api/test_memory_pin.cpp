// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <functional>
#include <utility>

#include <tt-metalium/memory_pin.hpp>

namespace tt::tt_metal {
namespace {

using ::testing::Eq;

TEST(MemoryPinTest, Lifecycle) {
    int inc_count = 0;
    int dec_count = 0;
    {
        MemoryPin pin([&]() { inc_count++; }, [&]() { dec_count++; });
        EXPECT_EQ(inc_count, 1);
        EXPECT_EQ(dec_count, 0);
    }
    EXPECT_EQ(inc_count, 1);
    EXPECT_EQ(dec_count, 1);
}

TEST(MemoryPinTest, EmptyPin) {
    MemoryPin pin;
    EXPECT_EQ(pin, nullptr);

    int inc_count = 0;
    int dec_count = 0;
    pin = MemoryPin([&]() { inc_count++; }, [&]() { dec_count++; });

    EXPECT_NE(pin, nullptr);
}

TEST(MemoryPinTest, FromSharedPtr) {
    auto ptr = std::make_shared<int>(42);
    MemoryPin pin(ptr);

    EXPECT_EQ(ptr.use_count(), 2);

    {
        auto pin2 = pin;
        EXPECT_EQ(ptr.use_count(), 3);
    }

    EXPECT_EQ(ptr.use_count(), 2);
}

TEST(MemoryPinTest, CopyConstruction) {
    int inc_count = 0;
    int dec_count = 0;
    MemoryPin pin1([&]() { inc_count++; }, [&]() { dec_count++; });
    EXPECT_EQ(inc_count, 1);
    EXPECT_EQ(dec_count, 0);
    {
        MemoryPin pin2(pin1);
        EXPECT_EQ(inc_count, 2);
        EXPECT_EQ(dec_count, 0);
    }
    EXPECT_EQ(inc_count, 2);
    EXPECT_EQ(dec_count, 1);
}

TEST(MemoryPinTest, CopyAssignment) {
    int inc_count1 = 0;
    int dec_count1 = 0;
    int inc_count2 = 0;
    int dec_count2 = 0;

    MemoryPin pin1([&]() { inc_count1++; }, [&]() { dec_count1++; });
    EXPECT_EQ(inc_count1, 1);
    EXPECT_EQ(dec_count1, 0);

    {
        MemoryPin pin2([&]() { inc_count2++; }, [&]() { dec_count2++; });
        EXPECT_EQ(inc_count2, 1);
        EXPECT_EQ(dec_count2, 0);

        pin2 = pin1;
        EXPECT_EQ(inc_count1, 2);
        EXPECT_EQ(dec_count1, 0);
        EXPECT_EQ(inc_count2, 1);
        EXPECT_EQ(dec_count2, 1);
    }
    EXPECT_EQ(inc_count1, 2);
    EXPECT_EQ(dec_count1, 1);
    EXPECT_EQ(inc_count2, 1);
    EXPECT_EQ(dec_count2, 1);
}

TEST(MemoryPinTest, CopyAssignmentToEmpty) {
    int inc_count = 0;
    int dec_count = 0;
    MemoryPin pin1([&]() { inc_count++; }, [&]() { dec_count++; });
    EXPECT_EQ(inc_count, 1);
    EXPECT_EQ(dec_count, 0);

    {
        MemoryPin pin2;
        pin2 = pin1;
        EXPECT_EQ(inc_count, 2);
        EXPECT_EQ(dec_count, 0);
    }
    EXPECT_EQ(inc_count, 2);
    EXPECT_EQ(dec_count, 1);
}

TEST(MemoryPinTest, MoveConstruction) {
    int inc_count = 0;
    int dec_count = 0;
    MemoryPin pin1([&]() { inc_count++; }, [&]() { dec_count++; });
    EXPECT_EQ(inc_count, 1);
    EXPECT_EQ(dec_count, 0);

    {
        MemoryPin pin2(std::move(pin1));
        EXPECT_EQ(inc_count, 1);
        EXPECT_EQ(dec_count, 0);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        EXPECT_EQ(pin1, nullptr);
    }
    EXPECT_EQ(inc_count, 1);
    EXPECT_EQ(dec_count, 1);
}

TEST(MemoryPinTest, MoveAssignment) {
    int inc_count1 = 0;
    int dec_count1 = 0;
    int inc_count2 = 0;
    int dec_count2 = 0;

    MemoryPin pin1([&]() { inc_count1++; }, [&]() { dec_count1++; });
    EXPECT_EQ(inc_count1, 1);
    EXPECT_EQ(dec_count1, 0);

    {
        MemoryPin pin2([&]() { inc_count2++; }, [&]() { dec_count2++; });
        EXPECT_EQ(inc_count2, 1);
        EXPECT_EQ(dec_count2, 0);

        pin2 = std::move(pin1);
        EXPECT_EQ(inc_count1, 1);
        EXPECT_EQ(dec_count1, 0);
        EXPECT_EQ(inc_count2, 1);
        EXPECT_EQ(dec_count2, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        EXPECT_EQ(pin1, nullptr);
    }
    EXPECT_EQ(inc_count1, 1);
    EXPECT_EQ(dec_count1, 1);
    EXPECT_EQ(inc_count2, 1);
    EXPECT_EQ(dec_count2, 1);
}

TEST(MemoryPinTest, MoveAssignmentToEmpty) {
    int inc_count = 0;
    int dec_count = 0;
    MemoryPin pin1([&]() { inc_count++; }, [&]() { dec_count++; });
    EXPECT_EQ(inc_count, 1);
    EXPECT_EQ(dec_count, 0);

    {
        MemoryPin pin2;
        EXPECT_NE(pin1, nullptr);
        EXPECT_EQ(pin2, nullptr);

        pin2 = std::move(pin1);

        EXPECT_EQ(inc_count, 1);
        EXPECT_EQ(dec_count, 0);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        EXPECT_EQ(pin1, nullptr);
        EXPECT_NE(pin2, nullptr);
    }
    EXPECT_EQ(inc_count, 1);
    EXPECT_EQ(dec_count, 1);
}

}  // namespace
}  // namespace tt::tt_metal
