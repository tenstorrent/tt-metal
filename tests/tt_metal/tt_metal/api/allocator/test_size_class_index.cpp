// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstddef>
#include <optional>
#include <vector>

#include "tt_metal/impl/allocator/algorithms/size_class_index.hpp"

namespace tt::tt_metal::allocator {
namespace {

using SearchPolicy = SizeClassIndex::SearchPolicy;
using Handles = std::vector<size_t>;

constexpr DeviceAddr KiB = 1024;
constexpr DeviceAddr MiB = 1024 * KiB;

// Blocks stored the way FreeListOpt stores them: a handle is an index into parallel vectors.
struct BlockStorage {
    std::vector<DeviceAddr> address;
    std::vector<DeviceAddr> size;

    size_t add(DeviceAddr block_address, DeviceAddr block_size) {
        address.push_back(block_address);
        size.push_back(block_size);
        return address.size() - 1;
    }
    auto address_accessor() const {
        return [this](size_t block) { return address[block]; };
    }
    auto size_accessor() const {
        return [this](size_t block) { return size[block]; };
    }
};

// An index over `blocks` that holds every block added through it.
struct IndexedBlocks {
    BlockStorage blocks;
    SizeClassIndex index{1 * MiB};

    size_t add(DeviceAddr block_address, DeviceAddr block_size) {
        const size_t block = blocks.add(block_address, block_size);
        insert(block);
        return block;
    }
    void insert(size_t block) { index.insert(block, blocks.size[block], blocks.address_accessor()); }
    std::optional<size_t> take_fit(DeviceAddr size, bool bottom_up, SearchPolicy policy) {
        return index.take_fit(size, bottom_up, policy, blocks.address_accessor(), blocks.size_accessor());
    }
    const Handles& class_of(DeviceAddr size) const { return index.blocks(index.class_of(size)); }
};

TEST(SizeClassIndexTest, ClassOf) {
    // The class count is clamped below at 2
    EXPECT_EQ(SizeClassIndex(1 * KiB).num_classes(), 2);
    // 1 MiB / 1 KiB = 2^10, which takes 11 bits
    EXPECT_EQ(SizeClassIndex(1 * MiB).num_classes(), 11);
    // The class count is capped at the class of 128 MiB
    EXPECT_EQ(SizeClassIndex(1024 * MiB).num_classes(), SizeClassIndex(128 * MiB).num_classes());
    // The class is floor(log2(size / base_size)), or 0 below base_size
    const SizeClassIndex index(1 * MiB);
    EXPECT_EQ(index.class_of(0), 0);
    EXPECT_EQ(index.class_of(1 * KiB), 0);
    EXPECT_EQ(index.class_of(2 * KiB - 1), 0);
    EXPECT_EQ(index.class_of(2 * KiB), 1);
    EXPECT_EQ(index.class_of(4 * KiB), 2);
    // The last class holds every larger size
    EXPECT_EQ(index.class_of(1024 * MiB), index.num_classes() - 1);
}

TEST(SizeClassIndexTest, Insert) {
    // Keeps a class sorted by address. More than 30 blocks also exercises the binary search.
    IndexedBlocks index;
    for (DeviceAddr step = 0; step < 40; ++step) {
        index.add(((step * 17) % 40) * 4 * KiB, 2 * KiB);
    }
    const Handles& in_class = index.class_of(2 * KiB);
    ASSERT_EQ(in_class.size(), 40);
    for (size_t position = 1; position < in_class.size(); ++position) {
        EXPECT_LT(index.blocks.address[in_class[position - 1]], index.blocks.address[in_class[position]]);
    }
}

TEST(SizeClassIndexTest, Erase) {
    // Removes only that block, found by binary search or by a scan
    IndexedBlocks index;
    const size_t low = index.add(0, 2 * KiB);
    const size_t middle = index.add(8 * KiB, 2 * KiB);
    const size_t high = index.add(16 * KiB, 2 * KiB);
    EXPECT_TRUE(index.index.erase(middle, 2 * KiB, index.blocks.address_accessor()));
    EXPECT_EQ(index.class_of(2 * KiB), (Handles{low, high}));
    EXPECT_FALSE(index.index.erase(middle, 2 * KiB, index.blocks.address_accessor()));
    EXPECT_TRUE(index.index.erase_by_scan(high, 2 * KiB));
    EXPECT_EQ(index.class_of(2 * KiB), (Handles{low}));
    EXPECT_FALSE(index.index.erase_by_scan(high, 2 * KiB));
    // Looks only in the class of the given size
    EXPECT_FALSE(index.index.erase(low, 8 * KiB, index.blocks.address_accessor()));
    EXPECT_FALSE(index.index.erase_by_scan(low, 8 * KiB));
    EXPECT_EQ(index.class_of(2 * KiB), (Handles{low}));
}

TEST(SizeClassIndexTest, TakeFit) {
    {
        // Nothing fits, and nothing is removed
        IndexedBlocks index;
        index.add(0, 2 * KiB);
        EXPECT_EQ(index.take_fit(3 * KiB, true, SearchPolicy::BEST), std::nullopt);
        EXPECT_EQ(index.take_fit(3 * KiB, false, SearchPolicy::FIRST), std::nullopt);
        EXPECT_EQ(index.class_of(2 * KiB).size(), 1);
    }
    {
        // BEST takes from the lowest class that fits
        IndexedBlocks index;
        const size_t large = index.add(0, 64 * KiB);
        const size_t small = index.add(100 * KiB, 4 * KiB);
        EXPECT_EQ(index.take_fit(4 * KiB, true, SearchPolicy::BEST), small);
        EXPECT_EQ(index.class_of(4 * KiB), Handles{});
        EXPECT_EQ(index.take_fit(4 * KiB, true, SearchPolicy::BEST), large);
    }
    {
        // BEST scans a class in address order, or in reverse for top-down. Both blocks are in the
        // class [4 KiB, 8 KiB).
        IndexedBlocks index;
        const size_t low = index.add(0, 6 * KiB);
        const size_t high = index.add(100 * KiB, 5 * KiB);
        EXPECT_EQ(index.take_fit(4 * KiB, true, SearchPolicy::BEST), low);
        index.insert(low);
        EXPECT_EQ(index.take_fit(4 * KiB, false, SearchPolicy::BEST), high);
    }
    {
        // FIRST takes the lowest address across classes, or the highest for top-down
        IndexedBlocks index;
        const size_t low_large = index.add(0, 64 * KiB);
        const size_t middle_small = index.add(100 * KiB, 4 * KiB);
        const size_t high_large = index.add(200 * KiB, 64 * KiB);
        EXPECT_EQ(index.take_fit(4 * KiB, true, SearchPolicy::FIRST), low_large);
        EXPECT_EQ(index.take_fit(4 * KiB, false, SearchPolicy::FIRST), high_large);
        EXPECT_EQ(index.take_fit(4 * KiB, false, SearchPolicy::FIRST), middle_small);
    }
}

TEST(SizeClassIndexTest, Clear) {
    // Empties every class
    IndexedBlocks index;
    index.add(0, 2 * KiB);
    index.add(8 * KiB, 64 * KiB);
    index.index.clear();
    for (size_t size_class = 0; size_class < index.index.num_classes(); ++size_class) {
        EXPECT_EQ(index.index.blocks(size_class), Handles{});
    }
}

}  // namespace
}  // namespace tt::tt_metal::allocator
