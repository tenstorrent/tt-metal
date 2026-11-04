// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>

#include "hal_types.hpp"

namespace tt::tt_metal::allocator {

/// Free blocks grouped by size class, so a search for a large block skips the small ones. The idea
/// comes from the TLSF paper, but instead of aiming for realtime the goal is to not look at small
/// blocks when allocating large blocks, which a naive free list does not do.
///
/// Blocks are opaque handles, such as indices into the owner's block storage. The owner passes the
/// address and size of a handle as accessors, which are templates so the hot path inlines them.
/// The search and removals are always_inline: FreeListOpt was measurably slower when the compiler
/// chose to call them instead.
class SizeClassIndex {
public:
    /// How take_fit() picks among the blocks that fit.
    enum class SearchPolicy {
        FIRST,
        BEST,
    };

    /// Configuring the base and the class count needs real world data, but for now they are numbers
    /// pulled out of thin air. Too low and it devolves into an array search, too high you pay cache
    /// misses.
    static constexpr DeviceAddr base_size = 1024;

    /// Creates empty classes for blocks of up to `max_size_bytes`.
    explicit SizeClassIndex(DeviceAddr max_size_bytes);

    size_t num_classes() const { return num_classes_; }

    /// Class 0 holds sizes below 2 * base_size, class i > 0 holds [base_size * 2^i, base_size * 2^(i+1)),
    /// and the last class also holds every larger size.
    size_t class_of(DeviceAddr size) const {
        const DeviceAddr multiple = size / base_size;
        const size_t floor_log2 = multiple == 0 ? 0 : static_cast<size_t>(std::bit_width(multiple)) - 1;
        return std::min(num_classes_ - 1, floor_log2);
    }

    /// Blocks of a class, sorted by address.
    const std::vector<size_t>& blocks(size_t size_class) const { return classes_[size_class]; }

    void reserve(size_t blocks_per_class);
    void clear();

    /// Inserts the block into the class of `size`, keeping the class sorted by address.
    template <typename AddressOf>
    __attribute__((always_inline)) void insert(size_t block, DeviceAddr size, AddressOf address_of) {
        auto& free_blocks = classes_[class_of(size)];
        // Sorted insertion is slower than pushing to the back, but pushing to the back increases
        // fragmentation.
        // from experience, the lower bound is only faster after a certain number of elements
        constexpr size_t linear_search_limit = 30;
        const auto after_block = [&](size_t other) { return address_of(other) > address_of(block); };
        const auto by_address = [&](size_t a, size_t b) { return address_of(a) < address_of(b); };
        const auto iter =  //
            free_blocks.size() < linear_search_limit
                ? std::find_if(free_blocks.begin(), free_blocks.end(), after_block)
                : std::lower_bound(free_blocks.begin(), free_blocks.end(), block, by_address);
        free_blocks.insert(iter, block);
    }

    /// Removes the block from the class of `size`, found by binary search on its address. Returns
    /// false when the block is not in that class.
    template <typename AddressOf>
    __attribute__((always_inline)) bool erase(size_t block, DeviceAddr size, AddressOf address_of) {
        auto& free_blocks = classes_[class_of(size)];
        const auto by_address = [&](size_t a, size_t b) { return address_of(a) < address_of(b); };
        // Assert (debug-only) that insert() kept the class sorted, which the lower_bound relies on.
        TT_ASSERT(
            std::is_sorted(free_blocks.begin(), free_blocks.end(), by_address),
            "Size segregated list must be sorted by block address");
        const auto iter = std::lower_bound(free_blocks.begin(), free_blocks.end(), block, by_address);
        if (iter == free_blocks.end() || *iter != block) {
            return false;
        }
        free_blocks.erase(iter);
        return true;
    }

    /// The same, found by a linear scan.
    __attribute__((always_inline)) bool erase_by_scan(size_t block, DeviceAddr size) {
        auto& free_blocks = classes_[class_of(size)];
        const auto iter = std::find(free_blocks.begin(), free_blocks.end(), block);
        if (iter == free_blocks.end()) {
            return false;
        }
        free_blocks.erase(iter);
        return true;
    }

    /// Removes and returns a block of at least `size` bytes, or nullopt when none fits. Each class is
    /// scanned in address order, or in reverse with bottom_up = false.
    /// - BEST takes the first fitting block of the lowest class that has one. Blocks within a class
    ///   are not sorted by size, so it may not find the smallest fitting block.
    /// - FIRST takes the fitting block with the lowest address, or the highest with bottom_up = false.
    template <typename AddressOf, typename SizeOf>
    __attribute__((always_inline)) std::optional<size_t> take_fit(
        DeviceAddr size, bool bottom_up, SearchPolicy policy, AddressOf address_of, SizeOf size_of) {
        std::optional<size_t> target_block;
        std::vector<size_t>* target_class = nullptr;
        size_t target_position = 0;
        DeviceAddr best_address = bottom_up ? std::numeric_limits<DeviceAddr>::max() : 0;

        for (size_t size_class = class_of(size); size_class < num_classes_; ++size_class) {
            auto& free_blocks = classes_[size_class];
            for (size_t step = 0; step < free_blocks.size(); ++step) {
                // Bottom-up walks front to back, top-down back to front. Deriving the position from an
                // unsigned step avoids an empty list underflowing the start index.
                const size_t position = bottom_up ? step : free_blocks.size() - 1 - step;
                const size_t block = free_blocks[position];
                if (policy == SearchPolicy::BEST) {
                    // Every block of a higher class is larger than every block of this one, so the
                    // first fitting block ends the search.
                    if (size_of(block) >= size) {
                        free_blocks.erase(free_blocks.begin() + static_cast<std::ptrdiff_t>(position));
                        return block;
                    }
                } else {
                    if (size_of(block) < size) {
                        continue;
                    }

                    bool address_better =
                        bottom_up ? address_of(block) < best_address : address_of(block) > best_address;
                    if (!target_block.has_value() || address_better) {
                        target_block = block;
                        target_class = &free_blocks;
                        target_position = position;
                        best_address = address_of(block);
                        break;
                    }
                }
            }
        }

        if (!target_block.has_value()) {
            return std::nullopt;
        }
        TT_ASSERT(target_position < target_class->size(), "Size segregated item index out of bounds");
        target_class->erase(target_class->begin() + static_cast<std::ptrdiff_t>(target_position));
        return target_block;
    }

private:
    /// classes_.size(), kept separately so class_of() reads one value.
    const size_t num_classes_;
    std::vector<std::vector<size_t>> classes_;
};

}  // namespace tt::tt_metal::allocator
