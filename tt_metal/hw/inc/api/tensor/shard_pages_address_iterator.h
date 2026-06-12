// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include "api/tensor/page.h"

namespace tensor_accessor {

/**
 * Iterator over pages in a shard.
 * The iterator is initialized with a shard_id and a start_page_offset.
 * It can be incremented by one page at a time, or by a given number of pages.
 * It can be compared to another iterator to check if they are equal.
 * It can be dereferenced to get the current Page
 * It can be indexed to get the n-th page.
 */
template <typename Accessor>
class ShardPagesAddressIterator {
public:
    using ArrayU32 = std::array<uint32_t, Accessor::DSpec::rank_ct>;
    using PageMapping = typename Accessor::PageMapping;

    using value_type = Page;
    using difference_type = std::ptrdiff_t;
    using reference = const Page&;
    using pointer = const Page*;

    // Constructor that initializes the iterator at a starting position
    ShardPagesAddressIterator(
        const Accessor& accessor,
        uint32_t shard_id = 0,
        uint32_t start_page_offset = 0,
        uint32_t end_page_offset = 0,
        uint8_t noc = noc_index) :
        accessor(accessor),
        current_page_id_in_shard(start_page_offset),
        end_page_id_in_shard(end_page_offset),
        current_shard_id(shard_id),
        noc(noc) {
        auto inside_tensor = calculate_current_location(shard_id, start_page_offset);

        // If starting page is outside logical tensor bounds, advance to next valid page
        if (current_page_id_in_shard < end_page_id_in_shard && !inside_tensor) {
            // Starting page is invalid, advance to next valid page or end
            do {
                current_page_id_in_shard++;
                if (current_page_id_in_shard >= end_page_id_in_shard) {
                    current_page_id_in_shard = end_page_id_in_shard;
                    break;
                }
            } while (!update_local_global_page_coord());
        }

        // Calculate NOC address for the final position
        const auto bank_shard = accessor.shard_to_bank(shard_id);
        PageMapping current_page_mapping{
            .bank_id = bank_shard.bank_id,
            .bank_page_offset =
                (bank_shard.shard_in_bank * accessor.dspec().shard_volume()) + current_page_id_in_shard};
        current_noc_addr = accessor.get_noc_addr(current_page_mapping, 0, noc);
        ASSERT(current_page_id_in_shard <= accessor.dspec().shard_volume());
        update_current_page();
    }

    // Getters
    uint32_t page_id() const {
        uint32_t page_id = 0;
        for (uint32_t i = 0; i < accessor.dspec().rank(); ++i) {
            page_id += global_page_coord[i] * accessor.dspec().tensor_strides()[i];
        }
        return page_id;
    }

    reference operator*() const { return current_page; }
    pointer operator->() const { return &current_page; }

    // Arithmetic operators
    ShardPagesAddressIterator& operator++() {
        if (current_page_id_in_shard >= end_page_id_in_shard) {
            return *this;  // End iterator
        }

        do {
            current_noc_addr += accessor.get_aligned_page_size();
            current_page_id_in_shard++;
            if (current_page_id_in_shard >= end_page_id_in_shard) {
                current_page_id_in_shard = end_page_id_in_shard;
                break;
            }
        } while (!update_local_global_page_coord());
        update_current_page();
        return *this;
    }

    ShardPagesAddressIterator operator++(int) {
        ShardPagesAddressIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    ShardPagesAddressIterator& operator+=(difference_type steps) {
        ASSERT(steps >= 0);
        if (current_page_id_in_shard >= end_page_id_in_shard) {
            return *this;  // End iterator
        }

        do {
            current_noc_addr += steps * accessor.get_aligned_page_size();
            current_page_id_in_shard += steps;
            if (current_page_id_in_shard >= end_page_id_in_shard) {
                current_page_id_in_shard = end_page_id_in_shard;
                break;
            }
        } while (!update_local_global_page_coord(steps));
        update_current_page();
        return *this;
    }

    ShardPagesAddressIterator operator+(difference_type steps) const {
        ShardPagesAddressIterator tmp = *this;
        tmp += steps;
        return tmp;
    }

    const Page& operator[](difference_type n) const {
        auto temp = *this;
        temp += n;
        return *temp;
    }

    // Comparison operators
    bool operator==(const ShardPagesAddressIterator& other) const {
        return (current_shard_id == other.current_shard_id) &&
               (current_page_id_in_shard == other.current_page_id_in_shard);
    }

    bool operator!=(const ShardPagesAddressIterator& other) const { return !(*this == other); }

    bool operator<(const ShardPagesAddressIterator& other) const {
        return (current_shard_id == other.current_shard_id) &&
               (current_page_id_in_shard < other.current_page_id_in_shard);
    }

    bool operator>(const ShardPagesAddressIterator& other) const { return other < *this; }

    bool operator<=(const ShardPagesAddressIterator& other) const { return *this < other || *this == other; }

    bool operator>=(const ShardPagesAddressIterator& other) const { return !(*this < other); }

private:
    const Accessor& accessor;
    uint32_t current_page_id_in_shard = 0;
    uint32_t end_page_id_in_shard = 0;
    uint32_t current_shard_id = 0;
    uint64_t current_noc_addr = 0;
    uint8_t noc = noc_index;

    ArrayU32 local_page_coord = {};
    ArrayU32 global_page_coord = {};
    ArrayU32 shard_coord = {};

    mutable Page current_page{0, 0};

    void update_current_page() {
        if (current_page_id_in_shard < end_page_id_in_shard) {
            current_page = Page(current_noc_addr, page_id());
        }
    }

    // Calculates global page coordinate and checks if it's within logical tensor bounds
    bool update_global_page_coord() {
        const auto& dspec = accessor.dspec();
        for (uint32_t i = 0; i < dspec.rank(); ++i) {
            global_page_coord[i] = shard_coord[i] * dspec.shard_shape()[i] + local_page_coord[i];
            // Check bounds - some shards at edges might have fewer pages (in case of padding)
            if (global_page_coord[i] >= dspec.tensor_shape()[i]) {
                return false;  // Page is outside logical tensor bounds
            }
        }
        return true;
    }

    // Calculates shard coordinate, page coordinate within shard, and global page coordinate
    bool calculate_current_location(uint32_t shard_id, uint32_t page_id_in_shard) {
        const auto& dspec = accessor.dspec();

        // Calculate shard coordinates once and store them
        uint32_t remaining_shard_id = shard_id;
        for (int i = dspec.rank() - 1; i >= 0; --i) {
            shard_coord[i] = remaining_shard_id % dspec.shard_grid()[i];
            remaining_shard_id /= dspec.shard_grid()[i];
        }

        // Calculate initial page coordinates within shard
        uint32_t temp_page_id = page_id_in_shard;
        for (int i = dspec.rank() - 1; i >= 0; --i) {
            local_page_coord[i] = temp_page_id % dspec.shard_shape()[i];
            temp_page_id /= dspec.shard_shape()[i];
        }

        // Calculate initial global coordinates
        return update_global_page_coord();
    }

    // Updates local_page_coord and global_page_coord by incrementing the rightmost dimension
    // Returns false if page is outside logical tensor bounds (padded tile)
    bool update_local_global_page_coord() {
        const auto& dspec = accessor.dspec();

        // Incrementally update local_page_coord like a multi-dimensional counter
        // Start from the rightmost (last) dimension and carry over when needed
        for (int i = dspec.rank() - 1; i >= 0; --i) {
            local_page_coord[i]++;
            // No overflow
            if (local_page_coord[i] < dspec.shard_shape()[i]) {
                break;
            }
            // Overflow - reset this dimension and continue to next dimension
            local_page_coord[i] = 0;
        }

        return update_global_page_coord();
    }

    // Generic version of update_local_global_page_coord for arbitrary steps
    bool update_local_global_page_coord(uint32_t step) {
        const auto& dspec = accessor.dspec();

        uint32_t carry = step;
        for (int i = dspec.rank() - 1; i >= 0 && carry > 0; --i) {
            uint32_t new_coord = local_page_coord[i] + carry;
            local_page_coord[i] = new_coord % dspec.shard_shape()[i];
            carry = new_coord / dspec.shard_shape()[i];
        }

        return update_global_page_coord();
    }
};

/**
 * Proxy for ShardPagesAddressIterator, to enable range-based for loop over pages in a shard.
 */
template <typename Accessor>
class ShardPages {
public:
    using iterator = ShardPagesAddressIterator<Accessor>;
    using const_iterator = ShardPagesAddressIterator<Accessor>;

    ShardPages(
        const Accessor& accessor,
        uint32_t shard_id,
        uint32_t start_page_offset,
        uint32_t end_page_offset,
        uint8_t noc = noc_index) :
        accessor_(accessor),
        shard_id_(shard_id),
        start_page_offset_(start_page_offset),
        end_page_offset_(end_page_offset),
        noc_(noc) {}

    iterator begin() const {
        return ShardPagesAddressIterator<Accessor>(accessor_, shard_id_, start_page_offset_, end_page_offset_, noc_);
    }

    iterator end() const {
        return ShardPagesAddressIterator<Accessor>(accessor_, shard_id_, end_page_offset_, end_page_offset_, noc_);
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

private:
    const Accessor& accessor_;
    uint32_t shard_id_;
    uint32_t start_page_offset_;
    uint32_t end_page_offset_;
    uint8_t noc_;
};
/**
 * Iterator over a strided subset of shards, yielding one ShardPages range per assigned shard.
 *
 * Each DM owns shards at indices: start_shard_id, start_shard_id + stride, ...
 * This implements the whole-shard granularity for multi-threaded shard iteration.
 */
template <typename Accessor>
class StridedShardPagesIterator {
public:
    using value_type = ShardPages<Accessor>;

    StridedShardPagesIterator(
        const Accessor& accessor,
        uint32_t current_shard_id,
        uint32_t end_shard_id,
        uint32_t stride,
        uint8_t noc) :
        accessor_(accessor),
        current_shard_id_(current_shard_id),
        end_shard_id_(end_shard_id),
        stride_(stride),
        noc_(noc) {}

    value_type operator*() const {
        return ShardPages<Accessor>(accessor_, current_shard_id_, 0, accessor_.dspec().shard_volume(), noc_);
    }

    StridedShardPagesIterator& operator++() {
        current_shard_id_ += stride_;
        return *this;
    }

    StridedShardPagesIterator operator++(int) {
        StridedShardPagesIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const StridedShardPagesIterator& other) const {
        // Normalize past-end: treat any id >= end_shard_id as "at end".
        // Stride overshoot is expected (stride > 1 means the last ++ may skip past the sentinel),
        // but the overshoot must be < stride.
        bool lhs_done = current_shard_id_ >= end_shard_id_;
        bool rhs_done = other.current_shard_id_ >= other.end_shard_id_;
        ASSERT(!lhs_done || (current_shard_id_ - end_shard_id_ < stride_));
        ASSERT(!rhs_done || (other.current_shard_id_ - other.end_shard_id_ < other.stride_));
        if (lhs_done && rhs_done) {
            return true;
        }
        if (lhs_done || rhs_done) {
            return false;
        }
        return current_shard_id_ == other.current_shard_id_;
    }

    bool operator!=(const StridedShardPagesIterator& other) const { return !(*this == other); }

private:
    const Accessor& accessor_;
    uint32_t current_shard_id_;
    uint32_t end_shard_id_;
    uint32_t stride_;
    uint8_t noc_;
};

/**
 * Proxy that enables range-based for over the shards owned by this DM.
 *
 * Each iteration yields a ShardPages range covering all pages within one assigned shard.
 * Thread i owns shards at indices: i, i + num_threads, i + 2*num_threads, ...
 */
template <typename Accessor>
class StridedShardPages {
public:
    using iterator = StridedShardPagesIterator<Accessor>;
    using const_iterator = iterator;

    StridedShardPages(
        const Accessor& accessor,
        uint32_t start_shard_id,
        uint32_t total_shards,
        uint32_t stride,
        uint8_t noc = noc_index) :
        accessor_(accessor),
        start_shard_id_(start_shard_id),
        total_shards_(total_shards),
        stride_(stride),
        noc_(noc) {}

    iterator begin() const {
        return StridedShardPagesIterator<Accessor>(accessor_, start_shard_id_, total_shards_, stride_, noc_);
    }

    iterator end() const {
        return StridedShardPagesIterator<Accessor>(accessor_, total_shards_, total_shards_, stride_, noc_);
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

private:
    const Accessor& accessor_;
    uint32_t start_shard_id_;
    uint32_t total_shards_;
    uint32_t stride_;
    uint8_t noc_;
};

}  // namespace tensor_accessor
