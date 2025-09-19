// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include "page.h"
#include "helpers.h"

namespace tensor_accessor {

/**
 * Iterator over all pages in a sharded tensor.
 * The iterator is initialized with a start_page_id and can be incremented by one page at a time,
 * or by a given number of pages. It efficiently computes NOC addresses by maintaining page
 * coordinates and minimizing divisions through incremental updates.
 */
template <typename Accessor>
class PagesAddressIteratorSharded {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Page;
    using difference_type = std::ptrdiff_t;
    using reference = const Page&;
    using pointer = const Page*;

    // Constructor that initializes the iterator at a starting position
    PagesAddressIteratorSharded(const Accessor& accessor, uint32_t start_page_id = 0, uint8_t noc = noc_index) :
        accessor(accessor), current_page_id(start_page_id), noc(noc) {
        if (current_page_id < accessor.dspec().tensor_volume()) {
            // Initialize coordinates and state from page_id
            initialize_from_page_id(start_page_id);
            update_current_page();
        }
    }

    // Getters
    uint32_t page_id() const { return current_page_id; }

    reference operator*() const { return current_page; }
    pointer operator->() const { return &current_page; }

    // Arithmetic operators
    PagesAddressIteratorSharded& operator++() {
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            return *this;  // End iterator
        }

        current_page_id++;
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            current_page_id = accessor.dspec().tensor_volume();
            return *this;
        }

        // Efficiently update coordinates and address without divisions
        increment_page_coordinate();
        update_current_page();
        return *this;
    }

    PagesAddressIteratorSharded operator++(int) {
        PagesAddressIteratorSharded tmp = *this;
        ++(*this);
        return tmp;
    }

    PagesAddressIteratorSharded& operator+=(difference_type steps) {
        ASSERT(steps >= 0);
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            return *this;  // End iterator
        }

        current_page_id += steps;
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            current_page_id = accessor.dspec().tensor_volume();
            return *this;
        }

        // For large steps or when crossing many boundaries, reinitialize from page_id
        initialize_from_page_id(current_page_id);
        update_current_page();
        return *this;
    }

    PagesAddressIteratorSharded operator+(difference_type steps) const {
        PagesAddressIteratorSharded tmp = *this;
        tmp += steps;
        return tmp;
    }

    const Page& operator[](difference_type n) const {
        auto temp = *this;
        temp += n;
        return *temp;
    }

    // Comparison operators
    bool operator==(const PagesAddressIteratorSharded& other) const { return current_page_id == other.current_page_id; }

    bool operator!=(const PagesAddressIteratorSharded& other) const { return !(*this == other); }

    bool operator<(const PagesAddressIteratorSharded& other) const { return current_page_id < other.current_page_id; }

    bool operator>(const PagesAddressIteratorSharded& other) const { return other < *this; }

    bool operator<=(const PagesAddressIteratorSharded& other) const { return *this < other || *this == other; }

    bool operator>=(const PagesAddressIteratorSharded& other) const { return !(*this < other); }

private:
    const Accessor& accessor;
    uint32_t current_page_id = 0;   // current page id
    uint64_t current_noc_addr = 0;  // current NOC address for this page
    uint8_t noc = noc_index;
    uint32_t current_shard_id = 0;  // Which shard this page belongs to

    // State for efficient incremental updates
    typename Accessor::PageMapping current_page_mapping{0, 0};  // {bank_id, bank_page_offset}
    mutable Page current_page{0, 0};

    // Coordinates and derived state for avoiding divisions
    [[no_unique_address]] mutable tensor_accessor::detail::
        ConditionalField<!Accessor::DSpec::has_static_rank, uint32_t[tensor_accessor::MAX_RANK]> _page_coord_buf;
    typename Accessor::DSpec::Shape page_coord;  // Current page coordinates [dim0, dim1, ...]
    uint32_t page_offset_within_shard = 0;       // Offset of this page within its shard
    uint32_t flattened_shard_id = 0;             // Linear shard id in the shard grid
    uint32_t bank_shard_id = 0;                  // Which shard within the bank this page belongs to

    void update_current_page() { current_page = Page(current_noc_addr, current_page_id); }

    // Initialize all state from a page_id (used in constructor and operator+=)
    void initialize_from_page_id(uint32_t page_id) {
        // Initialize page coordinate buffer if needed
        if constexpr (!Accessor::DSpec::has_static_rank) {
            page_coord = typename Accessor::DSpec::Shape(_page_coord_buf.value, accessor.dspec().rank());
        }

        // Convert page_id to coordinates (this is the only place we do divisions)
        uint32_t temp_page_id = page_id;
        for (int i = accessor.dspec().rank() - 1; i >= 0; --i) {
            page_coord[i] = temp_page_id % accessor.dspec().tensor_shape()[i];
            temp_page_id /= accessor.dspec().tensor_shape()[i];
        }

        // Calculate shard information
        flattened_shard_id = 0;
        page_offset_within_shard = 0;
        for (size_t i = 0; i < accessor.dspec().rank(); ++i) {
            flattened_shard_id +=
                (page_coord[i] / accessor.dspec().shard_shape()[i]) * accessor.dspec().shard_grid_strides()[i];
            page_offset_within_shard +=
                (page_coord[i] % accessor.dspec().shard_shape()[i]) * accessor.dspec().shard_strides()[i];
        }

        // Calculate bank mapping
        current_page_mapping.bank_id = flattened_shard_id % accessor.dspec().num_banks();
        bank_shard_id = flattened_shard_id / accessor.dspec().num_banks();
        current_page_mapping.bank_page_offset =
            bank_shard_id * accessor.dspec().shard_volume() + page_offset_within_shard;

        // Calculate NOC address and shard ID
        current_noc_addr = accessor.get_noc_addr(current_page_mapping, 0, noc);
        current_shard_id = current_page_mapping.bank_id + bank_shard_id * accessor.dspec().num_banks();
    }

    // Efficiently increment page coordinate and update derived state
    void increment_page_coordinate() {
        const auto& dspec = accessor.dspec();

        // Simple approach: just increment and recalculate when we can't use simple increment
        // This is more reliable than complex incremental logic

        // Try simple increment first (consecutive pages in same bank)
        if (can_use_simple_increment()) {
            apply_simple_increment();
            return;
        }

        // Need to update coordinates and recalculate (shard boundary crossed)
        increment_coordinates();
        recalculate_mapping_from_coordinates();
    }

    // Check if we can just increment the address (consecutive pages in same bank)
    bool can_use_simple_increment() const {
        const auto& dspec = accessor.dspec();

        // Check if incrementing rightmost coordinate would stay in same row
        uint32_t next_coord = page_coord[dspec.rank() - 1] + 1;
        if (next_coord >= dspec.tensor_shape()[dspec.rank() - 1]) {
            return false;  // Would overflow tensor bounds
        }

        // Check if still in same shard
        uint32_t current_shard_coord = page_coord[dspec.rank() - 1] / dspec.shard_shape()[dspec.rank() - 1];
        uint32_t next_shard_coord = next_coord / dspec.shard_shape()[dspec.rank() - 1];

        return current_shard_coord == next_shard_coord;
    }

    // Apply the simple increment (called after can_use_simple_increment returns true)
    void apply_simple_increment() {
        const auto& dspec = accessor.dspec();

        // Update coordinate tracking
        page_coord[dspec.rank() - 1]++;
        page_offset_within_shard += dspec.shard_strides()[dspec.rank() - 1];

        // Update NOC address and bank offset
        current_noc_addr += accessor.page_size;
        current_page_mapping.bank_page_offset++;
    }

    // Increment coordinates like a multi-dimensional counter
    void increment_coordinates() {
        const auto& dspec = accessor.dspec();

        for (int i = dspec.rank() - 1; i >= 0; --i) {
            page_coord[i]++;
            if (page_coord[i] < dspec.tensor_shape()[i]) {
                return;  // No carry needed
            }
            page_coord[i] = 0;  // Carry over
        }
    }

    // Recalculate all mapping info from current coordinates
    void recalculate_mapping_from_coordinates() {
        const auto& dspec = accessor.dspec();

        // Recalculate shard information from coordinates
        flattened_shard_id = 0;
        page_offset_within_shard = 0;
        for (size_t i = 0; i < dspec.rank(); ++i) {
            flattened_shard_id += (page_coord[i] / dspec.shard_shape()[i]) * dspec.shard_grid_strides()[i];
            page_offset_within_shard += (page_coord[i] % dspec.shard_shape()[i]) * dspec.shard_strides()[i];
        }

        // Recalculate bank mapping
        current_page_mapping.bank_id = flattened_shard_id % dspec.num_banks();
        bank_shard_id = flattened_shard_id / dspec.num_banks();
        current_page_mapping.bank_page_offset = bank_shard_id * dspec.shard_volume() + page_offset_within_shard;

        // Recalculate NOC address and shard ID
        current_noc_addr = accessor.get_noc_addr(current_page_mapping, 0, noc);
        current_shard_id = current_page_mapping.bank_id + bank_shard_id * dspec.num_banks();
    }
};

/**
 * Iterator over all pages in an interleaved tensor.
 * The iterator is initialized with a start_page_id and can be incremented by one page at a time,
 * or by a given number of pages. It uses a simpler implementation that just calls
 * accessor.get_noc_addr for each page without complex optimizations.
 */
template <typename Accessor>
class PagesAddressIteratorInterleaved {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Page;
    using difference_type = std::ptrdiff_t;
    using reference = const Page&;
    using pointer = const Page*;

    PagesAddressIteratorInterleaved(
        const Accessor& accessor, uint32_t start_page_id, uint32_t end_page_id, uint8_t noc) :
        accessor(accessor), current_page_id(start_page_id), end_page_id_(end_page_id), noc(noc) {
        // If start_page_id is beyond end_page_id, create an end iterator
        if (current_page_id >= end_page_id_) {
            current_page_id = end_page_id_;
            return;
        }
        update_current_page();
    }

    // Getters
    uint32_t page_id() const { return current_page_id; }

    reference operator*() const { return current_page; }
    pointer operator->() const { return &current_page; }

    // Arithmetic operators
    PagesAddressIteratorInterleaved& operator++() {
        current_page_id++;
        if (current_page_id >= end_page_id_) {
            current_page_id = end_page_id_;
            return *this;
        }

        update_current_page();
        return *this;
    }

    PagesAddressIteratorInterleaved operator++(int) {
        PagesAddressIteratorInterleaved tmp = *this;
        ++(*this);
        return tmp;
    }

    PagesAddressIteratorInterleaved& operator+=(difference_type steps) {
        ASSERT(steps >= 0);
        current_page_id += steps;
        if (current_page_id >= end_page_id_) {
            current_page_id = end_page_id_;
            return *this;
        }

        update_current_page();
        return *this;
    }

    PagesAddressIteratorInterleaved operator+(difference_type steps) const {
        PagesAddressIteratorInterleaved tmp = *this;
        tmp += steps;
        return tmp;
    }

    const Page& operator[](difference_type n) const {
        auto temp = *this;
        temp += n;
        return *temp;
    }

    // Comparison operators
    bool operator==(const PagesAddressIteratorInterleaved& other) const {
        return current_page_id == other.current_page_id;
    }

    bool operator!=(const PagesAddressIteratorInterleaved& other) const { return !(*this == other); }

    bool operator<(const PagesAddressIteratorInterleaved& other) const {
        return current_page_id < other.current_page_id;
    }

    bool operator>(const PagesAddressIteratorInterleaved& other) const { return other < *this; }

    bool operator<=(const PagesAddressIteratorInterleaved& other) const { return *this < other || *this == other; }

    bool operator>=(const PagesAddressIteratorInterleaved& other) const { return !(*this < other); }

private:
    const Accessor& accessor;
    uint32_t current_page_id = 0;
    const uint32_t end_page_id_ = 0;
    const uint8_t noc = noc_index;
    mutable Page current_page{0, 0};

    void update_current_page() {
        auto current_noc_addr = accessor.get_noc_addr(current_page_id, 0, noc);
        current_page = Page(current_noc_addr, current_page_id);
    }
};

/**
 * Proxy for PagesAddressIterator, to enable range-based for loop over all pages in a tensor.
 * Automatically selects the appropriate iterator type based on whether the accessor is interleaved.
 */
template <typename Accessor>
class Pages {
public:
    // Select iterator type based on accessor properties
    using iterator = std::conditional_t<
        Accessor::DSpec::is_interleaved,
        PagesAddressIteratorInterleaved<Accessor>,
        PagesAddressIteratorSharded<Accessor>>;
    using const_iterator = iterator;

    Pages(const Accessor& accessor, uint32_t start_page_id, uint32_t end_page_id, uint8_t noc = noc_index) :
        accessor_(accessor), start_page_id_(start_page_id), end_page_id_(end_page_id), noc_(noc) {}

    iterator begin() const {
        if constexpr (Accessor::DSpec::is_interleaved) {
            return PagesAddressIteratorInterleaved<Accessor>(accessor_, start_page_id_, end_page_id_, noc_);
        } else {
            return PagesAddressIteratorSharded<Accessor>(accessor_, start_page_id_, noc_);
        }
    }

    iterator end() const {
        if constexpr (Accessor::DSpec::is_interleaved) {
            return PagesAddressIteratorInterleaved<Accessor>(accessor_, end_page_id_, end_page_id_, noc_);
        } else {
            return PagesAddressIteratorSharded<Accessor>(accessor_, end_page_id_, noc_);
        }
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

private:
    const Accessor& accessor_;
    uint32_t start_page_id_;
    uint32_t end_page_id_;
    uint8_t noc_;
};

}  // namespace tensor_accessor
