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
 * Iterator over all pages in a tensor.
 * The iterator is initialized with a start_page_id and can be incremented by one page at a time,
 * or by a given number of pages. It efficiently computes NOC addresses by minimizing bank lookups
 * and reusing address computations when possible.
 */
template <typename Accessor>
class PagesAddressIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Page;
    using difference_type = std::ptrdiff_t;
    using reference = const Page&;
    using pointer = const Page*;

    // Constructor that initializes the iterator at a starting position
    PagesAddressIterator(const Accessor& accessor, uint32_t start_page_id = 0, uint8_t noc = noc_index) :
        accessor(accessor), current_page_id(start_page_id), noc(noc) {
        if (current_page_id < accessor.dspec().tensor_volume()) {
            current_page_mapping = accessor.get_bank_and_offset(current_page_id);
            current_noc_addr = accessor.get_noc_addr(current_page_mapping, 0, noc);
            current_shard_id = compute_shard_id(current_page_id);
            update_current_page();
        }
    }

    // Getters
    uint32_t page_id() const { return current_page_id; }

    reference operator*() const { return current_page; }
    pointer operator->() const { return &current_page; }

    // Arithmetic operators
    PagesAddressIterator& operator++() {
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            return *this;  // End iterator
        }

        current_page_id++;
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            current_page_id = accessor.dspec().tensor_volume();
            return *this;
        }

        // Efficient address computation: check if we're moving to the next page in the same bank
        if (can_use_incremental_update()) {
            current_noc_addr += accessor.page_size;
        } else {
            // Need to recompute bank mapping and NOC address
            current_page_mapping = accessor.get_bank_and_offset(current_page_id);
            current_noc_addr = accessor.get_noc_addr(current_page_mapping, 0, noc);
            current_shard_id = compute_shard_id(current_page_id);
        }

        update_current_page();
        return *this;
    }

    PagesAddressIterator operator++(int) {
        PagesAddressIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    PagesAddressIterator& operator+=(difference_type steps) {
        ASSERT(steps >= 0);
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            return *this;  // End iterator
        }

        current_page_id += steps;
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            current_page_id = accessor.dspec().tensor_volume();
            return *this;
        }

        // For large steps, it's more efficient to recompute from scratch
        current_page_mapping = accessor.get_bank_and_offset(current_page_id);
        current_noc_addr = accessor.get_noc_addr(current_page_mapping, 0, noc);
        current_shard_id = compute_shard_id(current_page_id);
        update_current_page();
        return *this;
    }

    PagesAddressIterator operator+(difference_type steps) const {
        PagesAddressIterator tmp = *this;
        tmp += steps;
        return tmp;
    }

    const Page& operator[](difference_type n) const {
        auto temp = *this;
        temp += n;
        return *temp;
    }

    // Comparison operators
    bool operator==(const PagesAddressIterator& other) const { return current_page_id == other.current_page_id; }

    bool operator!=(const PagesAddressIterator& other) const { return !(*this == other); }

    bool operator<(const PagesAddressIterator& other) const { return current_page_id < other.current_page_id; }

    bool operator>(const PagesAddressIterator& other) const { return other < *this; }

    bool operator<=(const PagesAddressIterator& other) const { return *this < other || *this == other; }

    bool operator>=(const PagesAddressIterator& other) const { return !(*this < other); }

private:
    const Accessor& accessor;
    uint32_t current_page_id = 0;
    uint64_t current_noc_addr = 0;
    uint8_t noc = noc_index;
    uint32_t current_shard_id = 0;

    typename Accessor::PageMapping current_page_mapping{0, 0};
    mutable Page current_page{0, 0, 0};

    void update_current_page() {
        if (current_page_id < accessor.dspec().tensor_volume()) {
            current_page = Page(current_noc_addr, current_page_id, current_shard_id);
        }
    }

    // Check if we can use incremental address update (same bank, consecutive pages in bank)
    bool can_use_incremental_update() const {
        auto next_page_mapping = accessor.get_bank_and_offset(current_page_id);

        // Same bank and consecutive pages within that bank
        return (next_page_mapping.bank_id == current_page_mapping.bank_id) &&
               (next_page_mapping.bank_page_offset == current_page_mapping.bank_page_offset + 1);
    }

    // Compute which shard a page belongs to
    uint32_t compute_shard_id(uint32_t page_id) const {
        auto page_mapping = accessor.get_bank_and_offset(page_id);

        // Shard ID is determined by the bank and the position within the bank
        uint32_t bank_shard_id = page_mapping.bank_page_offset / accessor.dspec().shard_volume();
        return page_mapping.bank_id + bank_shard_id * accessor.dspec().num_banks();
    }
};

/**
 * Proxy for PagesAddressIterator, to enable range-based for loop over all pages in a tensor.
 */
template <typename Accessor>
class Pages {
public:
    using iterator = PagesAddressIterator<Accessor>;
    using const_iterator = PagesAddressIterator<Accessor>;

    Pages(const Accessor& accessor, uint32_t start_page_id = 0, uint8_t noc = noc_index) :
        accessor_(accessor), start_page_id_(start_page_id), noc_(noc) {}

    iterator begin() const { return PagesAddressIterator<Accessor>(accessor_, start_page_id_, noc_); }

    iterator end() const { return PagesAddressIterator<Accessor>(accessor_, accessor_.dspec().tensor_volume(), noc_); }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

private:
    const Accessor& accessor_;
    uint32_t start_page_id_;
    uint8_t noc_;
};

}  // namespace tensor_accessor
