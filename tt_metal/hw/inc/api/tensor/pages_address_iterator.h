// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include "api/tensor/page.h"
#include "internal/tensor/helpers.h"

namespace tensor_accessor {

/**
 * Iterator over all pages in a sharded tensor.
 * The iterator is initialized with a start_page_id and can be incremented by one page at a time,
 * or by a given number of pages. It tracks how many pages ahead stay contiguous in memory, so a
 * page lookup is only needed when a run ends.
 */
template <typename Accessor>
class PagesAddressIteratorSharded {
public:
    using value_type = Page;
    using difference_type = std::ptrdiff_t;
    using reference = const Page&;
    using pointer = const Page*;

    // Constructor that initializes the iterator at a starting position
    PagesAddressIteratorSharded(
        const Accessor& accessor, uint32_t start_page_id = 0, uint32_t stride = 1, uint8_t noc = noc_index) :
        accessor(accessor), current_page_id(start_page_id), stride_(stride), noc(noc) {
        if (current_page_id < accessor.dspec().tensor_volume()) {
            // Fast path needs one step to be a whole number of run pages; 0 means it never is.
            const uint32_t page_stride = accessor.contiguous_page_stride();
            run_pages_per_step_ = (stride_ % page_stride == 0) ? stride_ / page_stride : 0;
            set_noc_addr(start_page_id);
            set_run_pages_left(start_page_id);
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

        current_page_id += stride_;
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            current_page_id = accessor.dspec().tensor_volume();
            return *this;
        }

        // Inside a run the address just walks forward, no lookup needed.
        if (run_pages_per_step_ != 0 && run_pages_per_step_ < run_pages_left_) {
            run_pages_left_ -= run_pages_per_step_;
            current_noc_addr += accessor.get_aligned_page_size() * run_pages_per_step_;
        } else {
            set_noc_addr(current_page_id);
            set_run_pages_left(current_page_id);
        }
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

        current_page_id += steps * stride_;
        if (current_page_id >= accessor.dspec().tensor_volume()) {
            current_page_id = accessor.dspec().tensor_volume();
            return *this;
        }

        set_noc_addr(current_page_id);
        set_run_pages_left(current_page_id);
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
    uint32_t stride_ = 1;           // step size per operator++ (1 for contiguous, N for DM thread stride)
    uint8_t noc = noc_index;
    // 0 when a step never lands inside a run.
    uint32_t run_pages_per_step_ = 0;
    // Contiguous pages from current_page_id, inclusive. Only valid when run_pages_per_step_ != 0.
    uint32_t run_pages_left_ = 0;
    mutable Page current_page{0, 0};

    void update_current_page() { current_page = Page(current_noc_addr, current_page_id); }

    // Keep num_contiguous_pages() out of this function: inlining it here costs ~2x per page.
    void set_noc_addr(uint32_t page_id) { current_noc_addr = accessor.get_noc_addr(page_id, 0, noc); }

    void set_run_pages_left(uint32_t page_id) {
        if (run_pages_per_step_ != 0) {  // at 0 the count is never read
            run_pages_left_ = accessor.num_contiguous_pages(page_id);
        }
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
    using value_type = Page;
    using difference_type = std::ptrdiff_t;
    using reference = const Page&;
    using pointer = const Page*;

    PagesAddressIteratorInterleaved(
        const Accessor& accessor,
        uint32_t start_page_id,
        uint32_t end_page_id,
        uint32_t stride,
        uint8_t noc) :
        accessor(accessor), current_page_id(start_page_id), end_page_id_(end_page_id), stride_(stride), noc(noc) {
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
        current_page_id += stride_;
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
        current_page_id += steps * stride_;
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
    const uint32_t stride_ = 1;
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

    Pages(
        const Accessor& accessor,
        uint32_t start_page_id,
        uint32_t end_page_id,
        uint32_t stride = 1,
        uint8_t noc = noc_index) :
        accessor_(accessor),
        start_page_id_(start_page_id),
        end_page_id_(end_page_id),
        stride_(stride),
        noc_(noc) {
        ASSERT(stride > 0);                    // stride=0 would make the iterator never advance
        ASSERT(start_page_id <= end_page_id);  // inverted range silently produces no iterations
    }

    iterator begin() const {
        if constexpr (Accessor::DSpec::is_interleaved) {
            return PagesAddressIteratorInterleaved<Accessor>(accessor_, start_page_id_, end_page_id_, stride_, noc_);
        } else {
            return PagesAddressIteratorSharded<Accessor>(accessor_, start_page_id_, stride_, noc_);
        }
    }

    iterator end() const {
        if constexpr (Accessor::DSpec::is_interleaved) {
            return PagesAddressIteratorInterleaved<Accessor>(accessor_, end_page_id_, end_page_id_, stride_, noc_);
        } else {
            return PagesAddressIteratorSharded<Accessor>(accessor_, end_page_id_, stride_, noc_);
        }
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

private:
    const Accessor& accessor_;
    uint32_t start_page_id_;
    uint32_t end_page_id_;
    uint32_t stride_;
    uint8_t noc_;
};

}  // namespace tensor_accessor
