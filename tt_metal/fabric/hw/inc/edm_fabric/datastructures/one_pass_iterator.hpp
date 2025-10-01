// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"


namespace tt::tt_fabric {

template <typename T, typename INHERITED_TYPE>
struct OnePassIteratorBase {
    T* current_ptr;
    T* end_ptr;

    OnePassIteratorBase() : current_ptr(nullptr), end_ptr(nullptr) {}

    FORCE_INLINE T* get_current_ptr() const { return current_ptr; }
    FORCE_INLINE void increment() { static_cast<INHERITED_TYPE*>(this)->increment_impl(); }

    FORCE_INLINE bool is_done() const { return current_ptr == end_ptr; }

    FORCE_INLINE void reset_to(T* base_ptr) { static_cast<INHERITED_TYPE*>(this)->reset_to_impl(base_ptr); }
};


template <typename T, size_t NUM_ENTRIES, size_t ENTRY_SIZE_BYTES>
struct OnePassIteratorStaticSizes
    : public OnePassIteratorBase<T, OnePassIteratorStaticSizes<T, NUM_ENTRIES, ENTRY_SIZE_BYTES>> {
    OnePassIteratorStaticSizes() :
        OnePassIteratorBase<T, OnePassIteratorStaticSizes<T, NUM_ENTRIES, ENTRY_SIZE_BYTES>>() {}

    FORCE_INLINE void increment_impl() { this->current_ptr += ENTRY_SIZE_BYTES; }

    FORCE_INLINE void reset_to_impl(T* base_ptr) {
        this->current_ptr = base_ptr;
        this->end_ptr = base_ptr + (NUM_ENTRIES * ENTRY_SIZE_BYTES);
    }
};

}  // namespace tt::tt_fabric