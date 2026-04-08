// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace tt {

// Stable hash for cache paths and persistence.
//
// Uses xxHash (XXH3_64bits) for speed and collision resistance.
// XXH3 is significantly faster than FNV1a on large inputs thanks to
// vectorised instruction use and better mixing.  The streaming API
// (XXH3_state_t) allows incremental feeding, matching the old
// FNV1a::update() pattern.
//
// NOTE: This is a drop-in replacement for the previous FNV1a class.
//       Hash values are NOT compatible with old caches — a one-time
//       cache rebuild will happen automatically.
class StableHasher {
public:
    StableHasher() { XXH3_64bits_reset(&state_); }

    void update(uint64_t data) {
        XXH3_64bits_update(&state_, &data, sizeof(data));
    }

    // Bulk update from a raw pointer range — the fast path.
    // Passes the entire region to xxHash in a single call.
    void update(const void* data, std::size_t size) {
        XXH3_64bits_update(&state_, data, size);
    }

    // Contiguous char* range — delegates to the bulk overload.
    // Preferred over the generic iterator template for buffered reads.
    void update(const char* begin, const char* end) {
        if (begin < end) {
            XXH3_64bits_update(&state_, begin, static_cast<std::size_t>(end - begin));
        }
    }

    // Generic iterator range fallback — element by element.
    // Prefer the pointer overloads above for contiguous data.
    template <typename ForwardIterator>
    void update(ForwardIterator begin, ForwardIterator end) {
        for (auto it = begin; it != end; ++it) {
            update(static_cast<uint64_t>(*it));
        }
    }

    void update(const std::string& s) {
        XXH3_64bits_update(&state_, s.data(), s.size());
    }

    uint64_t digest() const {
        return XXH3_64bits_digest(&state_);
    }

private:
    XXH3_state_t state_;
};

// Backward-compatible alias — existing call sites use tt::FNV1a.
// TODO(#41672): Migrate callers to StableHasher, then remove this alias.
using FNV1a = StableHasher;

}  // namespace tt
