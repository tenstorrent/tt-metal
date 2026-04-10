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
class StableHasher {
public:
    StableHasher() { XXH3_64bits_reset(&state_); }

    void update(uint64_t data) { XXH3_64bits_update(&state_, &data, sizeof(data)); }

    void update(const void* data, std::size_t size) { XXH3_64bits_update(&state_, data, size); }

    void update(const char* begin, const char* end) {
        if (begin < end) {
            XXH3_64bits_update(&state_, begin, static_cast<std::size_t>(end - begin));
        }
    }

    template <typename ForwardIterator>
    void update(ForwardIterator begin, ForwardIterator end) {
        for (auto it = begin; it != end; ++it) {
            update(static_cast<uint64_t>(*it));
        }
    }

    void update(const std::string& s) { XXH3_64bits_update(&state_, s.data(), s.size()); }

    uint64_t digest() const { return XXH3_64bits_digest(&state_); }

private:
    XXH3_state_t state_;
};

}  // namespace tt
