// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace tt {

// Stable hash for cache paths and persistence.
class StableHasher {
public:
    StableHasher();
    ~StableHasher();

    StableHasher(StableHasher&&) noexcept;
    StableHasher& operator=(StableHasher&&) noexcept;

    StableHasher(const StableHasher&) = delete;
    StableHasher& operator=(const StableHasher&) = delete;

    void update(uint64_t data);
    void update(const void* data, std::size_t size);
    void update(std::string_view data);

    template <typename ForwardIterator>
    void update(ForwardIterator begin, ForwardIterator end) {
        for (auto it = begin; it != end; ++it) {
            update(static_cast<uint64_t>(*it));
        }
    }

    uint64_t digest() const;

private:
    // Pimpl: keeps the BLAKE3 type out of this public header, and lets us
    // avoid reasoning about placement-new / implicit object lifetime for a
    // C-struct embedded in a byte buffer. The cost is one small heap
    // allocation per StableHasher, which is immaterial for this class's
    // cache-key-derivation use cases.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt
