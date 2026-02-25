// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace tt {

// Stable hash for cache paths and persistence. std::hash is not guaranteed to be
// stable across runs or implementations.
class FNV1a {
public:
    // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
    static constexpr uint64_t FNV_PRIME = 0x100000001b3;
    static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325;

    FNV1a(uint64_t offset = FNV_OFFSET) : hash_(offset) {}

    void update(uint64_t data) {
        hash_ ^= data;
        hash_ *= FNV_PRIME;
    }

    template <typename ForwardIterator>
    void update(ForwardIterator begin, ForwardIterator end) {
        for (auto it = begin; it != end; ++it) {
            update(static_cast<uint64_t>(*it));
        }
    }

    void update(const std::string& s) { update(s.begin(), s.end()); }

    uint64_t digest() const { return hash_; }

private:
    uint64_t hash_;
};

}  // namespace tt
