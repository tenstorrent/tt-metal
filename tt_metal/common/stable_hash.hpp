// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace tt {

// Stable hash for cache paths and persistence. std::hash is not guaranteed to be
// stable across runs or implementations; FNV1a is deterministic.
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
class FNV1a {
public:
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
            update(static_cast<uint64_t>(static_cast<unsigned char>(*it)));
        }
    }

    void update(const std::string& s) { update(s.begin(), s.end()); }

    uint64_t digest() const { return hash_; }

private:
    uint64_t hash_;
};

inline uint64_t stable_hash_string(const std::string& s) {
    FNV1a hasher;
    hasher.update(s);
    return hasher.digest();
}

}  // namespace tt
