// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tt {

// Stable hash for cache paths and persistence.
class StableHasher {
public:
    StableHasher();

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
    // BLAKE3 state is embedded inline to avoid per-hasher dynamic allocation
    // while keeping third-party type details out of this public header.
    static constexpr std::size_t kStateStorageBytes = 2048;
    alignas(std::max_align_t) std::array<std::byte, kStateStorageBytes> state_storage_{};
};

}  // namespace tt
