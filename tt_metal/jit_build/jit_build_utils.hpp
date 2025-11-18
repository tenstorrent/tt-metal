// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);
void create_file(const std::string& file_path_str);

// Need a stable hash for things to persist across runs.
// std::hash is not guaranteed to be stable across runs / implementations.
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

    uint64_t digest() const { return hash_; }

private:
    uint64_t hash_;
};

}  // namespace tt::jit_build::utils
