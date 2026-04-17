// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/stable_hash.hpp"

#include "blake3.h"

#include <cstddef>
#include <cstdint>

namespace tt {

namespace {

// Interpret the first 8 bytes of a BLAKE3 XOF output as a little-endian uint64_t.
// Defined explicitly so the reduction is portable and cannot drift.
uint64_t load_le_u64(const uint8_t bytes[8]) {
    uint64_t v = 0;
    v |= static_cast<uint64_t>(bytes[0]);
    v |= static_cast<uint64_t>(bytes[1]) << 8;
    v |= static_cast<uint64_t>(bytes[2]) << 16;
    v |= static_cast<uint64_t>(bytes[3]) << 24;
    v |= static_cast<uint64_t>(bytes[4]) << 32;
    v |= static_cast<uint64_t>(bytes[5]) << 40;
    v |= static_cast<uint64_t>(bytes[6]) << 48;
    v |= static_cast<uint64_t>(bytes[7]) << 56;
    return v;
}

}  // namespace

StableHasher::StableHasher() {
    static_assert(
        sizeof(blake3_hasher) <= kStateStorageBytes, "StableHasher inline storage is too small for blake3_hasher");
    static_assert(
        alignof(std::max_align_t) >= alignof(blake3_hasher), "StableHasher inline storage alignment is insufficient");
    blake3_hasher_init(reinterpret_cast<blake3_hasher*>(state_storage_.data()));
}

void StableHasher::update(uint64_t data) { update(&data, sizeof(data)); }

void StableHasher::update(const void* data, std::size_t size) {
    blake3_hasher_update(reinterpret_cast<blake3_hasher*>(state_storage_.data()), data, size);
}

void StableHasher::update(std::string_view data) { update(data.data(), data.size()); }

uint64_t StableHasher::digest() const {
    uint8_t out[8];
    blake3_hasher_finalize(reinterpret_cast<const blake3_hasher*>(state_storage_.data()), out, sizeof(out));
    return load_le_u64(out);
}

}  // namespace tt
