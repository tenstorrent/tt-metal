// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/stable_hash.hpp"

#include "blake3.h"

#include <cstddef>
#include <cstdint>
#include <memory>

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

// Serialize a uint64_t as 8 little-endian bytes so the hash input is
// independent of host byte order.
void store_le_u64(uint64_t v, uint8_t bytes[8]) {
    bytes[0] = static_cast<uint8_t>(v);
    bytes[1] = static_cast<uint8_t>(v >> 8);
    bytes[2] = static_cast<uint8_t>(v >> 16);
    bytes[3] = static_cast<uint8_t>(v >> 24);
    bytes[4] = static_cast<uint8_t>(v >> 32);
    bytes[5] = static_cast<uint8_t>(v >> 40);
    bytes[6] = static_cast<uint8_t>(v >> 48);
    bytes[7] = static_cast<uint8_t>(v >> 56);
}

}  // namespace

struct StableHasher::Impl {
    blake3_hasher hasher;
};

StableHasher::StableHasher() : impl_(std::make_unique<Impl>()) { blake3_hasher_init(&impl_->hasher); }

StableHasher::~StableHasher() = default;
StableHasher::StableHasher(StableHasher&&) noexcept = default;
StableHasher& StableHasher::operator=(StableHasher&&) noexcept = default;

void StableHasher::update(uint64_t data) {
    uint8_t bytes[8];
    store_le_u64(data, bytes);
    update(bytes, sizeof(bytes));
}

void StableHasher::update(const void* data, std::size_t size) { blake3_hasher_update(&impl_->hasher, data, size); }

void StableHasher::update(std::string_view data) { update(data.data(), data.size()); }

uint64_t StableHasher::digest() const {
    uint8_t out[8];
    blake3_hasher_finalize(&impl_->hasher, out, sizeof(out));
    return load_le_u64(out);
}

}  // namespace tt
