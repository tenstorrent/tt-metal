// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>

#include <fmt/format.h>

#include "blake3.h"

namespace tt::tt_fabric::capture {

struct Blake3Hasher::State {
    blake3_hasher hasher{};
};

Blake3Hasher::Blake3Hasher() : state_(std::make_unique<State>()) { blake3_hasher_init(&state_->hasher); }

Blake3Hasher::~Blake3Hasher() = default;
Blake3Hasher::Blake3Hasher(Blake3Hasher&&) noexcept = default;
Blake3Hasher& Blake3Hasher::operator=(Blake3Hasher&&) noexcept = default;

void Blake3Hasher::update(std::span<const std::byte> bytes) {
    blake3_hasher_update(&state_->hasher, bytes.data(), bytes.size());
}

std::string Blake3Hasher::get_hex() const {
    std::array<uint8_t, BLAKE3_OUT_LEN> digest{};
    blake3_hasher_finalize(&state_->hasher, digest.data(), digest.size());
    std::string hex;
    hex.reserve(digest.size() * 2);
    for (const uint8_t byte : digest) {
        fmt::format_to(std::back_inserter(hex), "{:02x}", byte);
    }
    return hex;
}

std::string compute_blake3_hex(std::span<const std::byte> bytes) {
    Blake3Hasher hasher;
    hasher.update(bytes);
    return hasher.get_hex();
}

}  // namespace tt::tt_fabric::capture
