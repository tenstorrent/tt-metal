// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

#include <array>
#include <cstdint>
#include <iterator>

#include <fmt/format.h>

#include "blake3.h"

namespace tt::tt_fabric::capture {

std::string compute_blake3_hex(std::span<const std::byte> bytes) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, bytes.data(), bytes.size());
    std::array<uint8_t, BLAKE3_OUT_LEN> digest{};
    blake3_hasher_finalize(&hasher, digest.data(), digest.size());

    std::string hex;
    hex.reserve(digest.size() * 2);
    for (const uint8_t byte : digest) {
        fmt::format_to(std::back_inserter(hex), "{:02x}", byte);
    }
    return hex;
}

}  // namespace tt::tt_fabric::capture
