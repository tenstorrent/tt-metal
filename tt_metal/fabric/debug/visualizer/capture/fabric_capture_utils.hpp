// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace tt::tt_fabric::capture {

// A core's translated NOC coordinates: the manifest's virtual_core, and what UMD gives for an Ethernet channel.
struct CoreXY {
    uint32_t x = 0;
    uint32_t y = 0;

    auto operator<=>(const CoreXY&) const = default;
};

// The BLAKE3 hash of `bytes`, as 64 lowercase hex digits: how a capture names the exact files it read and wrote.
std::string compute_blake3_hex(std::span<const std::byte> bytes);

}  // namespace tt::tt_fabric::capture
