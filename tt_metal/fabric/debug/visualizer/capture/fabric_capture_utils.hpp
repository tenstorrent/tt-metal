// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
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

// A BLAKE3 hash built up a piece at a time, for a file written a piece at a time. Its digest is compute_blake3_hex's
// of all the pieces together.
class Blake3Hasher {
public:
    Blake3Hasher();
    ~Blake3Hasher();
    Blake3Hasher(Blake3Hasher&&) noexcept;
    Blake3Hasher& operator=(Blake3Hasher&&) noexcept;
    Blake3Hasher(const Blake3Hasher&) = delete;
    Blake3Hasher& operator=(const Blake3Hasher&) = delete;

    void update(std::span<const std::byte> bytes);
    // The hash of everything passed to update so far, as 64 lowercase hex digits.
    std::string get_hex() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace tt::tt_fabric::capture
