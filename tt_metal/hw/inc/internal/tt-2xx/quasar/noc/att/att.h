// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace noc_att {

// Stable kernel-visible fields of one ATT mask-table window. The names match the ATT
// programming model so a generated configuration can be checked directly against
// its source data.
//
// mask_id is deliberately absent: it selects a hardware register slot but does
// not change an address. Register addresses, mask IDs, and endpoint coordinates
// belong to boot-owned programming data.
struct Window {
    uint64_t compare;

    // Number of low address bits ignored by the window comparison. A value N
    // describes a 2^N-byte window rooted at compare.
    uint8_t mask_bits;

    // Address bits [endpoint_shift +: endpoint_size] form the selector.
    uint8_t endpoint_shift;
    uint8_t endpoint_size;

    // Hardware resolves endpoint_table_offset + selector in the endpoint table.
    uint16_t endpoint_table_offset;

    // Whether the ATT rebases the address after matching this window.
    bool translate_address;
};

constexpr uint64_t low_mask(uint32_t bits) { return bits == 64 ? UINT64_MAX : ((uint64_t{1} << bits) - 1); }

constexpr uint64_t local_address_limit(const Window& window) {
    // A window with no selector dedicates all ignored comparison bits to the
    // local address. Otherwise the local address ends where the selector starts.
    return uint64_t{1} << (window.endpoint_size == 0 ? window.mask_bits : window.endpoint_shift);
}

constexpr uint64_t make_address(const Window& window, uint32_t selector, uint64_t local_address) {
    return window.compare | (uint64_t{selector} << window.endpoint_shift) | local_address;
}

constexpr uint64_t make_local_address(const Window& window, uint64_t local_address) {
    return window.compare | local_address;
}

constexpr bool matches(const Window& window, uint64_t address) {
    return (address & ~low_mask(window.mask_bits)) == window.compare;
}

constexpr uint64_t replace_local_address(const Window& window, uint64_t global_address, uint64_t new_local_address) {
    return (global_address & ~low_mask(window.endpoint_shift)) | new_local_address;
}

constexpr bool valid(const Window& window) {
    return window.mask_bits < 64 && window.endpoint_shift < 64 && window.endpoint_size < 32 &&
           uint32_t{window.endpoint_shift} + uint32_t{window.endpoint_size} <= window.mask_bits &&
           (window.compare & low_mask(window.mask_bits)) == 0;
}

}  // namespace noc_att
