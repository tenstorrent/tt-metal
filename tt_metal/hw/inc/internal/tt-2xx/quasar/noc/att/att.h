// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @file
 * @brief Quasar ATT mask-table window model and constexpr address math.
 */

namespace noc_att {

/// @brief Mask covering the low @p bits bits.
constexpr std::uint64_t low_mask(std::uint32_t bits) {
    return bits == 64 ? UINT64_MAX : ((std::uint64_t{1} << bits) - 1);
}

/**
 * @brief Stable kernel-visible fields of one ATT mask-table window, with the
 * constexpr address math over them.
 *
 * The field names match the ATT programming model so a generated configuration
 * can be checked directly against its source data.
 *
 * mask_id is deliberately absent: it selects a hardware register slot but does
 * not change an address. Register addresses, mask IDs, and endpoint coordinates
 * belong to boot-owned programming data.
 */
struct Window {
    /// Window base: the address bits above the wildcard region. Named compare
    /// to match the hardware's field name.
    std::uint64_t compare;

    /// Number of low address bits ignored by the window comparison. A value N
    /// describes a 2^N-byte window rooted at compare. This is the
    /// hardware/QMK-dump convention (count of LOW wildcard bits) - the
    /// programming guide's inverted "MSB count" framing is not used.
    std::uint8_t mask_bits;

    /// Address bits [endpoint_shift +: endpoint_size] form the selector.
    std::uint8_t endpoint_shift;
    std::uint8_t endpoint_size;

    /// Hardware resolves endpoint_table_offset + selector in the endpoint table.
    std::uint16_t endpoint_table_offset;

    /// Whether the ATT rebases the address after matching this window.
    bool translate_address;

    /// @brief Number of selector values the window supports (1 when selector-free).
    constexpr std::uint32_t selector_limit() const {
        return endpoint_size == 0 ? 1u : (std::uint32_t{1} << endpoint_size);
    }

    /// @brief Width in bits of the window's local-address field.
    constexpr std::uint32_t local_address_bits() const { return endpoint_size == 0 ? mask_bits : endpoint_shift; }

    /// @brief One past the largest local address the window can carry.
    ///
    /// A window with no selector dedicates all ignored comparison bits to the
    /// local address. Otherwise the local address ends where the selector starts.
    constexpr std::uint64_t local_address_limit() const { return std::uint64_t{1} << local_address_bits(); }

    /// @brief Whether @p selector_value is representable in the selector field.
    constexpr bool selector_supported(std::uint32_t selector_value) const { return selector_value < selector_limit(); }

    /// @brief Whether a transfer of @p size bytes at @p local stays inside the
    /// window (zero-size transfers are rejected).
    constexpr bool transfer_supported(std::uint64_t local, std::uint64_t size = 1) const {
        const std::uint64_t limit = local_address_limit();
        return size != 0 && local < limit && size <= limit - local;
    }

    /// @brief Compose the full ATT address: window base | selector | local address.
    constexpr std::uint64_t make_address(std::uint32_t selector_value, std::uint64_t local) const {
        return compare | (std::uint64_t{selector_value} << endpoint_shift) | local;
    }

    /// @brief Extract the selector field from an address in this window.
    constexpr std::uint32_t selector(std::uint64_t address) const {
        return endpoint_size == 0 ? 0u
                                  : static_cast<std::uint32_t>((address >> endpoint_shift) & low_mask(endpoint_size));
    }

    /// @brief Endpoint-table row the address resolves to (offset + selector).
    constexpr std::uint16_t endpoint_index(std::uint64_t address) const {
        return static_cast<std::uint16_t>(endpoint_table_offset + selector(address));
    }

    /// @brief Extract the local-address field from an address in this window.
    constexpr std::uint64_t local_address(std::uint64_t address) const {
        return address & low_mask(local_address_bits());
    }

    /// @brief Whether @p address falls inside the window's compare range.
    constexpr bool matches(std::uint64_t address) const { return (address & ~low_mask(mask_bits)) == compare; }
};

}  // namespace noc_att
