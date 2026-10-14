// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Pure checked local/host address views for programmable-core L1 windows.

#ifndef X280_HARNESS_REGIONS_L1_WINDOW_HPP
#define X280_HARNESS_REGIONS_L1_WINDOW_HPP

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace x280::harness::regions {

// A device-local byte range [base, base + size).
struct Window {
    uint64_t base;
    uint64_t size;

    bool operator==(const Window&) const = default;
};

// local is the device-local range passed to kernels. host_noc_base is the
// corresponding host-facing address at local.base.
struct L1Window {
    Window local;
    uint64_t host_noc_base;

    bool operator==(const L1Window&) const = default;
};

// Aligns the device-local start upward while preserving the corresponding host
// address view. alignment must be a non-zero power of two. The skipped prefix
// is removed from size; throws std::invalid_argument if either address would
// overflow or the window is too small to reach an aligned byte.
inline L1Window align_window_start(const L1Window& window, uint64_t alignment) {
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
        throw std::invalid_argument("align_window_start: alignment must be a non-zero power of two");
    }
    if (window.local.base > std::numeric_limits<uint64_t>::max() - (alignment - 1u)) {
        throw std::invalid_argument("align_window_start: device-local base alignment overflows uint64_t");
    }
    const uint64_t aligned_base = (window.local.base + alignment - 1u) & ~(alignment - 1u);
    const uint64_t skipped = aligned_base - window.local.base;
    if (window.local.size == 0u || skipped >= window.local.size) {
        throw std::invalid_argument("align_window_start: window ends before its first aligned byte");
    }
    if (window.host_noc_base > std::numeric_limits<uint64_t>::max() - skipped) {
        throw std::invalid_argument("align_window_start: host address alignment overflows uint64_t");
    }
    return L1Window{
        .local = {.base = aligned_base, .size = window.local.size - skipped},
        .host_noc_base = window.host_noc_base + skipped,
    };
}

// Returns the local interval [base, base + size) as a view of window while
// preserving its tagged host address mapping. Throws std::invalid_argument
// for an empty or non-contained interval and for host-address overflow.
inline L1Window subwindow(const L1Window& window, uint64_t base, uint64_t size) {
    if (size == 0u || base < window.local.base) {
        throw std::invalid_argument("subwindow: interval is empty or begins before the window");
    }
    const uint64_t offset = base - window.local.base;
    if (offset > window.local.size || size > window.local.size - offset) {
        throw std::invalid_argument("subwindow: interval exceeds the window");
    }
    if (window.host_noc_base > std::numeric_limits<uint64_t>::max() - offset) {
        throw std::invalid_argument("subwindow: host address overflows uint64_t");
    }
    const uint64_t host_noc_base = window.host_noc_base + offset;
    if (size - 1u > std::numeric_limits<uint64_t>::max() - host_noc_base) {
        throw std::invalid_argument("subwindow: host range overflows uint64_t");
    }
    return L1Window{
        .local = {.base = base, .size = size},
        .host_noc_base = host_noc_base,
    };
}

// Translates a contained local byte range into its host-facing NoC address.
// Throws std::invalid_argument when the range is empty, outside window, or
// cannot be represented as a uint64_t host address.
inline uint64_t host_noc_address(const L1Window& window, uint64_t local_addr, uint64_t bytes) {
    const L1Window range = subwindow(window, local_addr, bytes);
    return range.host_noc_base;
}

}  // namespace x280::harness::regions

#endif  // X280_HARNESS_REGIONS_L1_WINDOW_HPP
