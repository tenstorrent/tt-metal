// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/regions/regions.hpp -- checked arena and cache-line utilities.
//
// Suites that use this arena share one placement problem: partition the
// tile's available LIM into a small number of whole-cache-line-aligned
// regions (args block, per-participant mailboxes, staging, result), prime
// every line so the firmware never touches an unprimed ECC line, and hand
// the resulting addresses back as absolute LIM addresses.
//
// This header names that arena. It has no dependency on the suite (p2p_latency /
// transport_bandwidth): a case declares its regions in-order and the arena
// vends absolute addresses.

#ifndef X280_HARNESS_REGIONS_HPP
#define X280_HARNESS_REGIONS_HPP

#include "kernels/shared/ubench_region.h"
#include "kernels/shared/ubench_units.h"
#include "x280/device/x280_tile.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace x280::harness::regions {

constexpr uint64_t kCacheLineBytes = UBENCH_LIM_LINE_BYTES;

inline uint64_t round_up(uint64_t n, uint64_t to) { return (n + to - 1u) & ~(to - 1u); }

// A single LIM region: an absolute address plus a rounded-up size in bytes.
// The size is always a multiple of kCacheLineBytes so a whole-line write
// admissible to write_available_lim always covers the region exactly.
using Region = ubench_region_t;

// LIM cursor allocator. Construct with the tile's `available_lim().begin`,
// then call `allocate(bytes)` in the order the layout header specifies.
// Every returned Region is whole-cache-line aligned by size (the base is
// aligned as long as `available_lim().begin` was, which x280_tile.cpp
// guarantees). An allocation that would end past the arena throws.
class Arena {
public:
    /** Throws std::invalid_argument when `r` ends past the 64-bit address space. */
    explicit Arena(const Region& r) : base_(r.addr), cursor_(r.addr), limit_(r.addr + r.bytes) {
        if (limit_ < base_) {
            std::ostringstream msg;
            msg << "Arena: " << r.bytes << "-byte region at 0x" << std::hex << r.addr << " wraps uint64_t";
            throw std::invalid_argument(msg.str());
        }
    }

    /**
     * Vends the next `bytes`, rounded up to whole cache lines. Throws
     * std::invalid_argument naming the request and the arena, leaving the
     * cursor unchanged, when the rounded size would end past the arena.
     */
    Region allocate(uint64_t bytes) {
        const uint64_t rounded = round_up(bytes, kCacheLineBytes);
        if (rounded < bytes || rounded > limit_ - cursor_) {
            std::ostringstream msg;
            msg << "Arena: " << bytes << "-byte allocation at 0x" << std::hex << cursor_ << " exceeds the arena [0x"
                << base_ << ", 0x" << limit_ << "), " << std::dec << (limit_ - cursor_) << " bytes left";
            throw std::invalid_argument(msg.str());
        }
        Region r{};
        r.addr = cursor_;
        r.bytes = rounded;
        cursor_ += r.bytes;
        return r;
    }

    uint64_t base() const { return base_; }
    uint64_t cursor() const { return cursor_; }
    uint64_t remaining() const { return limit_ - cursor_; }

private:
    uint64_t base_;
    uint64_t cursor_;
    uint64_t limit_;
};

// Prime a Region with zero bytes: one write_available_lim call per whole
// cache line. Blackhole ECC is written per-line, so a partial-line firmware
// store must be preceded by a whole-line host-primed write.
inline void zero_region(X280Tile& tile, const Region& r) {
    static const std::vector<uint8_t> zero_line(kCacheLineBytes, 0);
    for (uint64_t off = 0; off < r.bytes; off += kCacheLineBytes) {
        tile.write_available_lim(r.addr + off, zero_line);
    }
}

/** Throws std::invalid_argument naming both sizes when `bytes` does not fit `r`. */
inline void require_fits(const char* what, uint64_t bytes, const Region& r) {
    if (bytes > r.bytes) {
        std::ostringstream msg;
        msg << what << ": " << bytes << "-byte struct does not fit the " << r.bytes << "-byte region at 0x" << std::hex
            << r.addr;
        throw std::invalid_argument(msg.str());
    }
}

/**
 * Write an args block into a Region, padding to whole cache lines.
 *
 * `Tile` is X280Tile, or a host-test stand-in with the same
 * write_available_lim. Throws std::invalid_argument, before any write, when
 * `sizeof(T) > r.bytes`.
 */
template <typename T, typename Tile>
inline void write_args(Tile& tile, const Region& r, const T& block) {
    require_fits("write_args", sizeof(block), r);
    std::vector<uint8_t> bytes(r.bytes, 0);
    std::memcpy(bytes.data(), &block, sizeof(block));
    tile.write_available_lim(r.addr, bytes);
}

/**
 * Read a struct from the start of a Region.
 *
 * Throws std::invalid_argument, before any read, when `sizeof(T) > r.bytes`,
 * and std::runtime_error when the tile returns other than `sizeof(T)` bytes.
 */
template <typename T, typename Tile>
inline T read_struct(Tile& tile, const Region& r) {
    require_fits("read_struct", sizeof(T), r);
    const auto raw = tile.read_available_lim(r.addr, sizeof(T));
    if (raw.size() != sizeof(T)) {
        std::ostringstream msg;
        msg << "read_struct: read " << raw.size() << " bytes at 0x" << std::hex << r.addr << std::dec << ", expected "
            << sizeof(T);
        throw std::runtime_error(msg.str());
    }
    T out{};
    std::memcpy(&out, raw.data(), sizeof(T));
    return out;
}

}  // namespace x280::harness::regions

#endif  // X280_HARNESS_REGIONS_HPP
