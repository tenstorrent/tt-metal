// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified Virtual Address: one 64-bit word naming a region, a target and a byte offset.
// Included by RV32 kernels, so <stdint.h> and constexpr only.
#pragma once

#include <stdint.h>

// For kMaxHosts: the credit array, not the selector, is the tighter bound on host count.
#include "tt_metal/distributed/host_uva_layout.hpp"

namespace tt::tt_metal::experimental {

// [63:60] region | [59:48] selector | [47:32] page | [31:0] offset
constexpr uint32_t kRegionShift = 60;
constexpr uint32_t kRegionBits = 4;
constexpr uint32_t kSelectorShift = 48;
constexpr uint32_t kSelectorBits = 12;
constexpr uint32_t kPageShift = 32;
constexpr uint32_t kPageBits = 16;
constexpr uint32_t kOffsetShift = 0;
constexpr uint32_t kOffsetBits = 32;

constexpr uint64_t field_mask(uint32_t bits) { return bits >= 64 ? ~0ull : ((1ull << bits) - 1ull); }

constexpr uint64_t kRegionMask = field_mask(kRegionBits);
constexpr uint64_t kSelectorMask = field_mask(kSelectorBits);
constexpr uint64_t kPageMask = field_mask(kPageBits);
constexpr uint64_t kOffsetMask = field_mask(kOffsetBits);

static_assert(kRegionBits + kSelectorBits + kPageBits + kOffsetBits == 64, "UVA fields must tile 64 bits");
static_assert(kRegionShift == kSelectorShift + kSelectorBits, "region must abut selector");
static_assert(kSelectorShift == kPageShift + kPageBits, "selector must abut page");
static_assert(kPageShift == kOffsetShift + kOffsetBits, "page must abut offset");

// Scoped: will not implicitly convert to or from an L1 address or a byte count.
enum class tt_uva_t : uint64_t {};

// Zero is also a well-formed kRegionDram address; this spelling means "no destination".
constexpr tt_uva_t kUvaNone = static_cast<tt_uva_t>(0);

constexpr uint64_t tt_uva_bits(tt_uva_t u) { return static_cast<uint64_t>(u); }

enum UvaRegion : uint32_t { kRegionDram = 0, kRegionHost = 1, kRegionT6 = 2, kRegionRdmaReg = 3, kRegionCount };
static_assert(kRegionCount <= (1u << kRegionBits), "region kinds must fit the region field");

// Positional, because a UVA is forwarded across hops unrewritten and so cannot mean
// something different depending on who holds it.
constexpr uint32_t kT6CoresPerChip = 256;

constexpr uint32_t tt_uva_t6_slot(uint32_t host, uint32_t chip, uint32_t chips_per_host) {
    return host * chips_per_host + chip;
}
constexpr uint32_t tt_uva_t6_host_stride(uint32_t chips_per_host) { return chips_per_host * kT6CoresPerChip; }
constexpr uint32_t tt_uva_t6_global_selector(uint32_t host, uint32_t chip, uint32_t core, uint32_t chips_per_host) {
    return tt_uva_t6_slot(host, chip, chips_per_host) * kT6CoresPerChip + core;
}
constexpr uint32_t tt_uva_t6_selector_host(uint32_t sel, uint32_t chips_per_host) {
    return sel / tt_uva_t6_host_stride(chips_per_host);
}
constexpr uint32_t tt_uva_t6_selector_core(uint32_t sel) { return sel % kT6CoresPerChip; }

static_assert(tt_uva_t6_selector_host(tt_uva_t6_global_selector(2, 3, 17, 4), 4) == 2, "host must round-trip");
static_assert(tt_uva_t6_selector_core(tt_uva_t6_global_selector(2, 3, 17, 4)) == 17, "core must round-trip");
static_assert(tt_uva_t6_global_selector(0, 0, 42, 1) == 42, "host 0 chip 0: the selector is the core index");

constexpr tt_uva_t tt_uva_encode(uint32_t region, uint32_t selector, uint32_t page, uint32_t offset) {
    return static_cast<tt_uva_t>(
        ((static_cast<uint64_t>(region) & kRegionMask) << kRegionShift) |
        ((static_cast<uint64_t>(selector) & kSelectorMask) << kSelectorShift) |
        ((static_cast<uint64_t>(page) & kPageMask) << kPageShift) |
        ((static_cast<uint64_t>(offset) & kOffsetMask) << kOffsetShift));
}

constexpr tt_uva_t tt_uva_t6_from_selector(uint32_t selector, uint32_t offset) {
    return tt_uva_encode(kRegionT6, selector, 0, offset);
}

constexpr uint32_t tt_uva_region(tt_uva_t u) {
    return static_cast<uint32_t>((tt_uva_bits(u) >> kRegionShift) & kRegionMask);
}
constexpr uint32_t tt_uva_selector(tt_uva_t u) {
    return static_cast<uint32_t>((tt_uva_bits(u) >> kSelectorShift) & kSelectorMask);
}
constexpr uint32_t tt_uva_offset(tt_uva_t u) {
    return static_cast<uint32_t>((tt_uva_bits(u) >> kOffsetShift) & kOffsetMask);
}

constexpr bool tt_uva_selector_is_t6(tt_uva_t u) {
    const uint32_t r = tt_uva_region(u);
    return r == kRegionT6 || r == kRegionRdmaReg;
}
constexpr uint32_t tt_uva_t6_host(tt_uva_t u, uint32_t chips_per_host) {
    return tt_uva_t6_selector_host(tt_uva_selector(u), chips_per_host);
}
constexpr uint32_t tt_uva_t6_core(tt_uva_t u) { return tt_uva_t6_selector_core(tt_uva_selector(u)); }

struct HostTopology {
    uint32_t ident;
    uint32_t num;
    uint32_t chips_per_host;
};

constexpr bool host_topology_ok(HostTopology t) {
    return t.num >= 1 && t.num <= kMaxHosts && t.ident < t.num && t.chips_per_host >= 1 &&
           (t.num - 1) <= static_cast<uint32_t>(kSelectorMask) &&
           static_cast<uint64_t>(t.num) * t.chips_per_host * kT6CoresPerChip <=
               static_cast<uint64_t>(kSelectorMask) + 1;
}

// DRAM names a bank on a chip and carries no host field, so it has no target host.
constexpr uint32_t kHostNone = 0xFFFFFFFFu;

constexpr uint32_t tt_uva_target_host(tt_uva_t u, HostTopology t) {
    if (tt_uva_selector_is_t6(u)) {
        return tt_uva_t6_host(u, t.chips_per_host);
    }
    if (tt_uva_region(u) == kRegionHost) {
        return tt_uva_selector(u);
    }
    return kHostNone;
}

}  // namespace tt::tt_metal::experimental
