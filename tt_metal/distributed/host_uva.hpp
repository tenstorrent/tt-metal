// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified Virtual Address (UVA) encoding for the host-target register file.
//
// Included by the host program AND by the RV32 Tensix kernels: <stdint.h> and constexpr
// only.
#pragma once

#include <stdint.h>

namespace tt::tt_metal::experimental {

// ---------------------------------------------------------------------------
// Field geometry
//
//   [63:60]  region kind        4 bits   DRAM / Host / T6 / RdmaReg
//   [59:48]  region selector   12 bits   Host: host identifier
//                                        T6, RdmaReg: global (host, chip, core)
//   [47:32]  page / subregion  16 bits
//   [31:0]   byte offset       32 bits
// ---------------------------------------------------------------------------
constexpr uint32_t kRegionShift = 60;
constexpr uint32_t kRegionBits = 4;
constexpr uint32_t kSelectorShift = 48;
constexpr uint32_t kSelectorBits = 12;
constexpr uint32_t kPageShift = 32;
constexpr uint32_t kPageBits = 16;
constexpr uint32_t kOffsetShift = 0;
constexpr uint32_t kOffsetBits = 32;

constexpr uint64_t field_mask(uint32_t bits) { return (bits >= 64) ? ~0ull : ((1ull << bits) - 1ull); }

constexpr uint64_t kRegionMask = field_mask(kRegionBits);
constexpr uint64_t kSelectorMask = field_mask(kSelectorBits);
constexpr uint64_t kPageMask = field_mask(kPageBits);
constexpr uint64_t kOffsetMask = field_mask(kOffsetBits);

static_assert(kRegionBits + kSelectorBits + kPageBits + kOffsetBits == 64, "UVA fields must tile 64 bits");
static_assert(kRegionShift == kSelectorShift + kSelectorBits, "region must abut selector");
static_assert(kSelectorShift == kPageShift + kPageBits, "selector must abut page");
static_assert(kPageShift == kOffsetShift + kOffsetBits, "page must abut offset");

enum UvaRegion : uint32_t {
    kRegionDram = 0,
    kRegionHost = 1,
    kRegionT6 = 2,
    kRegionRdmaReg = 3,
    kRegionCount
};

static_assert(kRegionCount <= (1u << kRegionBits), "region kinds must fit the region field");

// ---------------------------------------------------------------------------
// A T6 / RdmaReg selector is a GLOBAL (host, chip, core) address.
//
//   selector = (host * chips_per_host + chip) * kT6CoresPerChip + core
//
// POSITIONAL, and it must stay that way, because these words are FORWARDED. A UVA
// travels T6 -> host -> another host -> that host's T6 and no hop rewrites it, so its
// meaning cannot depend on who is holding it. Putting locality in the region nibble
// would make a forwarded word silently retarget the transfer at the first hop.
// ---------------------------------------------------------------------------
constexpr uint32_t kT6CoresPerChip = 256;
constexpr uint32_t kT6MaxSlots = (static_cast<uint32_t>(kSelectorMask) + 1) / kT6CoresPerChip;  // 16

constexpr uint32_t t6_slot(uint32_t host, uint32_t chip, uint32_t chips_per_host) {
    return host * chips_per_host + chip;
}
constexpr uint32_t t6_host_stride(uint32_t chips_per_host) { return chips_per_host * kT6CoresPerChip; }
constexpr uint32_t t6_global_selector(uint32_t host, uint32_t chip, uint32_t core, uint32_t chips_per_host) {
    return t6_slot(host, chip, chips_per_host) * kT6CoresPerChip + core;
}
constexpr uint32_t t6_selector_host(uint32_t selector, uint32_t chips_per_host) {
    return selector / t6_host_stride(chips_per_host);
}
constexpr uint32_t t6_selector_chip(uint32_t selector, uint32_t chips_per_host) {
    return (selector / kT6CoresPerChip) % chips_per_host;
}
constexpr uint32_t t6_selector_core(uint32_t selector) { return selector % kT6CoresPerChip; }
constexpr uint32_t t6_selector_slot(uint32_t selector) { return selector / kT6CoresPerChip; }

static_assert(kT6MaxSlots == 16, "12 selector bits over 256 cores is exactly 16 slots");
static_assert(t6_selector_host(t6_global_selector(2, 3, 17, 4), 4) == 2, "host must round-trip");
static_assert(t6_selector_chip(t6_global_selector(2, 3, 17, 4), 4) == 3, "chip must round-trip");
static_assert(t6_selector_core(t6_global_selector(2, 3, 17, 4)) == 17, "core must round-trip");
static_assert(t6_selector_slot(t6_global_selector(2, 3, 17, 4)) == t6_slot(2, 3, 4), "slot must round-trip");
static_assert(t6_global_selector(0, 0, 42, 1) == 42, "host 0 chip 0: the selector is the core index");

// ---------------------------------------------------------------------------
// Host identity is CONFIGURED, not inferred. How many hosts take part and which one this
// process is are facts about a deployment, so they are supplied rather than discovered:
//
//   TT_RDMA_HOST_NUM / --host-num        how many hosts are in this system
//   TT_RDMA_HOST_IDENT / --host-ident    which of them this process runs on
//   TT_RDMA_CHIPS_PER_HOST               the T6 selector's slot stride
//
// With NUM = 1 and IDENT = 0 -- the single-host, two-process case -- every legal host
// UVA is local and any selector above 0 is invalid. That matters for the requirement
// that address translation be exercised even at one host: the routing decision still
// runs, it just always answers "local", and the no-such-host branch is reachable from
// the configuration people actually run rather than only a hypothetical one.
// ---------------------------------------------------------------------------
enum HostReach : uint32_t {
    kHostReachNoSuchHost = 0,
    kHostReachLocal = 1,
    kHostReachRemote = 2,
    kHostReachCount
};

struct HostTopology {
    uint32_t ident;
    uint32_t num;
    uint32_t chips_per_host;
};

constexpr bool host_topology_ok(HostTopology t) {
    return t.num >= 1 && t.ident < t.num && t.chips_per_host >= 1 &&
           (t.num - 1) <= static_cast<uint32_t>(kSelectorMask) &&
           static_cast<uint64_t>(t.num) * t.chips_per_host * kT6CoresPerChip <=
               static_cast<uint64_t>(kSelectorMask) + 1;
}

constexpr uint32_t my_t6_slot(HostTopology t, uint32_t my_chip) {
    return t6_slot(t.ident, my_chip, t.chips_per_host);
}

constexpr uint32_t host_reach(uint32_t selector, HostTopology t) {
    if (selector >= t.num) {
        return kHostReachNoSuchHost;
    }
    return selector == t.ident ? kHostReachLocal : kHostReachRemote;
}

static_assert(host_topology_ok(HostTopology{0, 1, 1}), "one host, one chip, and we are it: the default");
static_assert(!host_topology_ok(HostTopology{0, 0, 1}), "zero hosts is not a system");
static_assert(!host_topology_ok(HostTopology{1, 1, 1}), "this host must be one of the hosts");
static_assert(!host_topology_ok(HostTopology{0, 2, 0}), "a host with no chips cannot be addressed");
static_assert(host_topology_ok(HostTopology{0, 16, 1}), "16 hosts x 1 chip fills the selector exactly");
static_assert(!host_topology_ok(HostTopology{0, 17, 1}), "17 hosts x 1 chip does not fit");
static_assert(host_topology_ok(HostTopology{0, 8, 2}), "8 hosts x 2 chips fills it exactly");
static_assert(!host_topology_ok(HostTopology{0, 2, 16}), "2 hosts x 16 chips does not fit");
static_assert(host_reach(0, HostTopology{0, 1, 1}) == kHostReachLocal, "single host: 0 is us");
static_assert(host_reach(1, HostTopology{0, 1, 1}) == kHostReachNoSuchHost, "single host: 1 exists nowhere");
static_assert(host_reach(1, HostTopology{0, 2, 1}) == kHostReachRemote, "two hosts: 1 is the peer");
static_assert(my_t6_slot(HostTopology{2, 8, 2}, 1) == 5, "host 2's chip 1 at stride 2 is slot 5");

inline const char* host_reach_name(uint32_t r) {
    switch (r) {
        case kHostReachNoSuchHost: return "no such host";
        case kHostReachLocal: return "this host";
        case kHostReachRemote: return "remote host";
        default: return "?";
    }
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------
constexpr uint64_t uva_encode(uint32_t region, uint32_t selector, uint32_t page, uint32_t offset) {
    return ((static_cast<uint64_t>(region) & kRegionMask) << kRegionShift) |
           ((static_cast<uint64_t>(selector) & kSelectorMask) << kSelectorShift) |
           ((static_cast<uint64_t>(page) & kPageMask) << kPageShift) |
           ((static_cast<uint64_t>(offset) & kOffsetMask) << kOffsetShift);
}

constexpr uint32_t uva_region(uint64_t uva) { return static_cast<uint32_t>((uva >> kRegionShift) & kRegionMask); }
constexpr uint32_t uva_selector(uint64_t uva) {
    return static_cast<uint32_t>((uva >> kSelectorShift) & kSelectorMask);
}
constexpr uint32_t uva_page(uint64_t uva) { return static_cast<uint32_t>((uva >> kPageShift) & kPageMask); }
constexpr uint32_t uva_offset(uint64_t uva) { return static_cast<uint32_t>((uva >> kOffsetShift) & kOffsetMask); }

constexpr bool uva_selector_is_t6(uint64_t uva) {
    const uint32_t r = uva_region(uva);
    return r == kRegionT6 || r == kRegionRdmaReg;
}

constexpr uint32_t uva_t6_host(uint64_t uva, uint32_t chips_per_host) {
    return t6_selector_host(uva_selector(uva), chips_per_host);
}
constexpr uint32_t uva_t6_chip(uint64_t uva, uint32_t chips_per_host) {
    return t6_selector_chip(uva_selector(uva), chips_per_host);
}
constexpr uint32_t uva_t6_core(uint64_t uva) { return t6_selector_core(uva_selector(uva)); }
constexpr uint32_t uva_t6_slot(uint64_t uva) { return t6_selector_slot(uva_selector(uva)); }

// A Host-region UVA carries the host identifier directly in its selector. A T6 or
// RdmaReg UVA carries it inside the positional selector and has to be divided out. Both
// answer the same three-way question, so both go through one function -- a second copy
// of this dispatch is how a Host-region word and a T6 word end up routed by different
// rules that agree right up until they do not.
constexpr uint32_t uva_host_reach(uint64_t uva, HostTopology t) {
    if (uva_selector_is_t6(uva)) {
        return host_reach(uva_t6_host(uva, t.chips_per_host), t);
    }
    if (uva_region(uva) == kRegionHost) {
        return host_reach(uva_selector(uva), t);
    }
    // DRAM is chip-local by construction: it names a bank on a chip, and there is no
    // host field to consult. Calling that "local" is right, and calling it "no such
    // host" would make a legal local address look like a corrupt one.
    return kHostReachLocal;
}

// host_reach() answers local/remote/no-such-host, which is all a two-host run ever needed:
// "remote" had exactly one meaning because there was exactly one peer. With more than two
// hosts "remote" stops identifying anything, and the destination host has to survive the
// routing decision so the sender can pick an endpoint.
//
// This is the DECODE half only, and it is symmetric -- anyone holding the UVA can call it.
// Turning the answer into a connected endpoint is RESOLUTION, needs a table only the host
// has, and is deliberately not here. Same split an on-chip resolver draws between decoding a
// UVA and resolving it against the published noc_xy table.
//
// kHostNone for an address with no host field: DRAM names a bank on a chip and is chip-local
// by construction, so "no host" is the right answer rather than host 0.
constexpr uint32_t kHostNone = 0xFFFFFFFFu;

constexpr uint32_t uva_target_host(uint64_t uva, HostTopology t) {
    if (uva_selector_is_t6(uva)) {
        return uva_t6_host(uva, t.chips_per_host);
    }
    if (uva_region(uva) == kRegionHost) {
        return uva_selector(uva);
    }
    return kHostNone;
}
static_assert(uva_target_host(uva_encode(kRegionT6, t6_global_selector(2, 1, 9, 4), 0, 0),
                              HostTopology{0, 8, 4}) == 2,
              "a t6 UVA names its host");
static_assert(uva_target_host(uva_encode(kRegionDram, 3, 0, 0), HostTopology{0, 8, 4}) == kHostNone,
              "dram has no host field");


constexpr bool uva_selector_fits(uint32_t selector) { return (selector & ~static_cast<uint32_t>(kSelectorMask)) == 0; }
constexpr bool uva_region_fits(uint32_t region) { return (region & ~static_cast<uint32_t>(kRegionMask)) == 0; }

inline const char* uva_region_name(uint32_t r) {
    switch (r) {
        case kRegionDram: return "dram";
        case kRegionHost: return "host";
        case kRegionT6: return "t6";
        case kRegionRdmaReg: return "rdmareg";
        default: return "?";
    }
}

}  // namespace tt::tt_metal::experimental
