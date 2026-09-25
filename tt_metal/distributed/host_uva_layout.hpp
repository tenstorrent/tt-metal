// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Region geometry. The register file is gone: the device touches only its own socket FIFO,
// so what remains is arena offsets, the credit array and the header both sides check.
#pragma once

#include <stdint.h>

namespace tt::tt_metal::experimental {

constexpr uint32_t kProvisionedCores = 128;

constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kAlign2M = 2ull * 1024ull * 1024ull;
constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// Credits: the only backward-flowing state, and the only region bytes that are neither
// header nor arena. A receiver writes an absolute count for the sending core.
constexpr uint32_t kMaxCreditPeers = 8;
// The credit line, not the UVA selector, is what caps the host count.
constexpr uint32_t kMaxHosts = kMaxCreditPeers;

// What the credit array can index is kMaxHosts; what the H2H RX ring can actually serve is
// this. rx_slot_offset() has no host dimension, so a second sender collides -- raise together.
constexpr uint32_t kMaxH2HHostsSupported = 2;

// Per-frame latency samples stop at this many: past it the percentiles have long converged,
// and a volume run would otherwise grow the vectors -- and realloc them -- while it is timed.
constexpr uint64_t kMaxTimingSamples = 4ull << 20;

// Packed [peer][core]: cores are adjacent at one peer, so a pass of credits arms in ONE put.
// Each peer owns whole 64-byte lines (1 KiB), so peers still never contend for one.
constexpr uint64_t kCreditPeerStride = static_cast<uint64_t>(kProvisionedCores) * sizeof(uint64_t);
constexpr uint64_t kCreditArrayOffset = kHeaderBytes;
constexpr uint64_t kCreditArrayBytes = static_cast<uint64_t>(kMaxCreditPeers) * kCreditPeerStride;

constexpr uint64_t credit_offset(uint32_t core, uint32_t peer) {
    return kCreditArrayOffset + static_cast<uint64_t>(peer) * kCreditPeerStride +
           static_cast<uint64_t>(core) * sizeof(uint64_t);
}

// A SECOND count, keyed on the SENDING core rather than the receiving one. The credit above
// frees a slot in one core's ring; this says what a given sender's frames have reached.
constexpr uint64_t kDoneArrayOffset = kCreditArrayOffset + kCreditArrayBytes;
constexpr uint64_t kDoneArrayBytes = kCreditArrayBytes;

constexpr uint64_t done_offset(uint32_t core, uint32_t peer) {
    return kDoneArrayOffset + static_cast<uint64_t>(peer) * kCreditPeerStride +
           static_cast<uint64_t>(core) * sizeof(uint64_t);
}

// One 8-byte guard per (core, slot), CONTIGUOUS. The device still writes a guard into each
// page's trailer, but the h2h hop publishes here instead: a run of K slots then arms in one
// put rather than K, which is what takes puts-per-frame from 2 to 2/K. A put coalesces only
// when its DESTINATION bytes are adjacent, so the relocation is the mechanism, not a tidy-up.
constexpr uint32_t kMaxRingSlots = 64;
constexpr uint64_t kGuardArrayOffset = kDoneArrayOffset + kDoneArrayBytes;
constexpr uint64_t kGuardArrayBytes =
    static_cast<uint64_t>(kProvisionedCores) * kMaxRingSlots * sizeof(uint64_t);

constexpr uint64_t guard_offset(uint32_t core, uint32_t slot) {
    return kGuardArrayOffset + (static_cast<uint64_t>(core) * kMaxRingSlots + slot) * sizeof(uint64_t);
}

// Arenas interleaved per core (TX = D2H FIFO, RX = H2D ring) so a run pins a PREFIX:
// pinned_bytes_for(cores) covers only cores in use. Both are overlaid by RingAlias.
constexpr uint64_t kArenaBytes = 1536ull * 1024ull;  // one Tensix L1
constexpr uint64_t kArenasPerCore = 2;
constexpr uint64_t kArenaStride = kArenaBytes * kArenasPerCore;

// 2 MiB-aligned so the block below it can be resized without shifting every arena -- the
// guard array was added under it and the arenas did not move.
constexpr uint64_t kArenaArrayOffset = align_up(kGuardArrayOffset + kGuardArrayBytes, kAlign2M);

constexpr uint64_t tx_arena_offset(uint32_t core) {
    return kArenaArrayOffset + static_cast<uint64_t>(core) * kArenaStride;
}
constexpr uint64_t rx_arena_offset(uint32_t core) { return tx_arena_offset(core) + kArenaBytes; }

// data_offset is where the aliased socket's ring starts inside the arena, and it is NOT
// defaulted: omitting it silently addressed the wrong bytes, so every caller must say.
constexpr uint64_t rx_slot_offset(uint32_t core, uint32_t slot, uint64_t page_bytes, uint64_t data_offset) {
    return rx_arena_offset(core) + data_offset + static_cast<uint64_t>(slot) * page_bytes;
}

// Ring depth in frames. The D2H leg, the H2H window and the peer's ring must all use this
// number or a run corrupts silently.
constexpr uint32_t kNumAliasRingSlots = 1;

// A core derives its own index from its firmware coordinates, so it structurally cannot
// address another core's arena.
constexpr uint32_t tt_uva_core_index(uint32_t logical_x, uint32_t logical_y, uint32_t grid_width) {
    return logical_y * grid_width + logical_x;
}

constexpr uint64_t pinned_bytes_for(uint32_t cores) {
    return kArenaArrayOffset + static_cast<uint64_t>(cores) * kArenaStride;
}

// The header. Two parties disagreeing on geometry compute different offsets for one core
// and each reads bytes that are legitimately idle, so the constants are published.
constexpr uint64_t kRegionMagic = 0x543648'4F535456ull;  // "T6HOSTV"
// 8 repacked the credit and done arrays to [peer][core] so credits coalesce; a v7 peer
// writes a credit where a v8 reader does not look, and the sender's ring gate never opens.
constexpr uint32_t kRegionVersion = 8;

struct RegionHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t provisioned_cores;
    uint64_t arena_bytes;
    uint64_t arena_stride;
    uint64_t credit_array_bytes;
    uint64_t done_array_bytes;
    uint64_t arena_array_offset;
    uint32_t cores_in_use;
    uint32_t host_id;
    uint32_t chips_per_host;
    uint32_t chip;
    // A mismatch here does not corrupt an offset, it silently names a different core.
    uint32_t grid_width;
    uint32_t grid_height;
    uint64_t pinned_bytes;
    uint64_t device_io_base;
    uint32_t pcie_xy_enc;
    uint32_t reserved;
};

}  // namespace tt::tt_metal::experimental
