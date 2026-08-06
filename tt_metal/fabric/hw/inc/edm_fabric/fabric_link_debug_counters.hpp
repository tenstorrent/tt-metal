// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "dev_mem_map.h"
#include "internal/risc_attribs.h"
#include "api/debug/ring_buffer.h"

// ─── Fabric ERISC router TX/RX packet counters (DIAGNOSTIC -- not for merge) ───
//
// Counts eth packets pushed onto and consumed from the link. The counters live in
// ERISC L1 so they survive a hang: read them post-mortem off the wedged device with
// exalens (read_fabric_counters.py), or watch them live in the watcher ring buffer.
//
// Carved out of the FabricTelemetry region at +64, which is unused while
// TT_METAL_FABRIC_TELEMETRY is off.
//
// FIELD OFFSETS ARE FROZEN. read_fabric_counters.py and FABRIC_PACKET_LOSS_REPRO.md
// index this region by WORD NUMBER, so append at the end only -- never reorder or
// insert. The static_asserts below turn a moved base into a compile error rather than
// plausible-looking shifted garbage.

namespace tt::tt_fabric::debug {

// Marks an L1 slot belonging to a real fabric-router ERISC. Non-router eth cores never
// write it, so readers filter on it instead of summing in unrelated cores' noise.
constexpr uint32_t kFabricLinkDbgMagic = 0x00C0FFEE;

struct FabricLinkCounters {
    uint32_t tx;               // word 0  packets pushed onto the eth link (after the TXQ drain => on the wire)
    uint32_t rx;               // word 1  packets consumed from the eth link
    uint32_t fused_inc_total;  // word 2  fused write+atomic-inc executed locally (any semaphore)
    uint32_t fused_inc_r3;     // word 3  ... of those, targeting the SDPA R3 semaphore
    uint32_t last_fused_sem;   // word 4  low word of the last fused-inc semaphore address
    uint32_t tx_r3;            // word 5  R3 packets pushed
    uint32_t txq;              // word 6  packets handed to the eth TXQ hardware
    uint32_t magic;            // word 7  kFabricLinkDbgMagic
    uint32_t txq_r3;           // word 8  ... of those, R3
    uint32_t rb_pushes;        // word 9  ring-buffer dumps emitted (dump liveness)
};

static_assert(sizeof(FabricLinkCounters) == 40, "layout is frozen -- readers index this region by word");

constexpr uint32_t kFabricLinkCountersAddr = MEM_AERISC_FABRIC_TELEMETRY_BASE + 64;

// read_fabric_counters.py hardcodes this address as BASE. If the ERISC L1 map shifts,
// fail the build instead of silently reading the wrong words.
static_assert(
    kFabricLinkCountersAddr == 458256, "counter base moved -- update BASE in read_fabric_counters.py to match");
static_assert(
    kFabricLinkCountersAddr + sizeof(FabricLinkCounters) <= MEM_AERISC_FABRIC_POSTCODES_BASE,
    "counters overrun the fabric postcode/scratch region");

inline volatile tt_l1_ptr FabricLinkCounters* fabric_link_counters() {
    return reinterpret_cast<volatile tt_l1_ptr FabricLinkCounters*>(kFabricLinkCountersAddr);
}

// Zero once per boot, magic-word guarded, so counts are cumulative across the run
// rather than starting from L1 garbage. Magic is written LAST: a reader that sees the
// magic is guaranteed to see zeroed-or-later counters, never a half-initialized slot.
inline void fabric_link_counters_init() {
    auto* counters = fabric_link_counters();
    if (counters->magic == kFabricLinkDbgMagic) {
        return;
    }
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kFabricLinkCountersAddr);
    constexpr uint32_t num_words = sizeof(FabricLinkCounters) / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; ++i) {
        words[i] = 0;
    }
    counters->magic = kFabricLinkDbgMagic;
}

// ─── Watcher ring-buffer dump ─────────────────────────────────────────────────
//
// Tag in [31:24], value in [23:0], matching the convention in
// tt_metal/impl/buffers/kernels/dram_core_prefetcher.cpp. 24 bits holds 16.7M
// packets, well past the ~15k/run these counters reach.
constexpr uint32_t kRbTagTx = 0xF1000000u;
constexpr uint32_t kRbTagRx = 0xF2000000u;
constexpr uint32_t kRbValueMask = 0x00FFFFFFu;

// Emit a (TX, RX) pair once this many packets have moved. The ring buffer holds only
// DEBUG_RING_BUFFER_ELEMENTS (32) entries, so pushing every router iteration would
// evict the whole buffer within microseconds and bury anything else in it.
//
// Deliberately change-driven, not a fixed heartbeat: when a link wedges the counters
// stop advancing, no further pushes happen, and the last real values stay visible in
// the buffer -- which is exactly the evidence a hang needs. A time-based heartbeat
// would overwrite them with duplicates of the frozen value.
constexpr uint32_t kRbPushIntervalPackets = 1024;

// WATCHER_RING_BUFFER_PUSH expands to nothing unless watcher is on, so rb_pushes must
// only count REAL pushes -- otherwise a run with watcher off reports thousands of dumps
// while the ring buffer stays empty, and "no ring-buffer output" reads as a fabric
// problem instead of a missing TT_METAL_WATCHER=1.
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER) && !defined(FORCE_WATCHER_OFF)
constexpr bool kRingBufferLive = true;
#else
constexpr bool kRingBufferLive = false;
#endif

class FabricLinkCountersRbDumper {
public:
    void maybe_dump() {
        if constexpr (!kRingBufferLive) {
            return;  // L1 counters keep running; only the live view is unavailable.
        }
        auto* counters = fabric_link_counters();
        const uint32_t tx = counters->tx;
        const uint32_t rx = counters->rx;
        // Unsigned wrap makes these deltas correct even across a counter rollover.
        if ((tx - last_tx_) < kRbPushIntervalPackets && (rx - last_rx_) < kRbPushIntervalPackets) {
            return;
        }
        last_tx_ = tx;
        last_rx_ = rx;
        WATCHER_RING_BUFFER_PUSH(kRbTagTx | (tx & kRbValueMask));
        WATCHER_RING_BUFFER_PUSH(kRbTagRx | (rx & kRbValueMask));
        counters->rb_pushes++;
    }

private:
    uint32_t last_tx_ = 0;
    uint32_t last_rx_ = 0;
};

}  // namespace tt::tt_fabric::debug
