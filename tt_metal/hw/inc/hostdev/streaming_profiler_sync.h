// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host/device shared layouts of the streaming profiler's clock sync: the sync records that the clock pusher, the link
// ends and the drainer write and the host's sync engine decodes, the device-to-device link sync's L1, the control
// block of the pusher and the drainer, and the tile clock network's scratch.

#include <cstddef>
#include <cstdint>

namespace kernel_profiler {

template <typename T>
constexpr std::uint32_t word_of(const T& v) {
    static_assert(sizeof(T) == sizeof(std::uint32_t));
    return __builtin_bit_cast(std::uint32_t, v);
}
template <typename T>
constexpr T word_as(std::uint32_t w) {
    static_assert(sizeof(T) == sizeof(std::uint32_t));
    return __builtin_bit_cast(T, w);
}

constexpr std::uint32_t kEthRefclkHz = 50'000'000u;  // the Ethernet tile's reference clock

// ---- Sync records -------------------------------------------------------------------------------------------------
// Eight words, shipped on the drainer's sync socket as sync frames: the SPSC frame prefix with the record count at
// SPSC_PREFIX_HEAD_0, then the records, padded to SPSC_SPAN_WIRE_CTRL_WORDS so the ingest's frame walk accepts the
// frame. The sync engine reads that socket itself; these are never profiler records. By kind:
//   LOCAL (the pusher): up to kSyncLocalPoints points of the chip's clock model. value and wall hold the first point's
//     refclk and wall (in eighths of a tick), ref the later points as SyncLocalSteps.
//   LINK (the link ends): value is a round's 1588 stamp average in kLinkSyncStampUnitsPerNs per ns.
//   ANCHOR (the drainer): its (wall, refclk) pair at a refclk update, in wall and ref.
//   ANCHOR_HIST (the drainer, at stop): its audit of its anchors against the pusher's points. Bin records hold the
//     first bin in round and six counts in the words from value_lo on; then one record with round
//     kSyncAnchorHistWorst holds the worst anchor's refclk in value, the anchors whose read found no refclk update in
//     wall, and the worst error in 1/16 ns, signed, in ref[0].
struct SyncRecord {
    std::uint32_t meta;   // SyncMeta; SyncLocalMeta in a LOCAL record
    std::uint32_t round;  // SyncLocalRound in a LOCAL record
    std::uint32_t value_lo, value_hi;
    std::uint32_t wall_lo, wall_hi;
    std::uint32_t ref[2];  // low word first
};
constexpr std::uint32_t kSyncRecordWords = sizeof(SyncRecord) / sizeof(std::uint32_t);

constexpr std::uint32_t kSyncKindLocal = 0, kSyncKindLink = 1, kSyncKindAnchor = 2, kSyncKindAnchorHist = 3;
// A LINK record's role: the receiver records the sender's egress average (T0) and its own ingress average (T1), the
// sender the receiver's egress average (T1B) and its own ingress average (T2).
constexpr std::uint32_t kSyncRoleT0 = 0, kSyncRoleT1 = 1, kSyncRoleT1B = 2, kSyncRoleT2 = 3;

struct SyncMeta {
    std::uint32_t role : 8;
    std::uint32_t kind : 8;
    std::uint32_t rsvd : 16;
};

constexpr std::uint32_t kSyncLocalPoints = 3;
struct SyncLocalMeta {
    std::uint32_t count : 2;
    std::uint32_t close : 3;  // bit i: point i closed a sample window
    std::uint32_t rsvd0 : 3;
    std::uint32_t kind : 8;
    std::uint32_t rsvd1 : 16;
};
struct SyncLocalRound {
    std::uint8_t k8[kSyncLocalPoints];  // each point's wall ticks per refclk tick, in eighths; 0 for a single sample
    std::uint8_t slope;                 // the k8 the later points' wall offsets are taken against
};
// A later point of a LOCAL record, relative to the first.
struct SyncLocalStep {
    std::uint32_t refclk : 16;   // refclk ticks past the first point
    std::int32_t wall_off : 16;  // eighths of a tick, from the first point's wall plus slope times the refclk step
};
static_assert(sizeof(SyncLocalRound) == sizeof(std::uint32_t) && sizeof(SyncLocalStep) == sizeof(std::uint32_t));

struct SyncLocalPoint {
    std::uint64_t r, w8;
    std::uint32_t k8;
    bool close;
};
// The points of a LOCAL record, in order; returns how many. `Record` is SyncRecord, possibly volatile.
template <typename Record>
inline std::uint32_t sync_local_unpack(const Record& rec, SyncLocalPoint* out) {
    const auto meta = word_as<SyncLocalMeta>(rec.meta);
    // Read in place: a bit_cast into a struct holding an array goes through the stack.
    const volatile SyncLocalRound& round = reinterpret_cast<const volatile SyncLocalRound&>(rec.round);
    const std::uint32_t slope = round.slope;
    const std::uint64_t r0 = (static_cast<std::uint64_t>(rec.value_hi) << 32) | rec.value_lo;
    const std::uint64_t w0 = (static_cast<std::uint64_t>(rec.wall_hi) << 32) | rec.wall_lo;
    for (std::uint32_t i = 0; i < meta.count; i++) {
        std::uint64_t r = r0, w = w0;
        if (i != 0) {
            const auto s = word_as<SyncLocalStep>(rec.ref[i - 1]);
            r += s.refclk;
            w += static_cast<std::uint64_t>(slope) * s.refclk +
                 static_cast<std::uint64_t>(static_cast<std::int64_t>(s.wall_off));
        }
        out[i] = SyncLocalPoint{r, w, round.k8[i], ((meta.close >> i) & 1u) != 0};
    }
    return meta.count;
}

constexpr std::uint32_t kSyncAnchorHistBins = 512;  // 1/16 ns each over +-16 ns
constexpr std::uint32_t kSyncAnchorHistWorst = 0xFFFFFFFFu;
// The pusher's ring: a firmware FBDIV walk sends a few records per ~1.25 us step.
constexpr std::uint32_t kSyncRingRecords = 512;
constexpr std::uint32_t kSyncRingBytes = kSyncRingRecords * sizeof(SyncRecord);
constexpr std::uint32_t kSyncFrameRecords = 32;  // records per sync frame at most

// ---- Device-to-device link sync ------------------------------------------------------------------------------------
// The contract between the host, the link sync's resident kernels and the fabric routers (eth_ptp_link.hpp).
constexpr std::uint32_t kLinkSyncStampUnitsPerNs = 64;
constexpr std::uint32_t kLinkSyncPaceTicks = 500'000;  // a round every 10 ms
constexpr std::uint32_t kLinkSyncCtlRun = 1, kLinkSyncCtlStop = 2;
constexpr std::uint32_t kLinkSyncTimerRan = 1, kLinkSyncTimerNoRate = 2;
constexpr std::uint32_t kLinkSyncSlotWords = 96;  // eth_ptp_link.hpp's frames in flight
constexpr std::uint32_t kLinkSyncRingRecords = 8;
static_assert(kLinkSyncRingRecords <= kSyncFrameRecords);

// What each end leaves for the host, rewritten at every round's close.
struct LinkSyncDiag {
    std::uint32_t timer;  // kLinkSyncTimerRan, or kLinkSyncTimerNoRate: this end sent no stamps
    std::uint32_t rounds_lost;
    std::uint32_t bursts_mismatched;  // bursts whose ingress stamps did not match their frames
    std::uint32_t frames_unstamped;   // frames that came in without an egress stamp
};

// The L1 both ends of a link own at the top of the active eth core's unreserved region, at the same address on both.
struct LinkSyncL1 {
    std::uint32_t slots[kLinkSyncSlotWords];
    std::uint32_t ctl;   // host-written: kLinkSyncCtlRun once the profiler is up, kLinkSyncCtlStop to stop
    std::uint32_t done;  // set by a resident end once it has stopped
    LinkSyncDiag diag;
    // A round's two stamp averages, their count published in the core's SPSC_LINK_SYNC_TAIL; an end never waits for a
    // reader, so a drainer a whole ring behind loses the oldest.
    alignas(32) SyncRecord ring[kLinkSyncRingRecords];
};
constexpr std::uint32_t kLinkSyncL1Bytes = sizeof(LinkSyncL1);

// ---- Pusher and drainer control block ------------------------------------------------------------------------------
// At the control address of the clock pusher and the eth drainer; the stop word opens the next kRelayCtrlWordStride
// block.
struct SyncCoreCtrl {
    std::uint32_t done;
    std::uint32_t heartbeat;
    std::uint32_t go;            // host-written once the receiver's ingest threads are up
    std::uint32_t sync_tail;     // pusher: records written to its sync ring
    std::uint32_t sync_head;     // pusher: records the drainer has taken, which the drainer writes back
    std::uint32_t dropped_pll;   // pusher, as it exits: samples dropped waiting on a PLL read
    std::uint32_t dropped_sync;  // pusher, as it exits: clock instants its sync ring had no room for
};

// ---- Tile clock network --------------------------------------------------------------------------------------------
constexpr std::uint32_t kTileNetMaxPartners = 40;
constexpr std::uint32_t kTileNetBins = 128;
constexpr std::uint32_t kTileNetGoMeasure = 1, kTileNetGoExit = 2;

struct TileNetPartner {
    std::int32_t median2;  // the median of 2 * (partner wall - bracket midpoint), in the clocks' low words
    std::int32_t spread2;  // the spread between that median's quartiles
    std::int32_t rtt;      // the median round trip, in ticks
    std::uint32_t coarse_lo, coarse_hi;  // the whole-clock difference, partner minus this tile
};
struct TileNetTable {
    std::uint32_t go;     // host-written: kTileNetGoMeasure when this tile's turn comes, kTileNetGoExit to release it
    std::uint32_t ready;  // the host's nonce once the tile is up, its inverse once every partner is written
    TileNetPartner partner[kTileNetMaxPartners];
};
// Each tile's scratch, which tile_sync.cpp fills and the host reads back.
struct TileNetScratch {
    std::uint32_t landing[16];  // where the tile's reads of its partners land
    TileNetTable table;
    std::uint32_t hist[2 * kTileNetBins];  // the counts of the median's samples, then of the round trips
};

}  // namespace kernel_profiler
