// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared control-plane layout for the IN-ROUTER device-to-device clock-sync hook
// (FABRIC_ROUTER_SYNC_HOOK): the L1 config/message block the host writes and the two routers
// exchange 16 B eth messages through, and the discovery words the hook publishes in the AERISC
// fabric scratch area.
//
//   Device side: tt_metal/fabric/hw/inc/edm_fabric/fabric_router_sync_hook.hpp (the hook).
//   Host side:   tools/profiler/perf_debug_profiler.cpp (discovery + config write).
//
// The SAMPLES do not live here. t0/t1/t2 are self-timestamping PP_SYNC packets on the streaming
// profiler wire (tools/profiler/spsc_packet.h; field packing in hostdevcommon/profiler_common.h):
// the hook emits a marker AT the instant, the marker's own timestamp IS the sample, and the DRISC
// fillers sweep it off the eth core like any other marker. This header is only the handshake
// state -- who initiates, how often, where the peer's message slots are.
#pragma once

#include <stdint.h>

namespace tt::tt_fabric::router_sync {

// ---- discovery: the hook publishes {magic, blk addr, status} into the fabric scratch area ------
// MEM_AERISC_FABRIC_SCRATCH_BASE is 28 B and nothing in the tree writes it today (only the
// address-map aliases reference it). Word offsets within that area:
constexpr uint32_t kScratchMagicWord = 0;  // kDiscMagic, written LAST by the hook at kernel start
constexpr uint32_t kScratchBlkWord = 1;    // L1 address of the hook's Blk (kernel .bss)
constexpr uint32_t kScratchStatWord = 2;   // (completed rounds << 8) | (failed rounds & 0xFF), live
constexpr uint32_t kScratchDbgWord = 3;    // (poll state << 28) | deadline-expiry count, live
constexpr uint32_t kDiscMagic = 0xFA5CD15Cu;

// ---- host-written config ------------------------------------------------------------------------
constexpr uint32_t kCfgMagic = 0xFA5CCF61u;  // host writes this word LAST; anything else = unconfigured
constexpr uint32_t kFlagInitiator = 1u << 0;
constexpr uint32_t kFlagEnabled = 1u << 1;
// Responder stamps TWICE per sample (t1 at doorbell detection, t1b immediately before the echo
// send) so the host can subtract the measured turnaround per sample instead of carrying it as a
// bias. Host-set; the responder reads it from cfg per round, so both A/B arms come from one build.
constexpr uint32_t kFlagTwoStamp = 1u << 2;
constexpr uint32_t kMaxSamples = 16;  // triples per round; bounds the ring-room reservation

struct Cfg {
    uint32_t magic;        // kCfgMagic (host writes it LAST, after every other field)
    uint32_t flags;        // kFlagInitiator | kFlagEnabled
    uint32_t interval_lo;  // initiator: round period, cycles. responder: doorbell poll period.
    uint32_t interval_hi;  //   0 = disabled even when compiled in; host-rewritable live.
    uint32_t n_samples;    // triples per round, 1..kMaxSamples
    uint32_t first_wait;   // cycles: sample-0 echo/doorbell wait (covers peer's worst notice latency)
    uint32_t next_wait;    // cycles: samples 1.. and txq-drain waits
    uint32_t peer_blk;     // peer router's Blk L1 address (eth-write target base)
};
static_assert(sizeof(Cfg) == 32, "Cfg layout is shared with the hook's eth-write offsets");

// One 16 B eth message. The TAG is the LAST word: eth delivers ascending, so a poller that sees
// the tag transition sees a complete message.
struct Msg {
    uint32_t pad[3];
    uint32_t tag;
};
static_assert(sizeof(Msg) == 16, "Msg must match the 16 B eth packet");

// Poll-cadence self-measurement, published by the hook once per deadline expiry. The gap between
// prescaler expiries IS the responder's doorbell-notice granularity (the fast-doorbell path runs
// once per expiry), so this measures directly what the sample-0 wait distribution only implies:
// on the chip-3 links that distribution fits a ~330 us notice period against ~33 us on healthy
// links, and this decides "the responder polls that rarely" vs "the ping itself arrives late".
// min/max are saturating 32-bit cycle counts; mean is derived host-side as (last-first)/cnt.
struct GapStats {
    uint32_t min_cy;    // smallest expiry-to-expiry gap seen (0 until 2 expiries happen)
    uint32_t max_cy;    // largest
    uint32_t cnt;       // gaps accumulated (expiries - 1)
    uint32_t first_lo;  // wall clock at the first expiry, 64-bit split
    uint32_t first_hi;
    uint32_t last_lo;  // wall clock at the most recent PUBLISHED expiry
    uint32_t last_hi;
    uint32_t pad;  // keep the block 16 B-granular like everything around it
};

// The hook's L1 block (kernel .bss on the device; address published via kScratchBlkWord).
struct Blk {
    Cfg cfg;       // +0   host-written
    Msg ping;      // +32  initiator -> responder (sample i request; ping 0 is the round doorbell)
    Msg echo;      // +48  responder -> initiator (sample i acknowledgement)
    Msg tx;        // +64  local staging for outbound eth sends (never a remote-write target)
    GapStats gap;  // +80  hook-published poll-cadence stats, host-read at teardown
};
constexpr uint32_t kPingOff = 32;
constexpr uint32_t kEchoOff = 48;
constexpr uint32_t kTxOff = 64;
constexpr uint32_t kGapOff = 80;
static_assert(sizeof(Blk) == 112, "Blk layout is shared with the host config writer AND the EDM builder's carve");

// ---- message tags -------------------------------------------------------------------------------
// [31:25] kind, [24:8] round (17 bits, same width as PP_SYNC's round field), [7:0] sample idx.
// Zero (freshly zeroed slot) matches no kind.
constexpr uint32_t kTagPing = 0x2Au;
constexpr uint32_t kTagEcho = 0x15u;
constexpr uint32_t tag(uint32_t kind, uint32_t round, uint32_t idx) {
    return (kind << 25) | ((round & 0x1FFFFu) << 8) | (idx & 0xFFu);
}
constexpr uint32_t tag_kind(uint32_t t) { return t >> 25; }
constexpr uint32_t tag_round(uint32_t t) { return (t >> 8) & 0x1FFFFu; }
constexpr uint32_t tag_idx(uint32_t t) { return t & 0xFFu; }

}  // namespace tt::tt_fabric::router_sync
