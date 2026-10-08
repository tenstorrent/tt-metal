// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Layout of the ERISC host bridge region: one arena per (LINK, RECEIVER CHANNEL) -- see below.
// Shared host/kernel, so the sending ERISC and both hosts agree without a runtime exchange.
#pragma once

#include <cstdint>

#include "hostdevcommon/erisc_host_bridge.h"

namespace tt::tt_fabric {

// Which half of a peer's span. TX is what this host's ERISC fills and this host puts from;
// RX is what the peer puts into and this host drains to its own ERISC.
enum class BridgeArena : std::uint32_t { Tx = 0, Rx = 1, Count = 2 };
inline constexpr std::uint32_t kBridgeArenas = static_cast<std::uint32_t>(BridgeArena::Count);

// One arena per (link, receiver channel): peer rank belongs to a link, and the arena names the
// buffer. Per channel, not per VC, so ordering matches the only guarantee fabric makes.

// Flattens the two dimensions. The host's link table and its receiver-channel table are both
// resolved once at setup, so both sides compute the same index without an exchange.
constexpr std::uint32_t bridge_arena_index(
    std::uint32_t link_idx, std::uint32_t recv_chan, std::uint32_t chans_per_link) {
    return link_idx * chans_per_link + recv_chan;
}
constexpr std::uint32_t bridge_arena_count(std::uint32_t links, std::uint32_t chans_per_link) {
    return links * chans_per_link;
}

// Ring depth in slots. A tunable with a default, like the H2H flush watermark: the D2H2H2D
// sweeps put the useful knee near 32, past which depth buys rate and costs latency linearly.
inline constexpr std::uint32_t kBridgeDefaultRingPages = 32;

// Segments need page alignment for MAP_FIXED, not the region's 2 MiB: D2H2H2D arenas are
// 1536 KiB and are overlaid fine. Huge pages are a property of the region, not each segment.
inline constexpr std::uint64_t kBridgeSegmentAlign = 4096;
inline constexpr std::uint64_t kBridgeRegionAlign = 2ull << 20;

constexpr std::uint64_t bridge_align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) & ~(a - 1); }

// One direction of one link: ring_pages slots, each [packet | descriptor].
constexpr std::uint64_t bridge_segment_bytes(std::uint32_t ring_pages, std::uint32_t packet_capacity) {
    return bridge_align_up(
        static_cast<std::uint64_t>(ring_pages) * bridge_slot_size(packet_capacity), kBridgeSegmentAlign);
}
// TX the D2H socket FIFO the router writes, RX the H2D ring it reads.
constexpr std::uint64_t bridge_arena_bytes(std::uint32_t ring_pages, std::uint32_t packet_capacity) {
    return kBridgeArenas * bridge_segment_bytes(ring_pages, packet_capacity);
}

// ---- control block: bridge-owned, never aliased ----------------------------------------
// The arenas are alias targets, so nothing of the bridge's own may live inside them.

constexpr std::uint64_t bridge_desc_array_bytes(std::uint32_t ring_pages) {
    return static_cast<std::uint64_t>(ring_pages) * kBridgeDescriptorBytes;
}

// Relayed EDM ack/completion words plus this bridge's own posted/seen pair, per link.
inline constexpr std::uint32_t kBridgeCreditWords = 8;

// Two credit systems: BridgeSeen/BridgePosted is hop-local ring-slot credit; EdmCompletions/
// EdmAcks is fabric's, relayed because no cable carries it. Absolute totals, never deltas.
enum class BridgeCreditWord : std::uint32_t {
    BridgeSeen = 0,      // RX host -> TX host: slots the RX has finished with
    BridgePosted = 1,    // TX host -> RX host: slots the TX has put
    EdmCompletions = 2,  // RX host -> TX host: completions the FAR RECEIVER produced
    EdmAcks = 3,         // RX host -> TX host: acks the far receiver produced
};

constexpr std::uint64_t bridge_credit_word_offset(
    std::uint32_t arena_idx, std::uint32_t ring_pages, BridgeCreditWord w);
inline constexpr std::uint64_t kBridgeCreditBytes = kBridgeCreditWords * sizeof(std::uint64_t);

constexpr std::uint64_t bridge_link_control_bytes(std::uint32_t ring_pages) {
    return kBridgeArenas * bridge_desc_array_bytes(ring_pages) + kBridgeCreditBytes;
}

// Rounded to the region alignment so the first segment starts aligned.
constexpr std::uint64_t bridge_control_bytes(std::uint32_t arenas, std::uint32_t ring_pages) {
    return bridge_align_up(
        static_cast<std::uint64_t>(arenas) * bridge_link_control_bytes(ring_pages), kBridgeRegionAlign);
}

// Takes an ARENA index (bridge_arena_index), not a link index -- control and arena are
// per-arena so the two stay in step.
constexpr std::uint64_t bridge_link_control_offset(std::uint32_t arena_idx, std::uint32_t ring_pages) {
    return static_cast<std::uint64_t>(arena_idx) * bridge_link_control_bytes(ring_pages);
}

// Descriptor `slot_idx` for one direction of one link. What a gathered put targets.
constexpr std::uint64_t bridge_desc_offset(
    std::uint32_t arena_idx, BridgeArena arena, std::uint32_t slot_idx, std::uint32_t ring_pages) {
    return bridge_link_control_offset(arena_idx, ring_pages) +
           static_cast<std::uint64_t>(arena) * bridge_desc_array_bytes(ring_pages) +
           static_cast<std::uint64_t>(slot_idx) * kBridgeDescriptorBytes;
}

constexpr std::uint64_t bridge_credit_offset(std::uint32_t arena_idx, std::uint32_t ring_pages) {
    return bridge_link_control_offset(arena_idx, ring_pages) + kBridgeArenas * bridge_desc_array_bytes(ring_pages);
}

// Word 0 is bridge_credit_offset itself, so existing callers keep working unchanged.
constexpr std::uint64_t bridge_credit_word_offset(
    std::uint32_t arena_idx, std::uint32_t ring_pages, BridgeCreditWord w) {
    return bridge_credit_offset(arena_idx, ring_pages) + static_cast<std::uint64_t>(w) * sizeof(std::uint64_t);
}

// ---- arenas: alias targets ---------------------------------------------------------------

constexpr std::uint64_t bridge_segment_offset(
    std::uint32_t arena_idx,
    BridgeArena arena,
    std::uint32_t arenas,
    std::uint32_t ring_pages,
    std::uint32_t packet_capacity) {
    return bridge_control_bytes(arenas, ring_pages) +
           static_cast<std::uint64_t>(arena_idx) * bridge_arena_bytes(ring_pages, packet_capacity) +
           static_cast<std::uint64_t>(arena) * bridge_segment_bytes(ring_pages, packet_capacity);
}

// Slot `slot_idx` inside a segment: [packet | descriptor], the socket's own page layout.
constexpr std::uint64_t bridge_slot_offset_in_segment(std::uint32_t slot_idx, std::uint32_t packet_capacity) {
    return static_cast<std::uint64_t>(slot_idx) * bridge_slot_size(packet_capacity);
}

// Takes the ARENA count: bridge_arena_count(links, chans_per_link).
constexpr std::uint64_t bridge_region_bytes(
    std::uint32_t arenas, std::uint32_t ring_pages, std::uint32_t packet_capacity) {
    return bridge_control_bytes(arenas, ring_pages) +
           static_cast<std::uint64_t>(arenas) * bridge_arena_bytes(ring_pages, packet_capacity);
}

// Every offset a peer names must land inside the segment it claims, so a wrong index cannot
// reach another arena. Checked before any write, like bridge_placement_ok().
constexpr bool bridge_segment_offset_ok(
    std::uint64_t offset,
    std::uint64_t length,
    std::uint32_t arena_idx,
    BridgeArena arena,
    std::uint32_t arenas,
    std::uint32_t ring_pages,
    std::uint32_t packet_capacity) {
    const std::uint64_t lo = bridge_segment_offset(arena_idx, arena, arenas, ring_pages, packet_capacity);
    const std::uint64_t seg = bridge_segment_bytes(ring_pages, packet_capacity);
    return offset >= lo && length <= seg && offset + length <= lo + seg;
}

}  // namespace tt::tt_fabric
