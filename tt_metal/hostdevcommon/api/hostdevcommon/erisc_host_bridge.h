// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The ERISC host bridge wire frame: one slot is [fabric packet | BridgeDescriptor].
// Descriptor last so the guard lands after the bytes it validates. See ERISC_HOST_BRIDGE_CONTRACT.md.
#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::tt_fabric {

// 32 B: the floor, not a budget. The guard must be the last 8 bytes and its transfer must
// START 16 B aligned, so the descriptor is two 16 B writes and nothing smaller is expressible.
inline constexpr std::uint32_t kBridgeDescriptorBytes = 32;

inline constexpr std::uint64_t kBridgeMagic = 0x4548ull;  // 'EH'
inline constexpr std::uint32_t kBridgeVersion = 1;
inline constexpr std::uint32_t kBridgeGuardMagicShift = 16;

// Armed after the packet bytes, checked before the descriptor is trusted. [63:32] is the lap tag
// for a reused slot -- not the ordering key, which is ordering_cntr.
constexpr std::uint64_t bridge_guard(std::uint32_t version, std::uint32_t seq = 0) {
    return (static_cast<std::uint64_t>(seq) << 32) | (kBridgeMagic << kBridgeGuardMagicShift) |
           static_cast<std::uint64_t>(version);
}
// Masked, so a reader that ignores the sequence behaves as it did before there was one.
constexpr bool bridge_guard_armed(std::uint64_t guard) {
    return (guard & 0xFFFFFFFFull) == bridge_guard(kBridgeVersion);
}
constexpr std::uint32_t bridge_guard_seq(std::uint64_t guard) { return static_cast<std::uint32_t>(guard >> 32); }

// Written by the sending ERISC, read by both hosts, forwarded unrewritten. No address and no
// rank: each host owns its own map. The guard is the slot's last 8 bytes, written last.
struct BridgeDescriptor {
    // Opaque fabric bytes ahead of this descriptor. Duplicates the fabric header's own size by
    // choice, so the host never depends on that header's layout or build config.
    std::uint32_t length;
    // Ordering key, monotonic per sender and absolute. MPI may reorder or retry, so the RX sorts
    // on it and delivers an unbroken run. Also the credit input, as a difference against delivered.
    std::uint32_t ordering_cntr;
    // The TX ERISC's mirror of the far ring's write cursor, and the only dynamic placement
    // field left: the arena names the receiver channel, this names the slot within it.
    std::uint16_t slot_idx;
    // Alignment holes, not spare capacity: slot_idx is 2 B and elapsed must start at 16 so the
    // guard's transfer is aligned. Zeroed, so a later version cannot read a v1 sender's garbage.
    std::uint16_t reserved0;
    // Release time on the sender's clock, low 32 bits, modular. The host clock differs by an
    // unknown constant, so only the spread is real -- report the subtracted floor as the bias.
    std::uint32_t release_cyc;
    // Sender cycles: low half the packet write and barrier, high half the wait for a free slot.
    // Rides the tail transfer the guard already requires, so it costs nothing.
    std::uint64_t elapsed;
    // Armed after every field above and after the packet bytes. The host polls THIS word.
    std::uint64_t guard;
};
static_assert(sizeof(BridgeDescriptor) == kBridgeDescriptorBytes, "the descriptor must fill its slot");
static_assert(alignof(BridgeDescriptor) == 8, "descriptor alignment must not exceed the 8 B it needs");
// The whole point of the layout: nothing may follow the guard, or a tear could arm it early.
static_assert(
    offsetof(BridgeDescriptor, guard) + sizeof(std::uint64_t) == kBridgeDescriptorBytes,
    "guard must be the final word of the descriptor");
// The head/tail split is 16/16 and the guard sits in the tail. elapsed must therefore begin at
// exactly 16, or the guard is no longer the tail's last word.
static_assert(offsetof(BridgeDescriptor, elapsed) == 16, "elapsed must open the tail transfer");

// A slot is a D2H socket PAGE, and the socket refuses a page that is not PCIe aligned
// (d2h_socket.cpp: page_size % pcie_alignment == 0). 64 on Blackhole.
inline constexpr std::uint32_t kBridgeSlotAlign = 64;

// One slot holds a whole packet plus its descriptor, rounded up to that alignment. The padding
// sits BETWEEN payload and descriptor, so the guard stays the slot's final 8 bytes.
constexpr std::uint32_t bridge_slot_size(std::uint32_t packet_capacity_bytes) {
    return (packet_capacity_bytes + kBridgeDescriptorBytes + kBridgeSlotAlign - 1) / kBridgeSlotAlign *
           kBridgeSlotAlign;
}
// Alias for the socket-config call site, so the dependency reads as intent rather than arithmetic.
constexpr std::uint32_t bridge_socket_page_bytes(std::uint32_t packet_capacity_bytes) {
    return bridge_slot_size(packet_capacity_bytes);
}
// Measured from the END of the slot, not from capacity: the padding is ahead of the descriptor
// so the guard remains the last word the host can observe.
constexpr std::uint32_t bridge_desc_offset_in_slot(std::uint32_t packet_capacity_bytes) {
    return bridge_slot_size(packet_capacity_bytes) - kBridgeDescriptorBytes;
}
constexpr std::uint32_t bridge_guard_offset_in_slot(std::uint32_t packet_capacity_bytes) {
    return bridge_slot_size(packet_capacity_bytes) - static_cast<std::uint32_t>(sizeof(std::uint64_t));
}

// A peer's bytes name where this host stores, so the span is bounded before use. Fields are
// checked before any address arithmetic: a bounds check on an already-wrapped address proves nothing.
constexpr bool bridge_placement_ok(
    std::uint32_t recv_chan,
    std::uint32_t slot_idx,
    std::uint32_t length,
    std::uint32_t num_chans,
    std::uint32_t slots_per_chan,
    std::uint32_t buf_bytes) {
    return recv_chan < num_chans && slot_idx < slots_per_chan && length != 0 && length <= buf_bytes;
}

// Widened before multiplying: slot_idx * buf_bytes in 32 bits can wrap and land inside a
// valid channel with a plausible address. Call only after bridge_placement_ok().
constexpr std::uint64_t bridge_slot_offset(std::uint32_t slot_idx, std::uint32_t buf_bytes) {
    return static_cast<std::uint64_t>(slot_idx) * static_cast<std::uint64_t>(buf_bytes);
}

// The EDM idiom: unsigned difference of two running totals, wrap-safe while the gap stays under
// 2^32. Never compare with >= -- that works for an hour after a wrap, then fails once.
constexpr std::uint32_t bridge_unprocessed(std::uint32_t sent, std::uint32_t processed) { return sent - processed; }

// Device-cycle instrumentation, packed as the D2H leg packs it: low half the packet write,
// high half the stall ahead of it. Saturating, so a stuck sender cannot alias a small value.
inline constexpr std::uint64_t kBridgeElapsedMask = 0xFFFFFFFFull;
constexpr std::uint64_t bridge_elapsed_pack(std::uint64_t issue, std::uint64_t stall) {
    return ((stall > kBridgeElapsedMask ? kBridgeElapsedMask : stall) << 32) |
           (issue > kBridgeElapsedMask ? kBridgeElapsedMask : issue);
}
constexpr std::uint32_t bridge_elapsed_issue(std::uint64_t packed) {
    return static_cast<std::uint32_t>(packed & kBridgeElapsedMask);
}
constexpr std::uint32_t bridge_elapsed_stall(std::uint64_t packed) { return static_cast<std::uint32_t>(packed >> 32); }

}  // namespace tt::tt_fabric
