// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-bridged fabric link: layout shared by the router kernel and the host relay.
//
// A bridged router replays the two eth txq operations through the host instead of the wire. Each
// operation is one frame in a D2H socket ring on the router's core; the host forwards the ring to the
// peer host (MPI), which applies the frame to the far router's core: a PACKET frame is a DMA write of
// the payload to `addr`, a REG_WRITE frame is a register write of `value` to `addr`. Frames apply in
// order, so the far router sees exactly what the wire would have delivered.
#pragma once

#include <cstdint>

namespace tt::tt_fabric::host_bridge {

enum FrameKind : uint32_t { PACKET = 1, REG_WRITE = 2 };

// One per ring page, followed by the payload for PACKET frames.
struct FrameHeader {
    uint32_t kind;   // FrameKind
    uint32_t addr;   // L1 address (PACKET) or register address (REG_WRITE) on the far core
    uint32_t value;  // payload bytes (PACKET) or the register value (REG_WRITE)
    uint32_t seq;    // frames sent on this ring so far, for host-side sanity checks
};

// Payload starts here; 64 B keeps it aligned for the host DMA engine.
constexpr uint32_t kFrameHeaderBytes = 64;
constexpr uint32_t kFrameAlign = 64;

constexpr uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// Ring page: header + the largest packet the router can send.
constexpr uint32_t frame_page_bytes(uint32_t packet_capacity) {
    return align_up(kFrameHeaderBytes + packet_capacity, kFrameAlign);
}

// Control block in the router's L1, reserved above the channel buffers by the fabric builder.
//   [kStatusOffset, +64)          Status
//   [kTxSocketOffset, +128)       D2H socket config for the sender direction (+ 16 B header scratch at +64)
//   [kRxSocketOffset, +128)       D2H socket config for the receiver direction (+ header scratch)
constexpr uint32_t kBlockAlign = 64;
constexpr uint32_t kStatusOffset = 0;
constexpr uint32_t kSocketConfigBytes = 64;
constexpr uint32_t kSocketSlotBytes = 128;
constexpr uint32_t kTxSocketOffset = 64;
constexpr uint32_t kRxSocketOffset = kTxSocketOffset + kSocketSlotBytes;
constexpr uint32_t kBlockBytes = kRxSocketOffset + kSocketSlotBytes;
constexpr uint32_t header_scratch_offset(uint32_t socket_offset) { return socket_offset + kSocketConfigBytes; }

constexpr uint32_t kArmed = 0x41524D44;  // 'ARMD': host wrote both socket configs
constexpr uint32_t kMagic = 0x48425247;  // 'HBRG'

struct Status {
    uint32_t magic;       // kMagic once the router ran its init
    uint32_t armed;       // kArmed once the host wrote both socket configs; the router opens them lazily
    uint32_t packets;     // PACKET frames sent from this core
    uint32_t reg_writes;  // REG_WRITE frames sent from this core
    // Per direction: opened, times ensure_open() found the block unarmed, times the ring was full
    uint32_t tx_opened;
    uint32_t tx_unarmed;
    uint32_t tx_ring_full;
    uint32_t rx_opened;
    uint32_t rx_unarmed;
    uint32_t rx_ring_full;
};
static_assert(sizeof(Status) <= kTxSocketOffset, "Status must fit before the socket configs");

}  // namespace tt::tt_fabric::host_bridge
