// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What the harness and its node kernel (kernels/vc_node.cpp) share (SPEC-0012 § The node): the runtime-argument
// order, the region layout, the packet format, the status and halt codes. Included by the host and the device; plain
// C++17 with no dependency beyond <cstdint>.

#pragma once

#include <cstdint>

namespace vc_node {

// Runtime-argument words, in order. Then one word a device from kArgNextPortBase (next_port[d]), then
// kArgsPerPort words a port from kArgNextPortBase + num_devices.
enum Arg : uint32_t {
    kArgDevice = 0,          // this device's fabric_chip, below num_devices
    kArgNumDevices = 1,      // N
    kArgRegionBase = 2,      // flat offset of the region in this CCE's SRAM, a multiple of kAlign
    kArgNumPorts = 3,        // k, at most kMaxPorts
    kArgPacketsPerFlow = 4,  // packets this device originates to each other device
    kArgPayloadBytes = 5,    // a multiple of kPayloadGranule
    kArgRingSlots = 6,       // slots of each port's ring, at least 1
    kArgNonce = 7,           // carried by every packet of the run
    kArgNextPortBase = 8,
};
// A port's words, after the next_port words.
enum PortArg : uint32_t {
    kPortArgKind = 0,          // PortKind
    kPortArgPeerTile = 1,      // the peer's HSIO tile
    kPortArgPeerPortBase = 2,  // flat offset, in the peer's CCE SRAM, of the peer's port block for this link
};
constexpr uint32_t kArgsPerPort = 3;
// next_port[d] for d this device.
constexpr uint32_t kNoPort = 0xFFFFFFFF;

enum PortKind : uint32_t {
    kPortKindNoc = 1,    // the peer is on another tile
    kPortKindLocal = 2,  // the peer is on this CCE
};

constexpr uint32_t kMaxPorts = 4;

// Runtime-argument words of one node.
constexpr uint32_t num_args(uint32_t num_devices, uint32_t num_ports) {
    return kArgNextPortBase + num_devices + kArgsPerPort * num_ports;
}

// Every block of the region starts on a multiple of this, from region_base.
constexpr uint32_t kAlign = 64;
constexpr uint32_t kPayloadGranule = 16;

// Control block: 32-bit words at region_base + 4 * word. Go and Stop are written by the host; the rest by the node.
enum ControlWord : uint32_t {
    kCtlStatus = 0,      // Status
    kCtlPostcode = 1,    // Halt, once status is kStatusHalted
    kCtlDetail = 2,      // what the Halt names
    kCtlGo = 3,          // nonzero: run
    kCtlStop = 4,        // nonzero: stop
    kCtlOriginated = 5,  // packets originated
    kCtlDelivered = 6,   // packets for this device received and checked good
    kCtlBad = 7,         // packets for this device received and checked bad
};
constexpr uint32_t kControlBytes = 64;
// Staging block: one 8-byte word a port, the source of a noc count or credit copy.
constexpr uint32_t kStagingBytes = 64;

enum Status : uint32_t {
    kStatusNotStarted = 0,
    kStatusWaitingForGo = 0xA0,
    kStatusRunning = 0xA1,
    kStatusDone = 0xA2,
    kStatusHalted = 0xAF,
};

// The postcode of a halt, and what its detail word holds.
enum Halt : uint32_t {
    kHaltBadArg = 0xCCEEF201,        // detail: the runtime-argument word's index
    kHaltNoRoute = 0xCCEEF202,       // detail: the packet's dst
    kHaltBadPacket = 0xCCEEF203,     // detail: the packet's seq; nonce, length or payload wrong
    kHaltNocPeerOnOwnTile = 0xCCEEF204,  // detail: the port
};

// Port block header: 64-bit words at the block's start. Arrived and ConsumedByPeer are written by the peer, Sent and
// Received by this node. Each is a free-running total.
enum PortHeaderOffset : uint32_t {
    kPortArrived = 0,          // packets the peer has put in this port's ring
    kPortConsumedByPeer = 8,   // packets of this port's sends the peer has taken from its ring
    kPortSent = 16,            // packets this node has sent on this port
    kPortReceived = 24,        // packets this node has taken from this port's ring
};
constexpr uint32_t kPortHeaderBytes = 64;

// Packet header, at the start of a slot; the payload follows it.
struct PacketHeader {
    uint16_t dst;
    uint16_t src;
    uint32_t seq;
    uint32_t nonce;
    uint32_t payload_bytes;
};
static_assert(sizeof(PacketHeader) == 16, "SPEC-0012 § Region layout: a 16-byte packet header");

// Byte i of the payload of packet seq from src to dst.
constexpr uint8_t payload_byte(uint32_t src, uint32_t dst, uint32_t seq, uint32_t i) {
    return static_cast<uint8_t>((src * 31 + dst * 17 + seq * 7 + i) & 0xFF);
}

constexpr uint64_t align_up(uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

// Byte offsets from region_base, each a multiple of kAlign. 64-bit so that the host can size any argument values
// without overflow; the node's values fit 32 bits once H8 has passed.
struct Layout {
    uint64_t slot_bytes;        // a packet header and its payload, rounded up to kAlign
    uint64_t port_block_bytes;  // header and ring of one port
    uint64_t control;
    uint64_t staging;
    uint64_t port_blocks;  // port p's block at port_blocks + p * port_block_bytes
    uint64_t source;       // the slot an originated packet is built in
    uint64_t total;        // bytes of the region the node uses

    constexpr uint64_t staging_word(uint32_t port) const { return staging + port * sizeof(uint64_t); }
    constexpr uint64_t port_block(uint32_t port) const { return port_blocks + port * port_block_bytes; }
    constexpr uint64_t ring_slot(uint32_t port, uint64_t slot) const {
        return port_block(port) + kPortHeaderBytes + slot * slot_bytes;
    }
};

// The region layout of a node with these arguments. Exact while ring_slots * slot_bytes fits 64 bits; H8 bounds it by
// the region before calling this.
constexpr Layout layout(uint32_t num_ports, uint32_t ring_slots, uint32_t payload_bytes) {
    Layout l{};
    l.slot_bytes = align_up(sizeof(PacketHeader) + static_cast<uint64_t>(payload_bytes));
    l.port_block_bytes = kPortHeaderBytes + static_cast<uint64_t>(ring_slots) * l.slot_bytes;
    l.control = 0;
    l.staging = l.control + kControlBytes;
    l.port_blocks = l.staging + kStagingBytes;
    l.source = l.port_blocks + num_ports * l.port_block_bytes;
    l.total = l.source + l.slot_bytes;
    return l;
}

static_assert(kStagingBytes >= kMaxPorts * sizeof(uint64_t));
// SPEC-0012 § The mesh, and its numbers.
static_assert(layout(2, 60, 256).slot_bytes == 320);
static_assert(layout(2, 60, 256).port_block_bytes == 19264);
static_assert(layout(2, 60, 256).total == 38976);

}  // namespace vc_node
