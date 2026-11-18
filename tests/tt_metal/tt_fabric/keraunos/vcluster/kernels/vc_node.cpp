// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The harness node of SPEC-0012 § The node: one virtual device on one hart of a Keraunos CCE0, an idle Ethernet
// kernel. Routes, originates and receives packets on up to vc_node::kMaxPorts links; its runtime arguments, region
// layout and codes are in vc_node_layout.hpp. Keeps its state in its region, not in kernel data.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "internal/hw_thread.h"
#include "internal/tt-2xx/quasar/eth_cce_send.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "tests/tt_metal/tt_fabric/keraunos/vcluster/vc_node_layout.hpp"

using namespace vc_node;
namespace cce_eth = internal_::cce_eth;

namespace {

constexpr overlay::CmdBuf kCmdBuf = overlay::CMDBUF_1;
// A node's one send in flight: request VC 4 + hart, transaction id 8 on every hart (SPEC-0012 § A send on a link).
// Each hart has its own command buffers and acknowledgement counts.
constexpr uint32_t kFirstReqVc = 4;
constexpr uint32_t kTrid = overlay::CMDBUF_1_TRID_START;
static_assert(kTrid == 8 && kTrid <= overlay::CMDBUF_1_TRID_END, "SPEC-0012: id 8, inside command buffer 1's range");
// What cce_noc::copy() programs; no document assigns it.
constexpr uint32_t kRespVc = overlay::CMDBUF_WR_RESP_VC;

struct Port {
    uint32_t kind;
    uint32_t peer_tile;
    uint32_t peer_port_base;  // flat, in the peer's CCE SRAM
    uint32_t block;           // flat, this node's block for the port
};

struct Node {
    uint32_t device;
    uint32_t num_devices;
    uint32_t num_ports;
    uint32_t packets_per_flow;
    uint32_t payload_bytes;
    uint32_t ring_slots;
    uint32_t nonce;
    uint32_t own_tile;
    uint32_t req_vc;
    uint32_t slot_bytes;
    uint32_t control;
    uint32_t staging;
    uint32_t source;
    Port ports[kMaxPorts];
};

// A word another hart, the iDMA or the host touches: through the uncached alias.
volatile uint32_t* word32(uint32_t flat) {
    return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(MEM_IERISC_UNCACHED_BASE + flat));
}
volatile uint64_t* word64(uint32_t flat) {
    return reinterpret_cast<volatile uint64_t*>(static_cast<uintptr_t>(MEM_IERISC_UNCACHED_BASE + flat));
}
volatile uint32_t* ctl(const Node& n, ControlWord w) { return word32(n.control + w * sizeof(uint32_t)); }

// A CPU store retires when it leaves the core complex, not when it reaches SRAM, which the iDMA reads
// (CCE-DM-PRIMITIVES § Fence before a send).
void fence() { asm volatile("fence" ::: "memory"); }

// Records the halt, then waits for stop so the program ends. Returns false for the caller to pass up.
bool halt(const Node& n, Halt code, uint32_t detail) {
    *ctl(n, kCtlDetail) = detail;
    *ctl(n, kCtlPostcode) = code;
    *ctl(n, kCtlStatus) = kStatusHalted;
    while (*ctl(n, kCtlStop) == 0) {
    }
    return false;
}

// The address the iDMA writes for the flat offset @p flat at the far end of @p port.
uint64_t peer_addr(const Node& n, const Port& port, uint32_t flat) {
    return port.kind == kPortKindLocal ? cce_eth::sram_local_addr(n.own_tile, flat)
                                       : cce_eth::sram_system_addr(port.peer_tile, flat);
}

// A coupled iDMA copy of @p bytes from this CCE's flat @p src to @p dest, awaited. Programs every register it uses:
// command-buffer fields persist across issues, and a node alternates local and noc copies on one buffer.
void copy(const Node& n, uint32_t src, uint64_t dest, uint32_t bytes) {
    using namespace overlay;
    idma_setup_as_copy_cmdbuf<kCmdBuf>();
    setup_ongoing_cmdbuf<kCmdBuf>(AutoIncConfig{});
    setup_wrapping_vcs_cmdbuf<kCmdBuf>(
        VcRange{.start = n.req_vc, .end = n.req_vc}, VcRange{.start = kRespVc, .end = kRespVc});
    setup_trids_cmdbuf<kCmdBuf>(kTrid);
    set_dest_cmdbuf<kCmdBuf>(dest, 0, 0, 0);
    set_src_cmdbuf<kCmdBuf>(cce_eth::sram_local_addr(n.own_tile, src), 0, 0, 0);
    set_len_cmdbuf<kCmdBuf>(bytes);
    setup_max_bytes_in_packet_cmdbuf<kCmdBuf>(cce_eth::kMaxBytesInPacket);
    setup_packet_tags_cmdbuf<kCmdBuf>(PacketTags{});
    issue_cmdbuf<kCmdBuf>();
    while (!idma_acked_cmdbuf<kCmdBuf>(kTrid)) {
    }
}

// Writes @p value to the 64-bit word at @p offset of the peer's port block for port @p p: on local a store, on noc an
// 8-byte copy from the port's staging word.
void write_peer_count(const Node& n, uint32_t p, uint32_t offset, uint64_t value) {
    const Port& port = n.ports[p];
    if (port.kind == kPortKindLocal) {
        *word64(port.peer_port_base + offset) = value;
        fence();
        return;
    }
    const uint32_t stage = n.staging + p * sizeof(uint64_t);
    *word64(stage) = value;
    fence();
    copy(n, stage, peer_addr(n, port, port.peer_port_base + offset), sizeof(uint64_t));
}

// Sends the slot at flat @p src on port @p p, its count after it. @p node_wrote: the CPU wrote the slot.
void send(const Node& n, uint32_t p, uint32_t src, bool node_wrote) {
    const Port& port = n.ports[p];
    const uint64_t sent = *word64(port.block + kPortSent);
    while (sent - *word64(port.block + kPortConsumedByPeer) >= n.ring_slots) {
    }
    if (node_wrote) {
        fence();
    }
    const uint32_t dest = port.peer_port_base + kPortHeaderBytes + (sent % n.ring_slots) * n.slot_bytes;
    copy(n, src, peer_addr(n, port, dest), n.slot_bytes);
    write_peer_count(n, p, kPortArrived, sent + 1);
    *word64(port.block + kPortSent) = sent + 1;
}

// Payload word w (bytes 4w..4w+3, little-endian) of packet seq from src to dst.
uint32_t payload_word(uint32_t src, uint32_t dst, uint32_t seq, uint32_t w) {
    uint32_t out = 0;
    for (uint32_t b = 0; b < sizeof(uint32_t); b++) {
        out |= static_cast<uint32_t>(payload_byte(src, dst, seq, w * sizeof(uint32_t) + b)) << (8 * b);
    }
    return out;
}

// Sends the packet in the slot at flat @p slot toward @p dst. False on a halt.
bool route(const Node& n, uint32_t slot, uint32_t dst, bool node_wrote) {
    const uint32_t p = dst < n.num_devices ? get_arg_val<uint32_t>(kArgNextPortBase + dst) : kNoPort;
    if (p == kNoPort) {
        return halt(n, kHaltNoRoute, dst);
    }
    send(n, p, slot, node_wrote);
    return true;
}

// Checks a packet for this device and counts it. False on a halt.
bool deliver(const Node& n, uint32_t slot) {
    const uint32_t src = *word32(slot) >> 16;
    const uint32_t seq = *word32(slot + 4);
    bool good = *word32(slot + 8) == n.nonce && *word32(slot + 12) == n.payload_bytes;
    const uint32_t payload = slot + sizeof(PacketHeader);
    for (uint32_t w = 0; good && w < n.payload_bytes / sizeof(uint32_t); w++) {
        good = *word32(payload + w * sizeof(uint32_t)) == payload_word(src, n.device, seq, w);
    }
    if (!good) {
        *ctl(n, kCtlBad) = *ctl(n, kCtlBad) + 1;
        return halt(n, kHaltBadPacket, seq);
    }
    *ctl(n, kCtlDelivered) = *ctl(n, kCtlDelivered) + 1;
    return true;
}

// Takes every packet that has arrived on port @p p, then credits the peer. False on a halt.
bool receive(const Node& n, uint32_t p) {
    const Port& port = n.ports[p];
    uint64_t received = *word64(port.block + kPortReceived);
    const uint64_t arrived = *word64(port.block + kPortArrived);
    if (arrived == received) {
        return true;
    }
    for (; received < arrived; received++) {
        const uint32_t slot = port.block + kPortHeaderBytes + (received % n.ring_slots) * n.slot_bytes;
        const uint32_t dst = *word32(slot) & 0xFFFF;
        if (!(dst == n.device ? deliver(n, slot) : route(n, slot, dst, false))) {
            return false;
        }
        *word64(port.block + kPortReceived) = received + 1;
    }
    write_peer_count(n, p, kPortConsumedByPeer, received);
    return true;
}

// Builds packet seq for dst in the source slot and sends it. False on a halt.
bool originate(const Node& n, uint32_t dst, uint32_t seq) {
    *word32(n.source) = dst | (n.device << 16);
    *word32(n.source + 4) = seq;
    *word32(n.source + 8) = n.nonce;
    *word32(n.source + 12) = n.payload_bytes;
    const uint32_t payload = n.source + sizeof(PacketHeader);
    for (uint32_t w = 0; w < n.payload_bytes / sizeof(uint32_t); w++) {
        *word32(payload + w * sizeof(uint32_t)) = payload_word(n.device, dst, seq, w);
    }
    if (!route(n, n.source, dst, true)) {
        return false;
    }
    *ctl(n, kCtlOriginated) = *ctl(n, kCtlOriginated) + 1;
    return true;
}

bool aligned_in_sram(uint64_t flat, uint64_t bytes) {
    return flat % kAlign == 0 && flat <= MEM_IERISC_L1_SIZE && bytes <= MEM_IERISC_L1_SIZE - flat;
}

// Reads and checks the runtime arguments into @p n. False on a halt, or, when region_base cannot hold the control
// block, with nothing written: there is nowhere to report it, and the host's wait for go times out.
bool load_args(Node& n) {
    const uint32_t region = get_arg_val<uint32_t>(kArgRegionBase);
    if (!aligned_in_sram(region, kControlBytes)) {
        return false;
    }
    n.control = region;
    n.device = get_arg_val<uint32_t>(kArgDevice);
    n.num_devices = get_arg_val<uint32_t>(kArgNumDevices);
    n.num_ports = get_arg_val<uint32_t>(kArgNumPorts);
    n.packets_per_flow = get_arg_val<uint32_t>(kArgPacketsPerFlow);
    n.payload_bytes = get_arg_val<uint32_t>(kArgPayloadBytes);
    n.ring_slots = get_arg_val<uint32_t>(kArgRingSlots);
    n.nonce = get_arg_val<uint32_t>(kArgNonce);
    if (n.num_devices == 0) {
        return halt(n, kHaltBadArg, kArgNumDevices);
    }
    if (n.device >= n.num_devices) {
        return halt(n, kHaltBadArg, kArgDevice);
    }
    if (n.num_ports > kMaxPorts) {
        return halt(n, kHaltBadArg, kArgNumPorts);
    }
    if (n.packets_per_flow == 0) {
        return halt(n, kHaltBadArg, kArgPacketsPerFlow);
    }
    if (n.payload_bytes % kPayloadGranule != 0) {
        return halt(n, kHaltBadArg, kArgPayloadBytes);
    }
    if (n.ring_slots == 0 && n.num_ports > 0) {
        return halt(n, kHaltBadArg, kArgRingSlots);
    }
    const Layout l = layout(n.num_ports, n.ring_slots, n.payload_bytes);
    if (!aligned_in_sram(region, l.total)) {
        return halt(n, kHaltBadArg, kArgRegionBase);
    }
    n.slot_bytes = l.slot_bytes;
    n.staging = region + l.staging;
    n.source = region + l.source;
    for (uint32_t d = 0; d < n.num_devices; d++) {
        const uint32_t p = get_arg_val<uint32_t>(kArgNextPortBase + d);
        if (p != kNoPort && p >= n.num_ports) {
            return halt(n, kHaltBadArg, kArgNextPortBase + d);
        }
    }
    n.own_tile = cce_eth::hsio_tile();
    for (uint32_t p = 0; p < n.num_ports; p++) {
        const uint32_t base = kArgNextPortBase + n.num_devices + p * kArgsPerPort;
        Port& port = n.ports[p];
        port.kind = get_arg_val<uint32_t>(base + kPortArgKind);
        port.peer_tile = get_arg_val<uint32_t>(base + kPortArgPeerTile);
        port.peer_port_base = get_arg_val<uint32_t>(base + kPortArgPeerPortBase);
        port.block = region + l.port_block(p);
        if (port.kind != kPortKindNoc && port.kind != kPortKindLocal) {
            return halt(n, kHaltBadArg, base + kPortArgKind);
        }
        if (!aligned_in_sram(port.peer_port_base, l.port_block_bytes)) {
            return halt(n, kHaltBadArg, base + kPortArgPeerPortBase);
        }
        if (port.kind == kPortKindNoc && port.peer_tile == n.own_tile) {
            return halt(n, kHaltNocPeerOnOwnTile, p);
        }
    }
    n.req_vc = kFirstReqVc + internal_::get_hw_thread_idx();
    return true;
}

}  // namespace

void kernel_main() {
    Node n{};
    if (!load_args(n)) {
        return;
    }
    // Once, before go: go is written after every node of the run reports kStatusWaitingForGo, so no copy is in flight.
    overlay::reset_cmdbuf<kCmdBuf>();
    *ctl(n, kCtlStatus) = kStatusWaitingForGo;
    // Stop before go: the host gave up on another node, and this one ends without running.
    while (*ctl(n, kCtlGo) == 0) {
        if (*ctl(n, kCtlStop) != 0) {
            return;
        }
    }
    *ctl(n, kCtlStatus) = kStatusRunning;

    const uint32_t to_originate = n.packets_per_flow * (n.num_devices - 1);
    uint32_t originated = 0;
    uint32_t dst = n.device == 0 ? 1 : 0;
    uint32_t seq = 0;
    while (*ctl(n, kCtlStop) == 0) {
        for (uint32_t p = 0; p < n.num_ports; p++) {
            if (!receive(n, p)) {
                return;
            }
        }
        if (originated < to_originate) {
            if (!originate(n, dst, seq)) {
                return;
            }
            originated++;
            if (++seq == n.packets_per_flow) {
                seq = 0;
                dst = dst + 1 == n.device ? dst + 2 : dst + 1;
            }
        }
    }
    *ctl(n, kCtlStatus) = kStatusDone;
}
