// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// host_rdma service kernel: services host write requests on one chip (the device side of HostRdmaService,
// tt_metal/fabric/host_rdma/host_rdma.hpp; the contract is host_rdma_service_abi.hpp).
//
// The host posts requests into per-router queues in this core's L1. For each request the kernel splits the transfer
// into packets and, per packet: forms the header (route to the destination chip from this core's routing tables; a
// NoC write at the destination, the last packet's fused with the request's completion increment) in a staging slot,
// reads the payload from the request's source on this chip into the slot after the header, then writes header and
// payload into the router's next host_rdma slot. Up to WINDOW packets are in flight, each on its own NoC transaction
// ID: a read is issued as soon as the packet's router slot is free, its write as soon as the read lands, and as writes
// are acked (in order) the packets' slots are taken on the router's free-slots register, as a worker connected to the
// router would. A request is consumed once its last packet has been taken.
//
// The router credits this kernel: its host_rdma credit target is the channel's `credits` word here.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/host_rdma_service_abi.hpp"

namespace abi = tt::tt_fabric::host_rdma::service_abi;

namespace {

// The router's free-slots register takes a signed slot increment at this bit of its increment-on-write register
// (REMOTE_DEST_BUF_WORDS_FREE_INC).
constexpr uint32_t FREE_SLOTS_INC_SHIFT = 6;
constexpr uint32_t L1_ALIGN = 16;
constexpr uint32_t HEADER_BYTES = sizeof(PACKET_HEADER_TYPE);
static_assert(HEADER_BYTES % L1_ALIGN == 0);

// The kernel's state per channel.
struct ChannelState {
    uint32_t taken;     // requests loaded from the queue
    uint32_t consumed;  // requests whose last packet has been taken by the router
    uint32_t claimed;   // router slots claimed by packets (absolute, from the channel's start)
    uint32_t retired;   // packets whose write is acked
    uint32_t rung;      // packets whose slots are taken on the router
    bool carving;       // `req` has bytes left to packetize
    uint32_t off;       // bytes of `req` packetized so far
    abi::Request req;
};

// A packet in flight, on transaction ID index + 1.
struct Packet {
    uint64_t dst;    // its router slot
    uint32_t bytes;  // header and payload
    uint8_t channel;
    bool last;  // the last packet of its request
};

volatile tt_l1_ptr abi::Header* header;
volatile tt_l1_ptr abi::Channel* channels;
ChannelState state[abi::MAX_CHANNELS];
bool initialized[abi::MAX_CHANNELS];

// The window, oldest first: [head, head + writing) have their writes issued, the rest of the n are reading.
Packet window[abi::WINDOW];
uint32_t head = 0;
uint32_t n = 0;
uint32_t writing = 0;
uint32_t next_channel = 0;

FORCE_INLINE uint32_t trid_of(uint32_t index) { return index + 1; }
FORCE_INLINE uint32_t staging_of(uint32_t index) { return header->staging_addr + index * header->staging_stride; }

// After issuing a request on `cmd_buf`: the NIU has taken it (so its transaction ID's outstanding count includes it)
// before any later status read. A later load may otherwise be performed before the stores that issued the request
// (Blackhole's BabyRISCV reorders a load to another address ahead of an earlier store), and a poll of the ID would see
// nothing outstanding.
FORCE_INLINE void settle(uint32_t cmd_buf) {
    asm volatile("fence" ::: "memory");
    while (!noc_cmd_buf_ready(noc_index, cmd_buf)) {
    }
    asm volatile("fence" ::: "memory");
}

FORCE_INLINE abi::Endpoint load_endpoint(const volatile abi::Endpoint& e) {
    abi::Endpoint out;
    out.addr = e.addr;
    out.x = e.x;
    out.y = e.y;
    return out;
}

FORCE_INLINE abi::Request load_request(const volatile abi::Request& r) {
    abi::Request q{};
    q.src = load_endpoint(r.src);
    q.dst = load_endpoint(r.dst);
    q.completion = load_endpoint(r.completion);
    q.size = r.size;
    q.completion_inc = r.completion_inc;
    q.dst_dev_id = r.dst_dev_id;
    q.dst_mesh_id = r.dst_mesh_id;
    q.hops = r.hops;
    return q;
}

// The header of the next packet of `s.req` (`payload` bytes at offset s.off), in the staging slot at `addr`.
FORCE_INLINE void form_header(uint32_t addr, const ChannelState& s, uint32_t payload, bool last) {
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(addr);
#if defined(FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE)  // 2D fabric (FabricContext::get_fabric_kernel_defines)
    fabric_set_unicast_route(hdr, s.req.dst_dev_id, s.req.dst_mesh_id);
#else
    fabric_set_unicast_route<false>(hdr, static_cast<uint16_t>(s.req.hops));
#endif
    const uint64_t dst = get_noc_addr(s.req.dst.x, s.req.dst.y, s.req.dst.addr + s.off);
    if (last && s.req.completion_inc != 0) {
        hdr->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                dst,
                get_noc_addr(s.req.completion.x, s.req.completion.y, s.req.completion.addr),
                s.req.completion_inc,
                /*flush=*/true},
            payload);
    } else {
        hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, payload);
    }
}

// Retires packets in order whose writes are acked, takes each channel's retired slots on its router, and publishes
// each channel's consumed requests.
void retire() {
    bool any = false;
    while (writing != 0 && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid_of(head))) {
        const Packet& p = window[head];
        ChannelState& s = state[p.channel];
        s.retired++;
        if (p.last) {
            s.consumed++;
        }
        head = (head + 1) % abi::WINDOW;
        n--;
        writing--;
        any = true;
    }
    if (!any) {
        return;
    }
    for (uint32_t c = 0; c < abi::MAX_CHANNELS; c++) {
        ChannelState& s = state[c];
        const uint32_t count = s.retired - s.rung;
        if (count == 0) {
            continue;
        }
        const volatile abi::Channel& ch = channels[c];
        noc_inline_dw_write<InlineWriteDst::REG>(
            get_noc_addr(ch.router_x, ch.router_y, ch.free_slots_addr),
            static_cast<uint32_t>(-static_cast<int32_t>(count)) << FREE_SLOTS_INC_SHIFT);
        s.rung = s.retired;
        channels[c].consumed = s.consumed;
    }
}

// Issues writes, in order, for packets whose payload reads have landed (each on the transaction ID its read used).
void issue_writes() {
    while (writing != n) {
        const uint32_t index = (head + writing) % abi::WINDOW;
        if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid_of(index))) {
            return;
        }
        const Packet& p = window[index];
        noc_async_write_one_packet_with_trid(staging_of(index), p.dst, p.bytes, trid_of(index));
        settle(write_cmd_buf);
        writing++;
    }
}

// Starts packets, round-robin over the channels, while the window has room: each takes the channel's next router
// slot (once the router has freed it), gets its header, and has its payload read issued.
void accept() {
    for (uint32_t tried = 0; tried < abi::MAX_CHANNELS && n != abi::WINDOW; tried++) {
        const uint32_t c = next_channel;
        next_channel = (next_channel + 1) % abi::MAX_CHANNELS;
        if (!initialized[c]) {
            continue;
        }
        volatile abi::Channel& ch = channels[c];
        ChannelState& s = state[c];
        while (n != abi::WINDOW) {
            if (!s.carving) {
                if (s.taken == ch.produced) {
                    break;
                }
                s.req = load_request(ch.queue[s.taken % abi::QUEUE_DEPTH]);
                s.taken++;
                s.off = 0;
                s.carving = true;
            }
            // A free router slot: fewer than num_slots claimed and not yet credited.
            if (s.claimed - ch.credits >= ch.num_slots) {
                break;
            }
            const uint32_t remaining = s.req.size - s.off;
            const uint32_t payload = remaining < ch.max_payload ? remaining : ch.max_payload;
            const bool last = payload == remaining;
            const uint32_t index = (head + n) % abi::WINDOW;
            const uint32_t staging = staging_of(index);

            form_header(staging, s, payload, last);
            noc_async_read_set_trid(trid_of(index));
            noc_async_read_one_packet(
                get_noc_addr(s.req.src.x, s.req.src.y, s.req.src.addr + s.off), staging + HEADER_BYTES, payload);
            settle(read_cmd_buf);

            Packet& p = window[index];
            p.channel = static_cast<uint8_t>(c);
            p.last = last;
            p.bytes = (HEADER_BYTES + payload + L1_ALIGN - 1) & ~(L1_ALIGN - 1);
            p.dst = get_noc_addr(ch.router_x, ch.router_y, ch.slots_base + (s.claimed % ch.num_slots) * ch.slot_bytes);
            s.claimed++;
            n++;
            s.off += payload;
            s.carving = !last;
        }
    }
}

// A channel becoming active starts where the host says the router is; one going inactive is forgotten.
void scan_channels() {
    for (uint32_t c = 0; c < abi::MAX_CHANNELS; c++) {
        const bool active = channels[c].active != 0;
        if (active && !initialized[c]) {
            const uint32_t start = channels[c].start;
            state[c] = ChannelState{};
            state[c].taken = channels[c].consumed;
            state[c].consumed = channels[c].consumed;
            state[c].claimed = start;
            state[c].retired = start;
            state[c].rung = start;
        }
        initialized[c] = active;
    }
}

bool idle() {
    if (n != 0) {
        return false;
    }
    for (uint32_t c = 0; c < abi::MAX_CHANNELS; c++) {
        if (initialized[c] && (state[c].carving || state[c].taken != channels[c].produced)) {
            return false;
        }
    }
    return true;
}

}  // namespace

void kernel_main() {
    header = reinterpret_cast<volatile tt_l1_ptr abi::Header*>(get_arg_val<uint32_t>(0));
    channels =
        reinterpret_cast<volatile tt_l1_ptr abi::Channel*>(reinterpret_cast<uint32_t>(header) + sizeof(abi::Header));
    for (uint32_t c = 0; c < abi::MAX_CHANNELS; c++) {
        initialized[c] = false;
    }

    uint32_t pass = 0;
    while (true) {
        invalidate_l1_cache();
        if ((pass++ & 63) == 0) {
            header->heartbeat = pass;
            scan_channels();
        }
        retire();
        issue_writes();
        accept();
        if (header->terminate != 0 && idle()) {
            break;
        }
    }
    noc_async_full_barrier();
}
