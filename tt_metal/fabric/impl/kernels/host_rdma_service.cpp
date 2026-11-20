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
// payload into the router's host_rdma channel. Up to WINDOW packets are in flight, each on its own NoC transaction ID:
// a read is issued as soon as the channel has a free slot for the packet, the packet is sent as soon as the read lands,
// and packets retire (in order) as their writes are acked. A request is consumed once its last packet has retired.
//
// Each channel is a fabric worker connection (WorkerToFabricEdmSender) to the router's host_rdma channel, a worker
// channel the fabric's worker connection APIs do not hand out: it sends each packet into the router's next slot and
// takes the slot on the router's free-slots register, keeps the slot cursor and the free-slot count, and opens and
// closes as any worker. The router credits it in the channel's `credits` word here.

#include <cstdint>
#include <limits>
#include <new>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/host_rdma_service_abi.hpp"

namespace abi = tt::tt_fabric::host_rdma::service_abi;

namespace {

constexpr uint32_t L1_ALIGN = 16;
constexpr uint32_t HEADER_BYTES = sizeof(PACKET_HEADER_TYPE);
static_assert(HEADER_BYTES % L1_ALIGN == 0);

// The kernel's state per channel.
struct ChannelState {
    tt::tt_fabric::WorkerToFabricEdmSender conn;  // the router channel
    uint32_t taken;                               // requests loaded from the queue
    uint32_t consumed;                            // requests whose last packet has retired
    uint32_t reading;                             // packets accepted and not yet sent (each holds a free slot)
    bool carving;                                 // `req` has bytes left to packetize
    uint32_t off;                                 // bytes of `req` packetized so far
    abi::Request req;
};

// A packet in flight, on transaction ID index + 1.
struct Packet {
    uint32_t bytes;  // header and payload
    uint8_t channel;
    bool last;  // the last packet of its request
};

volatile tt_l1_ptr abi::Header* header;
volatile tt_l1_ptr abi::Channel* channels;
// The channels' state, constructed when a channel opens (the kernel environment has no static constructors).
alignas(ChannelState) uint8_t state_storage[sizeof(ChannelState) * abi::MAX_CHANNELS];
bool initialized[abi::MAX_CHANNELS];
FORCE_INLINE ChannelState& state(uint32_t c) { return reinterpret_cast<ChannelState*>(state_storage)[c]; }

// The channel whose router the connections' stateful command buffers are set up for (none: NO_CHANNEL).
constexpr uint32_t NO_CHANNEL = abi::MAX_CHANNELS;
uint32_t stateful_channel = NO_CHANNEL;

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

// Retires packets in order whose writes are acked, and publishes each channel's consumed requests.
void retire() {
    bool any = false;
    while (writing != 0 && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid_of(head))) {
        const Packet& p = window[head];
        if (p.last) {
            state(p.channel).consumed++;
        }
        head = tt::tt_fabric::wrap_increment<abi::WINDOW>(head);
        n--;
        writing--;
        any = true;
    }
    if (!any) {
        return;
    }
    for (uint32_t c = 0; c < abi::MAX_CHANNELS; c++) {
        if (initialized[c]) {
            channels[c].consumed = state(c).consumed;
        }
    }
}

// Sends, in order, the packets whose payload reads have landed, each through its channel's connection on the
// transaction ID its read used.
void issue_writes() {
    while (writing != n) {
        const uint32_t index = tt::tt_fabric::wrap_increment_n<abi::WINDOW>(head, writing);
        if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid_of(index))) {
            return;
        }
        const Packet& p = window[index];
        ChannelState& s = state(p.channel);
        if (stateful_channel != p.channel) {
            s.conn.setup_stateful_send_cmd_bufs(noc_index);
            stateful_channel = p.channel;
        }
        s.conn.send_current_slot_stateful_non_blocking_from_address_with_trid(
            staging_of(index), p.bytes, trid_of(index), noc_index);
        settle(s.conn.get_stateful_send_data_noc_cmd_buf());
        s.reading--;
        writing++;
    }
}

// Starts packets, round-robin over the channels, while the window has room: each takes the channel's next router
// slot (once the router has freed it), gets its header, and has its payload read issued.
void accept() {
    for (uint32_t tried = 0; tried < abi::MAX_CHANNELS && n != abi::WINDOW; tried++) {
        const uint32_t c = next_channel;
        next_channel = tt::tt_fabric::wrap_increment<abi::MAX_CHANNELS>(next_channel);
        if (!initialized[c]) {
            continue;
        }
        volatile abi::Channel& ch = channels[c];
        ChannelState& s = state(c);
        while (n != abi::WINDOW) {
            if (!s.carving) {
                if (s.taken == ch.produced) {
                    break;
                }
                s.req = load_request(ch.queue[s.taken & (abi::QUEUE_DEPTH - 1)]);
                s.taken++;
                s.off = 0;
                s.carving = true;
            }
            // A free slot for the packet, besides those held by the channel's packets still reading.
            if (s.conn.get_num_free_write_slots() <= s.reading) {
                break;
            }
            const uint32_t remaining = s.req.size - s.off;
            const uint32_t payload = remaining < ch.max_payload ? remaining : ch.max_payload;
            const bool last = payload == remaining;
            const uint32_t index = tt::tt_fabric::wrap_increment_n<abi::WINDOW>(head, n);
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
            s.reading++;
            n++;
            s.off += payload;
            s.carving = !last;
        }
    }
}

// Opens channel c's connection: from the producer cursor the channel's last producer left in the router, and the
// router's count of slots freed (into `credits`).
void open_channel(uint32_t c) {
    volatile abi::Channel& ch = channels[c];
    ChannelState& s = *new (&state(c)) ChannelState{};
    s.taken = ch.consumed;
    s.consumed = ch.consumed;
    s.conn.init<ProgrammableCoreType::TENSIX>(
        /*connected_to_persistent_fabric=*/true,
        static_cast<uint8_t>(ch.router_x),
        static_cast<uint8_t>(ch.router_y),
        ch.slots_base,
        static_cast<uint8_t>(ch.num_slots),
        ch.connection_semaphore_addr,
        ch.connection_info_addr,
        static_cast<uint16_t>(ch.slot_bytes),
        ch.producer_cursor_addr,
        &ch.credits,
        &ch.teardown,
        reinterpret_cast<uint32_t>(&ch.cursor_scratch[0]),
        ch.free_slots_stream,
        StreamId{std::numeric_limits<uint32_t>::max()});  // unused: credits come in `credits`
    noc_async_read_set_trid(0);                           // the open's reads are on no packet's transaction ID
    s.conn.open();
    stateful_channel = NO_CHANNEL;  // the open's writes reprogram the command buffers
}

// Closes channel c's connection (and waits for the router's teardown ack).
void close_channel(uint32_t c) {
    state(c).conn.close();
    stateful_channel = NO_CHANNEL;  // the close's writes reprogram the command buffers
}

// A channel becoming active is opened; one going inactive is closed.
void scan_channels() {
    for (uint32_t c = 0; c < abi::MAX_CHANNELS; c++) {
        const bool active = channels[c].active != 0;
        if (active && !initialized[c]) {
            open_channel(c);
        } else if (!active && initialized[c]) {
            close_channel(c);
        }
        initialized[c] = active;
    }
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
        // The host terminates only once it has drained every channel, so the kernel stops at once.
        if (header->terminate != 0) {
            break;
        }
    }
    noc_async_full_barrier();
    for (uint32_t c = 0; c < abi::MAX_CHANNELS; c++) {
        if (initialized[c]) {
            close_channel(c);
        }
    }
}
