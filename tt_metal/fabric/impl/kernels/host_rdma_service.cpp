// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

constexpr uint32_t HEADER_BYTES = sizeof(PACKET_HEADER_TYPE);
static_assert(HEADER_BYTES % abi::NOC_ALIGNMENT == 0);

struct ChannelState {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    uint32_t loaded;
    uint32_t consumed;
    uint32_t packets_reading;
    bool packetizing;
    uint32_t bytes_packetized;
    abi::Request req;
};

struct Packet {
    uint32_t header_and_payload_bytes;
    uint8_t channel;
    bool last_of_request;
};

volatile tt_l1_ptr abi::Header* header;
volatile tt_l1_ptr abi::Channel* channels;
// Kernels run no static constructors: each state is constructed in open_channel().
alignas(ChannelState) uint8_t state_storage[sizeof(ChannelState) * abi::MAX_CHANNELS];
bool initialized[abi::MAX_CHANNELS];
FORCE_INLINE ChannelState& state(uint32_t c) { return reinterpret_cast<ChannelState*>(state_storage)[c]; }

constexpr uint32_t NO_CHANNEL = abi::MAX_CHANNELS;
uint32_t stateful_cmd_bufs_channel = NO_CHANNEL;
FORCE_INLINE void forget_stateful_cmd_bufs() { stateful_cmd_bufs_channel = NO_CHANNEL; }

constexpr uint32_t NO_PACKET_TRID = 0;
FORCE_INLINE StreamId unused_stream() { return StreamId{std::numeric_limits<uint32_t>::max()}; }

Packet window[abi::WINDOW];
uint32_t head = 0;
uint32_t in_flight = 0;
uint32_t writes_issued = 0;
uint32_t next_channel = 0;

FORCE_INLINE uint32_t trid_of(uint32_t index) { return index + 1; }
FORCE_INLINE uint32_t staging_of(uint32_t index) { return header->staging_addr + index * header->staging_stride; }

// Blackhole's BabyRISCV can perform a later load before the stores that issued a NoC request, so a poll of the
// request's transaction ID could otherwise see nothing outstanding.
FORCE_INLINE void wait_until_noc_took_request(uint32_t cmd_buf) {
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
    return q;
}

FORCE_INLINE void form_header(uint32_t addr, const ChannelState& s, uint32_t payload, bool last) {
    auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(addr);
    fabric_set_unicast_route(hdr, s.req.dst_dev_id, s.req.dst_mesh_id);
    const uint64_t dst = get_noc_addr(s.req.dst.x, s.req.dst.y, s.req.dst.addr + s.bytes_packetized);
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

void retire_acked_packets() {
    bool any = false;
    while (writes_issued != 0 && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid_of(head))) {
        const Packet& p = window[head];
        if (p.last_of_request) {
            state(p.channel).consumed++;
        }
        head = tt::tt_fabric::wrap_increment<abi::WINDOW>(head);
        in_flight--;
        writes_issued--;
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

void send_packets_whose_reads_landed() {
    while (writes_issued != in_flight) {
        const uint32_t index = tt::tt_fabric::wrap_increment_n<abi::WINDOW>(head, writes_issued);
        if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid_of(index))) {
            return;
        }
        const Packet& p = window[index];
        ChannelState& s = state(p.channel);
        if (stateful_cmd_bufs_channel != p.channel) {
            s.conn.setup_stateful_send_cmd_bufs(noc_index);
            stateful_cmd_bufs_channel = p.channel;
        }
        s.conn.send_current_slot_stateful_non_blocking_from_address_with_trid(
            staging_of(index), p.header_and_payload_bytes, trid_of(index), noc_index);
        wait_until_noc_took_request(s.conn.get_stateful_send_data_noc_cmd_buf());
        s.packets_reading--;
        writes_issued++;
    }
}

void start_packets_round_robin() {
    for (uint32_t tried = 0; tried < abi::MAX_CHANNELS && in_flight != abi::WINDOW; tried++) {
        const uint32_t c = next_channel;
        next_channel = tt::tt_fabric::wrap_increment<abi::MAX_CHANNELS>(next_channel);
        if (!initialized[c]) {
            continue;
        }
        volatile abi::Channel& ch = channels[c];
        ChannelState& s = state(c);
        while (in_flight != abi::WINDOW) {
            if (!s.packetizing) {
                if (s.loaded == ch.produced) {
                    break;
                }
                s.req = load_request(ch.queue[s.loaded & (abi::QUEUE_DEPTH - 1)]);
                s.loaded++;
                s.bytes_packetized = 0;
                s.packetizing = true;
            }
            const bool router_slot_free = s.conn.get_num_free_write_slots() > s.packets_reading;
            if (!router_slot_free) {
                break;
            }
            const uint32_t remaining = s.req.size - s.bytes_packetized;
            const uint32_t payload = remaining < ch.max_payload ? remaining : ch.max_payload;
            const bool last = payload == remaining;
            const uint32_t index = tt::tt_fabric::wrap_increment_n<abi::WINDOW>(head, in_flight);
            const uint32_t staging = staging_of(index);

            form_header(staging, s, payload, last);
            noc_async_read_set_trid(trid_of(index));
            noc_async_read_one_packet(
                get_noc_addr(s.req.src.x, s.req.src.y, s.req.src.addr + s.bytes_packetized),
                staging + HEADER_BYTES,
                payload);
            wait_until_noc_took_request(read_cmd_buf);

            Packet& p = window[index];
            p.channel = static_cast<uint8_t>(c);
            p.last_of_request = last;
            p.header_and_payload_bytes = abi::align_up(HEADER_BYTES + payload, abi::NOC_ALIGNMENT);
            s.packets_reading++;
            in_flight++;
            s.bytes_packetized += payload;
            s.packetizing = !last;
        }
    }
}

void open_channel(uint32_t c) {
    volatile abi::Channel& ch = channels[c];
    ChannelState& s = *new (&state(c)) ChannelState{};
    s.loaded = ch.consumed;
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
        unused_stream());
    noc_async_read_set_trid(NO_PACKET_TRID);
    s.conn.open();
    forget_stateful_cmd_bufs();
}

void close_channel(uint32_t c) {
    state(c).conn.close();
    forget_stateful_cmd_bufs();
}

void open_or_close_channels_by_active_flag() {
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
            open_or_close_channels_by_active_flag();
        }
        retire_acked_packets();
        send_packets_whose_reads_landed();
        start_packets_round_robin();
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
