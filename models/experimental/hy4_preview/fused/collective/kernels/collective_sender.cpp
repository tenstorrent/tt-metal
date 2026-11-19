// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The sender core of the in-program collective (fused/collective/collective.py), one per chip. It holds this
// chip's fabric connections for the whole program and serves `num_ops` hand-overs in whatever order the worker
// cores make them: an op is ready when a worker has written its payload into the op's staging buffer on this core
// and set the op's bit in the ready word (collective.hpp hand_over). For a ready op the sender
//   - copies the payload over the NoC into this chip's own slot of the landing and counts one arrival there;
//   - sends it over the fabric into the same slot on every other ring member: `chunk` bytes per packet, the last
//     packet of each route a fused write + increment of the member's arrival counter (packets of one route arrive
//     in order, so the count means the whole payload is there).
// A route is one unicast with 2D routing to a member any number of hops away, or one multicast walk (hop counts
// east / west / north / south from this chip) that delivers to every chip it passes.
//
// Runtime args (this core):
//   ready semaphore id, num_ops, num_connections, num_routes,
//   per route: connection index, 0 (unicast) | 1 (multicast), destination chip id, hops e | w << 8 | n << 16 | s << 24
//   per op:    staging address, payload bytes, chunk bytes, landing core NoC x | y << 16,
//              this chip's slot of parity 0, bytes from parity 0 to parity 1, the site's control words,
//              number of routes, then that many route indices
//   then the routing plane connection manager's arguments (ttnn.setup_routing_plane_connection).
// Compile-time args: the number of routes (one packet header each; ops over the same ring share them).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#include "collective.hpp"

#if !defined(FABRIC_2D)
#error "the collective sender routes by chip id and by 2D multicast walks: it needs a 2D fabric"
#endif

using namespace tt::tt_fabric;

constexpr uint32_t num_routes = get_compile_time_arg_val(0);
constexpr uint32_t MAX_OPS = 16;  // two bits each in one ready word
static_assert(
    num_routes >= 1 && num_routes <= MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType / PACKET_HEADER_MAX_SIZE,
    "the sender's routes exceed this RISC's share of the packet header pool");

namespace {

constexpr uint32_t ROUTE_ARGS = 4;

struct Op {
    uint32_t staging;
    uint32_t bytes;
    uint32_t chunk;
    uint32_t landing_xy;
    uint32_t slot;
    uint32_t parity_stride;
    uint32_t ctrl;
    uint32_t num_routes;
    uint32_t route_args;  // index of this op's first route index argument
};

}  // namespace

void kernel_main() {
    [[maybe_unused]] uint32_t stamps[5] = {hy4_collective::now(), 0, 0, 0, 0};  // start, open, ready, sent, closed
    size_t arg = 0;
    const uint32_t ready_id = get_arg_val<uint32_t>(arg++);
    const uint32_t num_ops = get_arg_val<uint32_t>(arg++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg++);
    const uint32_t routes_given = get_arg_val<uint32_t>(arg++);
    const size_t route_table = arg;
    arg += routes_given * ROUTE_ARGS;

    Op ops[MAX_OPS];
    for (uint32_t k = 0; k < num_ops; ++k) {
        Op& op = ops[k];
        op.staging = get_arg_val<uint32_t>(arg++);
        op.bytes = get_arg_val<uint32_t>(arg++);
        op.chunk = get_arg_val<uint32_t>(arg++);
        op.landing_xy = get_arg_val<uint32_t>(arg++);
        op.slot = get_arg_val<uint32_t>(arg++);
        op.parity_stride = get_arg_val<uint32_t>(arg++);
        op.ctrl = get_arg_val<uint32_t>(arg++);
        op.num_routes = get_arg_val<uint32_t>(arg++);
        op.route_args = arg;
        arg += op.num_routes;
    }

    WAYPOINT("MKSO");
    auto fabric = RoutingPlaneConnectionManager::build_from_args<
        RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg, num_connections);

    // One packet header per route, routed once: the routes do not change over the program.
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[num_routes];
    uint8_t route_connection[num_routes];
    for (uint32_t r = 0; r < num_routes; ++r) {
        const size_t ra = route_table + r * ROUTE_ARGS;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
        headers[r] = header;
        route_connection[r] = static_cast<uint8_t>(get_arg_val<uint32_t>(ra));
        if (get_arg_val<uint32_t>(ra + 1) == 0) {
            (void)fabric_set_unicast_route(
                header, static_cast<uint16_t>(get_arg_val<uint32_t>(ra + 2)), fabric.my_mesh_id);
        } else {
            const uint32_t hops = get_arg_val<uint32_t>(ra + 3);
            fabric_set_mcast_route(
                header,
                fabric.my_chip_id,
                fabric.my_mesh_id,
                hops & 0xFF,
                (hops >> 8) & 0xFF,
                (hops >> 16) & 0xFF,
                (hops >> 24) & 0xFF);
        }
    }
    fabric.open_finish();
    stamps[1] = hy4_collective::now();
    WAYPOINT("MKSR");

    volatile tt_l1_ptr uint32_t* ready = hy4_collective::word(get_semaphore(ready_id));
    const uint32_t all_ops = num_ops >= 32 ? 0xFFFFFFFFu : ((1u << num_ops) - 1u);
    uint32_t pending = all_ops;
#if HY4_COLLECTIVE_WAIT_LIMIT != 0
    uint32_t polls = 0;
#endif
    while (pending != 0) {
        invalidate_l1_cache();
        const uint32_t bits = ready[0];
        bool served = false;
        for (uint32_t k = 0; k < num_ops; ++k) {
            if ((pending & (1u << k)) == 0 || ((bits >> (2 * k)) & 3u) == 0) {
                continue;
            }
            const Op& op = ops[k];
            if (pending == all_ops) {
                stamps[2] = hy4_collective::now();
            }
            const uint32_t parity = (bits >> (2 * k + 1)) & 1u;
            const uint32_t lx = op.landing_xy & 0xFFFF;
            const uint32_t ly = op.landing_xy >> 16;
            const uint32_t slot = op.slot + parity * op.parity_stride;
            const uint64_t counter = get_noc_addr(lx, ly, op.ctrl + hy4_collective::CTRL_COUNTER + 16 * parity);

            // This chip's own slot.
            noc_async_write(op.staging, get_noc_addr(lx, ly, slot), op.bytes);

            for (uint32_t offset = 0; offset < op.bytes; offset += op.chunk) {
                const bool last = offset + op.chunk >= op.bytes;
                const uint32_t bytes = last ? op.bytes - offset : op.chunk;
                const uint64_t destination = get_noc_addr(lx, ly, slot + offset);
                // A header's previous packet (of this op or an earlier one) must have left L1 before it changes.
                noc_async_writes_flushed();
                for (uint32_t i = 0; i < op.num_routes; ++i) {
                    const uint32_t r = get_arg_val<uint32_t>(op.route_args + i);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = headers[r];
                    if (last) {
                        header->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{destination, counter, 1, true}, bytes);
                    } else {
                        header->to_noc_unicast_write(NocUnicastCommandHeader{destination}, bytes);
                    }
                    auto& connection = fabric.get(route_connection[r]).sender;
                    connection.wait_for_empty_write_slot();
                    connection.send_payload_without_header_non_blocking_from_address(op.staging + offset, bytes);
                    connection.send_payload_flush_non_blocking_from_address(
                        reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
                }
            }
            // The own slot is counted last, after every write out of the staging buffer has been acknowledged: the
            // landing core cannot see its round complete, and so cannot hand the next payload into the same
            // staging buffer, while a packet of this one is still being read from it.
            noc_async_write_barrier();
            noc_semaphore_inc(counter, 1);
            pending &= ~(1u << k);
            served = true;
        }
#if HY4_COLLECTIVE_WAIT_LIMIT != 0
        polls = served ? 0 : polls + 1;
        if (polls >= HY4_COLLECTIVE_WAIT_LIMIT) {
            // A worker never handed its payload over: report on every op still waiting and let the program end.
            for (uint32_t k = 0; k < num_ops; ++k) {
                if (pending & (1u << k)) {
                    noc_inline_dw_write(
                        get_noc_addr(
                            ops[k].landing_xy & 0xFFFF,
                            ops[k].landing_xy >> 16,
                            ops[k].ctrl + hy4_collective::CTRL_STATUS),
                        hy4_collective::STATUS_READY_TIMEOUT | (pending & 0xFFFF));
                }
            }
            break;
        }
#else
        (void)served;
#endif
    }

    stamps[3] = hy4_collective::now();
    WAYPOINT("MKSC");
    noc_async_writes_flushed();
    fabric.close();
    noc_async_full_barrier();
#if HY4_COLLECTIVE_TIMING
    stamps[4] = hy4_collective::now();
    for (uint32_t i = 0; i < 5; ++i) {
        noc_inline_dw_write(
            get_noc_addr(
                ops[0].landing_xy & 0xFFFF,
                ops[0].landing_xy >> 16,
                ops[0].ctrl + hy4_collective::CTRL_STAMPS + 16 * (hy4_collective::STAMP_SENDER_START + i)),
            stamps[i]);
    }
    noc_async_full_barrier();
#endif
    WAYPOINT("MKSD");
}
