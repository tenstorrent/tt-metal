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
//
// Forms a program built outside collective.py's Fragment may ask for (fused/layer_common/sender.py), all off in a
// Fragment's argument table:
//   - the first argument with bit 31 set: the ready word is not a program semaphore but the persistent L1 word at
//     that address (bits 0..30) on this core. The sender takes the bits it served off the word when it ends, so the
//     word is zero again for the next launch (a program semaphore is rewritten by the launch itself).
//   - an op's number of routes with bit 8 set (OP_NO_OWN_SLOT): the payload goes to the routes only; this chip has
//     no slot of its own in the landing and counts nothing here (a member of a reduce towards another chip).
//   - no routes at all (compile-time arg 0) and no connections: every op is this chip's own copy (a one-chip run).
//   - the first argument with bit 30 set (READY_HAS_SERVED; with bit 31 only): one more argument follows the four
//     head arguments, the address of a second persistent L1 word on this core, the "served" word. When an op has
//     been served (every packet handed to the routers and every NoC write out of its staging acknowledged) the
//     sender adds bit `1 << op` to it by an atomic on itself: whoever handed the op over may then reuse the op's
//     staging for another op. The reader of the word takes the bits it has seen off again (collective.hpp is not
//     involved; fused/layer_common/sender.py Sender(served=...)).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#include "collective.hpp"

// HY4_COLLECTIVE_PARSE_ONLY (with HY4_COLLECTIVE_PARSE_PAGE, an L1 address on this core): the sender reads its whole
// argument table exactly as the real build does (head arguments, served word, every route, every op), writes what
// it parsed to that page and returns WITHOUT opening a fabric connection and without sending. The host compares the
// page with the table it built (fused/layer_common/sender.py Sender(parse_page=...), Sender.parsed). Every change of
// this kernel's argument layout is run this way on the board before a build that sends: a mis-parsed route with
// connections open does not recover by the operation timeout.
// Page (uint32): 0x50415253, ready word (bit 30 stripped), has served word, served word address, ops, connections,
// routes; 4 words per route; per op: staging, bytes, chunk, landing x | y << 16, slot, parity stride, control words,
// routes, own slot, then its route indices; the index of the first fabric argument; 0x454E4421.

#if !defined(FABRIC_2D)
#error "the collective sender routes by chip id and by 2D multicast walks: it needs a 2D fabric"
#endif

using namespace tt::tt_fabric;

constexpr uint32_t num_routes = get_compile_time_arg_val(0);
constexpr uint32_t MAX_OPS = 16;  // two bits each in one ready word
static_assert(
    num_routes <= MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType / PACKET_HEADER_MAX_SIZE,
    "the sender's routes exceed this RISC's share of the packet header pool");
constexpr uint32_t header_slots = num_routes == 0 ? 1 : num_routes;
constexpr uint32_t READY_IS_ADDRESS = 1u << 31;
constexpr uint32_t READY_HAS_SERVED = 1u << 30;
constexpr uint32_t OP_ROUTES_MASK = 0xFFu;
constexpr uint32_t OP_NO_OWN_SLOT = 1u << 8;
#ifdef HY4_COLLECTIVE_HANDOVERS
// One more form a program may ask for: HY4_COLLECTIVE_HANDOVERS (a define; without it this file is the text it was) and
// the first argument with bit 29 set (READY_HAS_HANDOVERS; with bits 31 and 30 only): one more argument follows the
// served word's address, the address of a persistent, host-zeroed table on this core, four words per op, the "hand-over
// words", and bits 16..23 of an op's routes word say how many MORE times than once the op is handed over in the
// program. Whoever hands op k over writes, BEFORE it adds the ready bit:
//     word 4 k        1 << 31 | the payload bytes of this hand-over (0 .. the op's bytes)
//     word 4 k + 1    a byte offset added to the op's landing slot for this hand-over
//     word 4 k + 2    a byte offset added to the op's staging address for this hand-over
//     word 4 k + 3    NOT the hander's: the sender adds 1 to it for every hand-over of the op it has closed (a
//                     count for tests: with the ready and served words at zero it shows that every hand-over was
//                     taken exactly once)
// A first word without bit 31 means the op's own bytes, slot and staging. A hand-over of 0 bytes sends nothing, copies
// nothing and counts no arrival anywhere: it only ends that hand-over of the op (a payload none of whose rows is
// live). After every hand-over the sender takes the op's ready bit off the ready word itself, then adds the
// op's bit to the served word; the next hand-over of the same op may be made only after its served bit was seen
// and taken off (so neither word ever gets a second add of the same bit). The kernel ends when every op has had
// all its hand-overs.
constexpr uint32_t READY_HAS_HANDOVERS = 1u << 29;
constexpr uint32_t OP_MORE_HANDOVERS_SHIFT = 16;
constexpr uint32_t HANDOVER_VALID = 1u << 31;
#endif

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
    uint32_t own_slot;    // 0: the payload goes to the routes only (OP_NO_OWN_SLOT)
    uint32_t route_args;  // index of this op's first route index argument
#ifdef HY4_COLLECTIVE_HANDOVERS
    uint32_t handovers;  // how many times the op is handed over in the program (1 unless the table says more)
#endif
};

}  // namespace

void kernel_main() {
    [[maybe_unused]] uint32_t stamps[5] = {hy4_collective::now(), 0, 0, 0, 0};  // start, open, ready, sent, closed
    size_t arg = 0;
    const uint32_t ready_word = get_arg_val<uint32_t>(arg++);
    const uint32_t num_ops = get_arg_val<uint32_t>(arg++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg++);
    const uint32_t routes_given = get_arg_val<uint32_t>(arg++);
    const bool has_served = (ready_word & READY_IS_ADDRESS) != 0 && (ready_word & READY_HAS_SERVED) != 0;
    const uint32_t served_address = has_served ? get_arg_val<uint32_t>(arg++) : 0;
#ifdef HY4_COLLECTIVE_HANDOVERS
    const bool has_handovers = has_served && (ready_word & READY_HAS_HANDOVERS) != 0;
    const uint32_t handover_table = has_handovers ? get_arg_val<uint32_t>(arg++) : 0;
    const uint32_t ready_id = has_served ? (ready_word & ~(READY_HAS_SERVED | READY_HAS_HANDOVERS)) : ready_word;
#else
    const uint32_t ready_id = has_served ? (ready_word & ~READY_HAS_SERVED) : ready_word;
#endif
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
        const uint32_t routes_word = get_arg_val<uint32_t>(arg++);
        op.num_routes = routes_word & OP_ROUTES_MASK;
        op.own_slot = (routes_word & OP_NO_OWN_SLOT) ? 0u : 1u;
#ifdef HY4_COLLECTIVE_HANDOVERS
        op.handovers = has_handovers ? ((routes_word >> OP_MORE_HANDOVERS_SHIFT) & 0xFFu) + 1u : 1u;
#endif
        op.route_args = arg;
        arg += op.num_routes;
    }

#ifdef HY4_COLLECTIVE_PARSE_ONLY
    {
        volatile tt_l1_ptr uint32_t* page = hy4_collective::word(HY4_COLLECTIVE_PARSE_PAGE);
        uint32_t n = 0;
        page[n++] = 0x50415253u;
        page[n++] = ready_id;
#ifdef HY4_COLLECTIVE_HANDOVERS
        page[n++] = (has_served ? 1u : 0u) | (has_handovers ? 2u : 0u);
        page[n++] = served_address;
        if (has_handovers) {  // (two more kinds of word in the page: this one, and each op's hand-overs)
            page[n++] = handover_table;
        }
#else
        page[n++] = has_served ? 1u : 0u;
        page[n++] = served_address;
#endif
        page[n++] = num_ops;
        page[n++] = num_connections;
        page[n++] = routes_given;
        for (uint32_t r = 0; r < routes_given; ++r) {
            for (uint32_t i = 0; i < ROUTE_ARGS; ++i) {
                page[n++] = get_arg_val<uint32_t>(route_table + r * ROUTE_ARGS + i);
            }
        }
        for (uint32_t k = 0; k < num_ops; ++k) {
            const Op& op = ops[k];
            page[n++] = op.staging;
            page[n++] = op.bytes;
            page[n++] = op.chunk;
            page[n++] = op.landing_xy;
            page[n++] = op.slot;
            page[n++] = op.parity_stride;
            page[n++] = op.ctrl;
            page[n++] = op.num_routes;
            page[n++] = op.own_slot;
#ifdef HY4_COLLECTIVE_HANDOVERS
            if (has_handovers) {
                page[n++] = op.handovers;
            }
#endif
            for (uint32_t i = 0; i < op.num_routes; ++i) {
                page[n++] = get_arg_val<uint32_t>(op.route_args + i);
            }
        }
        page[n++] = static_cast<uint32_t>(arg);
        page[n++] = 0x454E4421u;
        return;
    }
#endif
    WAYPOINT("MKSO");
    auto fabric = RoutingPlaneConnectionManager::build_from_args<
        RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg, num_connections);

    // One packet header per route, routed once: the routes do not change over the program.
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[header_slots];
    uint8_t route_connection[header_slots];
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

    const bool ready_is_address = (ready_id & READY_IS_ADDRESS) != 0;
    volatile tt_l1_ptr uint32_t* ready =
        hy4_collective::word(ready_is_address ? (ready_id & ~READY_IS_ADDRESS) : get_semaphore(ready_id));
    uint32_t served_bits = 0;
    const uint32_t all_ops = num_ops >= 32 ? 0xFFFFFFFFu : ((1u << num_ops) - 1u);
    uint32_t pending = all_ops;
#ifdef HY4_COLLECTIVE_HANDOVERS
    uint32_t left[MAX_OPS];  // hand-overs each op still has to get
    for (uint32_t k = 0; k < num_ops; ++k) {
        left[k] = ops[k].handovers;
    }
#endif
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
#ifdef HY4_COLLECTIVE_HANDOVERS
            // This hand-over's bytes and landing offset. They were written before the ready bit this pass saw was
            // added, and the cache was invalidated before that bit was read.
            Op now = ops[k];
            if (has_handovers) {
                volatile tt_l1_ptr uint32_t* words = hy4_collective::word(handover_table + 16 * k);
                const uint32_t first = words[0];
                if (first & HANDOVER_VALID) {
                    const uint32_t bytes = first & ~HANDOVER_VALID;
                    now.bytes = bytes < now.bytes ? bytes : now.bytes;
                    now.slot += words[1];
                    now.staging += words[2];
                }
            }
            const Op& op = now;
#else
            const Op& op = ops[k];
#endif
            if (pending == all_ops) {
                stamps[2] = hy4_collective::now();
            }
            const uint32_t parity = (bits >> (2 * k + 1)) & 1u;
            const uint32_t lx = op.landing_xy & 0xFFFF;
            const uint32_t ly = op.landing_xy >> 16;
            const uint32_t slot = op.slot + parity * op.parity_stride;
            const uint64_t counter = get_noc_addr(lx, ly, op.ctrl + hy4_collective::CTRL_COUNTER + 16 * parity);

#ifdef HY4_COLLECTIVE_HANDOVERS
            if (op.bytes != 0) {  // (a hand-over of 0 bytes: nothing is copied, sent or counted)
#endif
                // This chip's own slot.
                if (op.own_slot) {
                    noc_async_write(op.staging, get_noc_addr(lx, ly, slot), op.bytes);
                }

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
                if (op.own_slot) {
                    noc_semaphore_inc(counter, 1);
                }
#ifdef HY4_COLLECTIVE_HANDOVERS
            }
            if (has_handovers) {
                // The op's ready bit comes off the word now, before its served bit is added: whoever hands the op
                // over again adds the ready bit only after it has seen the served bit.
                noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(ready)), 0u - (1u << (2 * k + parity)));
                noc_async_atomic_barrier();
                if (--left[k] == 0) {
                    pending &= ~(1u << k);
                }
                volatile tt_l1_ptr uint32_t* words = hy4_collective::word(handover_table + 16 * k);
                words[3] = words[3] + 1;
            } else {
                served_bits += 1u << (2 * k + parity);
                pending &= ~(1u << k);
            }
#else
            served_bits += 1u << (2 * k + parity);
            pending &= ~(1u << k);
#endif
            if (has_served) {
                noc_semaphore_inc(get_noc_addr(served_address), 1u << k);
            }
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
    if (ready_is_address && served_bits != 0) {
        // A persistent ready word: what was served comes off it (an atomic of this core on itself: a hand-over
        // of a later op may be adding to the word at this moment when a wait ran out).
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(ready)), 0u - served_bits);
        noc_async_atomic_barrier();
    }
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
