// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Column stage of the cross-chip log-sum-exp merge (fused/merges/lse_merge.py): the sender core of one chip.
//
// The chip in row r of its column holds, per user, one partial per head group (sparse_mla's output). Head group g
// is merged on the chip of row g of the same column, so for every g != r and every user u this kernel sends the
// partial (u, g) as ONE fabric packet into slot r of the column landing on merge core u of that chip, fused with
// an increment of that core's arrival counter (kernels of fused/collective: collective.hpp is the receiving
// side's protocol; the landing's parity is the round parity of this chip's own merge cores, read at start).
// Nothing is copied to this chip's own landing: its own partial of group r is read by its merge cores.
//
// Standalone form: the partials come from the attention's output tensor in DRAM (page u * 8 + g = O tiles | max /
// sum tile | status); they are read into the staging buffer as [max / sum | O], then sent. In a layer program the
// attention cores would write their partial into the same staging slots and count on a word of this core.
//
// Compile-time args: users, num_routes (= column members - 1), cb_staging, cb_scratch, then TensorAccessorArgs of
// the attention output.
// Runtime args: output buffer address, own slot r, merge core 0 NoC x | y << 16, ctrl address, landing address
// (parity 0, slot 0), parity stride, slot stride, per user the merge core's NoC x | y << 16, num_connections,
// per route: head group (= destination slot), destination chip id, connection index; then the routing plane
// connection manager's arguments (HY4_LSE_DRY: instead, the address of the merge cores' state tile).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifdef HY4_MERGE_NOP
// The launch floor: this core with a kernel that returns at once (no fabric connection is set up for it).
void kernel_main() {}
#else

// HY4_LSE_DRY: everything but the fabric (the DRAM reads into the staging buffer, the parity read), for a run on
// one chip before the kernel ever opens a connection; it leaves the bytes it would send in the staging buffer
// and a checksum of them, the parity and a marker in the first 16 bytes of merge core 0's state tile.
#ifndef HY4_LSE_DRY
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#if !defined(FABRIC_2D)
#error "the column sender routes by chip id: it needs a 2D fabric"
#endif

using namespace tt::tt_fabric;
#endif

#include "collective.hpp"
#include "lse_common.hpp"

using namespace hy4_lse;

constexpr uint32_t users = get_compile_time_arg_val(0);
constexpr uint32_t num_routes = get_compile_time_arg_val(1);
constexpr uint32_t cb_staging = get_compile_time_arg_val(2);
constexpr uint32_t cb_scratch = get_compile_time_arg_val(3);
constexpr uint32_t MAX_USERS = 8;
static_assert(users <= MAX_USERS);
#ifndef HY4_LSE_DRY
static_assert(
    num_routes >= 1 && num_routes <= MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType / PACKET_HEADER_MAX_SIZE,
    "the routes exceed this RISC's share of the packet header pool");
#endif

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<4>();
    size_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t own = get_arg_val<uint32_t>(arg++);
    const uint32_t merge0_xy = get_arg_val<uint32_t>(arg++);
    const uint32_t ctrl = get_arg_val<uint32_t>(arg++);
    const uint32_t landing = get_arg_val<uint32_t>(arg++);
    const uint32_t parity_stride = get_arg_val<uint32_t>(arg++);
    const uint32_t slot_stride = get_arg_val<uint32_t>(arg++);
    uint32_t merge_xy[MAX_USERS];
    for (uint32_t u = 0; u < users; ++u) {
        merge_xy[u] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t num_connections = get_arg_val<uint32_t>(arg++);
    const size_t route_table = arg;
    arg += 3 * num_routes;

    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t staging = get_write_ptr(cb_staging);
    volatile tt_l1_ptr uint32_t* scratch = hy4_collective::word(get_write_ptr(cb_scratch));

    // The partials that leave, into slot (route, user) of the staging buffer, and the round's parity. All of it
    // is in L1 before the first fabric call: from here to the end the sequence is collective_sender.cpp's.
    WAYPOINT("LSRD");
    noc_async_read(
        get_noc_addr(merge0_xy & 0xFFFF, merge0_xy >> 16, ctrl + hy4_collective::CTRL_ROUND),
        get_write_ptr(cb_scratch),
        16);
    for (uint32_t i = 0; i < num_routes; ++i) {
        const uint32_t group = get_arg_val<uint32_t>(route_table + 3 * i);
        for (uint32_t u = 0; u < users; ++u) {
            const uint32_t slot = staging + (i * users + u) * PART_BYTES;
            const uint32_t page = u * 8 + group;
            noc_async_read(out.get_noc_addr(page, O_BYTES), slot, TINY_BYTES);
            noc_async_read(out.get_noc_addr(page, 0), slot + TINY_BYTES, O_BYTES);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t parity = scratch[0] & 1u;
    const uint32_t slot_address = landing + parity * parity_stride + own * slot_stride;

#ifdef HY4_LSE_DRY
    // Report to merge core 0's state tile (words 16..19): marker, parity, slot address, sum of the staged words.
    {
        uint32_t sum = 0;
        const volatile tt_l1_ptr uint32_t* staged = hy4_collective::word(staging);
        for (uint32_t w = 0; w < num_routes * users * PART_BYTES / 4; ++w) {
            sum += staged[w];
        }
        scratch[4] = 0x4C534452;  // "LSDR"
        scratch[5] = parity;
        scratch[6] = slot_address;
        scratch[7] = sum;
        const uint32_t state = get_arg_val<uint32_t>(arg);
        (void)num_connections;
        noc_async_write(
            get_write_ptr(cb_scratch) + 16, get_noc_addr(merge0_xy & 0xFFFF, merge0_xy >> 16, state + 64), 16);
        noc_async_write_barrier();
    }
    WAYPOINT("LSSD");
#else
    WAYPOINT("LSSO");
    auto fabric = RoutingPlaneConnectionManager::build_from_args<
        RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg, num_connections);

    // One packet header per route, routed once.
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[num_routes];
    for (uint32_t i = 0; i < num_routes; ++i) {
        headers[i] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(
            headers[i], static_cast<uint16_t>(get_arg_val<uint32_t>(route_table + 3 * i + 1)), fabric.my_mesh_id);
    }
    fabric.open_finish();

    WAYPOINT("LSSS");
    for (uint32_t u = 0; u < users; ++u) {
        const uint32_t mx = merge_xy[u] & 0xFFFF;
        const uint32_t my = merge_xy[u] >> 16;
        const uint64_t destination = get_noc_addr(mx, my, slot_address);
        const uint64_t counter = get_noc_addr(mx, my, ctrl + hy4_collective::CTRL_COUNTER + 16 * parity);
        for (uint32_t i = 0; i < num_routes; ++i) {
            volatile tt_l1_ptr PACKET_HEADER_TYPE* header = headers[i];
            // The header's previous packet must have left L1 before it changes.
            noc_async_writes_flushed();
            header->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{destination, counter, 1, true}, PART_BYTES);
            auto& connection = fabric.get(static_cast<uint8_t>(get_arg_val<uint32_t>(route_table + 3 * i + 2))).sender;
            connection.wait_for_empty_write_slot();
            connection.send_payload_without_header_non_blocking_from_address(
                staging + (i * users + u) * PART_BYTES, PART_BYTES);
            connection.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
        }
    }

    WAYPOINT("LSSC");
    noc_async_writes_flushed();
    fabric.close();
    noc_async_full_barrier();
    WAYPOINT("LSSD");
#endif  // HY4_LSE_DRY
}

#endif  // HY4_MERGE_NOP
