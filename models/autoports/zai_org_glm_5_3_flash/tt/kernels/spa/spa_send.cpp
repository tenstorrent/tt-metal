// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention, the union GATHER (tt/spa_gather.py), lane BRISC: per staged entry (4 latent rows, dst_slot,
// dst_pos) the 4 rows go to compact rows 4 dst_slot .. + 3 of chip dst_pos: a local write (dst_pos = p) or ONE fabric
// scatter packet of 4 x 1 KiB on the connection towards dst_pos (to_chip_unicast(hops)). After its entries this chip
// sends every peer ONE atomic increment of the peer's arrival word ARR + p (flush: every earlier write of that
// connection landed); then it waits ARR + q >= GEN + 1 for every peer q (each peer finished writing into this chip's
// compact buffer) and bumps GEN. Packets of one connection arrive in order. Compile-time: CB_STAGE, CB_META, then the
// compact tensor's TensorAccessorArgs. Runtime: sync (lane L1 words), p, n chips on the line, up range, down range,
// compact address, entries, then the fabric connections (up first, then down).
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

constexpr uint32_t CB_STAGE = get_compile_time_arg_val(0);
constexpr uint32_t CB_META = get_compile_time_arg_val(1);
constexpr uint32_t RB = 1024;
constexpr uint32_t NHDR = 4;
constexpr uint32_t GEN = 0, ARR = 1;  // sync words (4 B each): GEN, ARR + source position

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct Dir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[NHDR];
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr auto y_args = TensorAccessorArgs<2>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t nchips = get_arg_val<uint32_t>(a++);
    const uint32_t up = get_arg_val<uint32_t>(a++);
    const uint32_t dn = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const auto y = TensorAccessor(y_args, y_addr);
    const uint32_t ranges[2] = {up, dn};
    Dir dirs[2];
    for (uint32_t d = 0; d < 2; d++) {
        dirs[d].range = ranges[d];
        if (ranges[d]) {
            dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        }
    }
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            dirs[d].conn.open();
            for (uint32_t k = 0; k < NHDR; k++) {
                dirs[d].data[k] = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
            }
        }
    }
    auto hdr = [&](Dir& d) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();  // the ring wraps: the headers' previous packets have left L1
            d.next = 0;
        }
        return d.data[d.next++];
    };
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(CB_META, 1);
        cb_wait_front(CB_STAGE, 1);
        volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_META));
        const uint32_t slot = mt[0], pos = mt[1];
        const uint32_t src = get_read_ptr(CB_STAGE);
        if (pos == p) {
            for (uint32_t j = 0; j < 4; ++j) {
                noc_async_write_page(4 * slot + j, y, src + j * RB);
            }
        } else {
            Dir& d = dirs[pos > p ? 0 : 1];
            const uint32_t hops = pos > p ? pos - p : p - pos;
            volatile tt_l1_ptr Hdr* h = hdr(d);
            h->to_chip_unicast(static_cast<uint8_t>(hops));
            h->to_noc_unicast_scatter_write(
                NocUnicastScatterCommandHeader(
                    {y.get_noc_addr(4 * slot, 0, 0),
                     y.get_noc_addr(4 * slot + 1, 0, 0),
                     y.get_noc_addr(4 * slot + 2, 0, 0),
                     y.get_noc_addr(4 * slot + 3, 0, 0)},
                    {static_cast<uint16_t>(RB), static_cast<uint16_t>(RB), static_cast<uint16_t>(RB)}),
                4 * RB);
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_without_header_non_blocking_from_address(src, 4 * RB);
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
        }
        noc_async_writes_flushed();  // the staged rows have left (local writes and packet payload)
        cb_pop_front(CB_STAGE, 1);
        cb_pop_front(CB_META, 1);
    }
    noc_async_write_barrier();
    // one arrival increment per peer (flush: the peer's writes from this chip landed first)
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q == p) {
            continue;
        }
        Dir& d = dirs[q > p ? 0 : 1];
        const uint32_t hops = q > p ? q - p : p - q;
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>(hops));
        h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], sync + (ARR + p) * 4, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    }
    noc_async_writes_flushed();
    // the waiter: every peer's rows landed in this chip's compact buffer
    volatile tt_l1_ptr uint32_t* gen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + GEN * 4);
    const uint32_t g = *gen + 1;
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q != p) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + (ARR + q) * 4);
            while (*w < g) {
                invalidate_l1_cache();
            }
        }
    }
    *gen = g;
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {  // drain (every slot free again) then close, as fs_common.hpp drain_close
            while (dirs[d].conn.get_num_free_write_slots() != dirs[d].conn.num_buffers_per_channel) {
            }
            dirs[d].conn.close();
        }
    }
    noc_async_full_barrier();
}
