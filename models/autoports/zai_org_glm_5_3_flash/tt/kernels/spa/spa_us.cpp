// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2, the device-side UNION + gather (tt/spa_union.py), lane BRISC (spa_send.cpp's v1 protocol
// plus the bitmap exchange): after member 0 delivered this chip's union bitmap (S_LOCAL), its bitmap slot goes to every
// peer's slot of this chip's position (unicast writes, <= 4 KiB per packet) + one atomic increment of the peer's ARR_BM
// + p (flush: the bitmap landed first). A peer starts sending into this chip's compact buffer only after it has this
// chip's bitmap of the SAME launch, so the exchange is also the handshake that this chip left its previous launch (and
// the attention that read the compact buffer). Then per staged pool (spa_ur.cpp) the 4 rows go to compact rows 4 slot
// .. + 3 of chip pos: a local write or ONE 4 x 1 KiB scatter packet; the end marker (slot SKIP) stops the loop. Then
// one atomic increment of every peer's ARR_DATA + p, the wait ARR_DATA + q >= GEN + 1 for every peer, and GEN + 1. The
// 2x4 (tt/spa_union.py, two stages) runs this kernel on BOTH lanes: the row lane (bitmap slot p -> the row-mates' slot
// p, then the stage-B pools into their compact buffers) and the column lane (its row's OR, slot 0 -> the column
// partner's slot 1, then the stage-A pools into the partner's relay rows; `extra` = one more increment after the data:
// the partner's ROW lane word ARR_DATA_A, which its stage-B walk waits for before reading relay rows).
// The bitmap slots are a CB of this program (CB_SLOTS: same L1 address on every chip, live only while the program
// runs): at its start every lane sends each peer one increment of the peer's RDY + p (its slots are live for this
// launch), and writes a peer's slot only after the peer's RDY of the same generation -- no persistent L1 beyond the
// sync words. Compile-time: CB_STAGE, CB_META, WORDS, CB_SLOTS, then the destination tensor's TensorAccessorArgs.
// Runtime: sync, the bitmap source slot, the destination slot (on the peers), the semaphore id to wait for before the
// bitmap send, p, n, up range, down range, destination address, extra x, extra y, extra address (0: none), then the
// fabric connections (up first, then down).
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_stage.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

constexpr uint32_t CB_STAGE = get_compile_time_arg_val(0);
constexpr uint32_t CB_META = get_compile_time_arg_val(1);
constexpr uint32_t WORDS = get_compile_time_arg_val(2);
constexpr uint32_t CB_SLOTS = get_compile_time_arg_val(3);
constexpr uint32_t RB = 1024;
constexpr uint32_t NHDR = 4;
constexpr uint32_t PKT = 4096;  // bitmap bytes per packet
constexpr uint32_t SKIP = 0xFFFFFFFFu;
constexpr uint32_t GEN = 0,
                   ARR_BM = 1;  // sync words: GEN, ARR_BM + source position, ARR_DATA = 1 + n + source position

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct Dir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[NHDR];
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr auto y_args = TensorAccessorArgs<4>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    cb_reserve_back(CB_SLOTS, 1);
    const uint32_t slots = get_write_ptr(CB_SLOTS);
    const uint32_t bm_src = slots + get_arg_val<uint32_t>(a++) * WORDS * 4;
    const uint32_t bm_dst = slots + get_arg_val<uint32_t>(a++) * WORDS * 4;
    const uint32_t wait_sem = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t nchips = get_arg_val<uint32_t>(a++);
    const uint32_t up = get_arg_val<uint32_t>(a++);
    const uint32_t dn = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t extra_x = get_arg_val<uint32_t>(a++);
    const uint32_t extra_y = get_arg_val<uint32_t>(a++);
    const uint32_t extra_addr = get_arg_val<uint32_t>(a++);
    const auto y = TensorAccessor(y_args, y_addr);
    const uint32_t ARR_DATA = 1 + nchips, RDY = 3 + 2 * nchips;
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* gen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + GEN * 4);
    const uint32_t g = *gen + 1;
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
    // RDY: this lane's slots are live for this launch (one increment of every peer's RDY + p)
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q == p) {
            continue;
        }
        Dir& d = dirs[q > p ? 0 : 1];
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>(q > p ? q - p : p - q));
        h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], sync + (RDY + p) * 4, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    }
    // the bitmap exchange (this chip's slot -> every peer's slot of position p, once the peer's slots are live) + one
    // ARR_BM increment per peer
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(wait_sem)), 1);
    const uint32_t own = bm_src;
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q == p) {
            continue;
        }
        Dir& d = dirs[q > p ? 0 : 1];
        const uint32_t hops = q > p ? q - p : p - q;
        volatile tt_l1_ptr uint32_t* rdy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + (RDY + q) * 4);
        while (*rdy < g) {
            invalidate_l1_cache();
        }
        for (uint32_t o = 0; o < WORDS * 4; o += PKT) {
            const uint32_t len = WORDS * 4 - o < PKT ? WORDS * 4 - o : PKT;
            volatile tt_l1_ptr Hdr* h = hdr(d);
            h->to_chip_unicast(static_cast<uint8_t>(hops));
            h->to_noc_unicast_write(
                NocUnicastCommandHeader{safe_get_noc_addr(my_x[noc_index], my_y[noc_index], bm_dst + o, 0)}, len);
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_without_header_non_blocking_from_address(own + o, len);
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
        }
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>(hops));
        h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], sync + (ARR_BM + p) * 4, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    }
    for (bool done = false; !done;) {  // one staged batch of spa::NB pools per pass (spa_stage.hpp)
        uint32_t k = 0;
        for (; k < spa::NB; ++k) {
            cb_wait_front(CB_META, k + 1);
            volatile tt_l1_ptr uint32_t* mt =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_META) + k * spa::MB);
            const uint32_t slot = mt[0], pos = mt[1];
            if (slot == spa::SKIP) {  // the end marker: pushed alone after a closed batch (k == 0)
                done = true;
                break;
            }
            if (slot == spa::DUMMY) {
                continue;
            }
            cb_wait_front(CB_STAGE, k + 1);
            const uint32_t src = get_read_ptr(CB_STAGE) + k * spa::PB;
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
        }
        if (done) {
            cb_pop_front(CB_META, 1);
            break;
        }
        noc_async_writes_flushed();        // the batch's staged rows have left (local writes and packet payloads)
        cb_wait_front(CB_STAGE, spa::NB);  // (a batch's DUMMY pages too: pushed with it)
        cb_pop_front(CB_STAGE, spa::NB);
        cb_pop_front(CB_META, spa::NB);
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
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], sync + (ARR_DATA + p) * 4, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
        if (extra_addr !=
            0) {  // the partner's row lane: its relay rows from this chip landed (same route: after the data)
            volatile tt_l1_ptr Hdr* h2 = hdr(d);
            h2->to_chip_unicast(static_cast<uint8_t>(hops));
            h2->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(extra_x, extra_y, extra_addr, 0), 1});
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h2, sizeof(Hdr));
        }
    }
    noc_async_writes_flushed();
    // the waiter: every peer's rows landed in this chip's compact buffer
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q != p) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + (ARR_DATA + q) * 4);
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
