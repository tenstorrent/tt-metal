// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 14 fabric all-reduce, lane core BRISC = the fabric sender (tt/fabric_ar.py). Lane h of every chip owns
// ethernet link h in every direction it uses (one worker connection per router: one lane per link). Per iteration g:
//   wait BARRIER >= (g-1) * writers  (every chip this lane writes to finished iteration g-1: its landing slots are
//   free) round 1 (mesh axis 0, n1 chips, my position p1; skipped when n1 == 1): wait the producers' packed partial
//     (OWN_READY >= g * n_prod), then send it as fused write + atomic-inc packets into slot p1 of every axis-0 peer
//     (line multicast "up" over up1 hops and "down" over dn1 hops; same L1 address on every chip);
//   round 2 (mesh axis 1, n2 chips, position p2): send the round-1 result (CB_SEND, packed by TRISC, consumed here
//     packet by packet) -- or, without round 1, the producers' partial -- into slot p2 of every axis-1 peer;
//   wait RES_DONE >= g (NCRISC: final result packed), then send the end-of-iteration atomic inc to every chip written
//   to
//   -- with FAR_CREDIT_NEXT (the default; GLM53_MK_FAR_CREDIT_NEXT=0 turns it off): except after the LAST iteration of
//   a launch, whose inc is sent at the START of the next launch.
// Why (Stage 14, 2026-10-20 soak race): in "cb" mode the landing slots are CBs of the layer programs; between two
// launches other ops (e.g. a trace's state-restore copies) run on the lane cores with their own CBs at the same L1
// addresses (the bottom of the CB region). An inc sent at the end of a launch let a peer that was ahead write the
// next launch's data into those addresses while such an op still used them (TP2 B=1: land2[0] = 312448 inside a copy
// op's 8 KB of CBs at 308352 -> chip 1 reduced garbage, ~1 replay in 60k of a back-to-back soak). Sent from the next
// launch, the inc certifies that the landing CBs are reserved again on this core.
#if defined(FAR_CREDIT_NEXT)
constexpr bool CREDIT_NEXT = true;
#else
constexpr bool CREDIT_NEXT = false;
#endif
// Packets of one connection arrive in order and every data packet increments the receiver's per-source counter after
// its write landed (flush), so the receiver tracks per-source progress packet by packet.
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "far_common.hpp"

constexpr uint32_t PT = get_compile_time_arg_val(0);     // pseudo-tiles (2048 B) per lane message
constexpr uint32_t PKT_T = get_compile_time_arg_val(1);  // pseudo-tiles per packet
constexpr uint32_t N1 = get_compile_time_arg_val(2);     // chips along mesh axis 0 (round 1)
constexpr uint32_t N2 = get_compile_time_arg_val(3);     // chips along mesh axis 1 (round 2)
constexpr uint32_t CB_SEND = get_compile_time_arg_val(4);
constexpr uint32_t GRP = get_compile_time_arg_val(5);  // packets per counter increment (plain writes in between)
constexpr uint32_t TB = 2048;
constexpr uint32_t NPKT = (PT + PKT_T - 1) / PKT_T;
constexpr uint32_t NHDR = 2;  // data-header ring per direction: flush only when a header is reused

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct Dir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[NHDR];
    volatile tt_l1_ptr Hdr* flag;
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t own = get_arg_val<uint32_t>(a++);    // the producers' packed partial (same address on every chip)
    const uint32_t land1 = get_arg_val<uint32_t>(a++);  // land1[p1] address = the round-1 destination on the peers
    const uint32_t land2 = get_arg_val<uint32_t>(a++);  // land2[p2] address = the round-2 destination on the peers
    const uint32_t p1 = get_arg_val<uint32_t>(a++);
    const uint32_t up1 = get_arg_val<uint32_t>(a++);
    const uint32_t dn1 = get_arg_val<uint32_t>(a++);
    const uint32_t p2 = get_arg_val<uint32_t>(a++);
    const uint32_t up2 = get_arg_val<uint32_t>(a++);
    const uint32_t dn2 = get_arg_val<uint32_t>(a++);
    const uint32_t lane_x = get_arg_val<uint32_t>(a++);
    const uint32_t lane_y = get_arg_val<uint32_t>(a++);
    const uint32_t n_prod = get_arg_val<uint32_t>(a++);
    const uint32_t iters = get_arg_val<uint32_t>(a++);
#if defined(FAR_LATE_OPEN)  // the DSA shard phase (tt/tp_dsa_shard.py): one worker connection per link at a time
    const uint32_t n_lo = get_arg_val<uint32_t>(a++);
    const uint32_t lo_sem = get_arg_val<uint32_t>(a++);
    const uint32_t lo0 = a;
    a += 2 * n_lo;
#endif
    const uint32_t ranges[4] = {up1, dn1, up2, dn2};
    Dir dirs[4];
    size_t a_conn[4] = {0, 0, 0, 0};  // (FAR_LATE_OPEN: the reopen rebuilds a fresh adapter from the same args)
    for (uint32_t d = 0; d < 4; d++) {
        dirs[d].range = ranges[d];
        if (ranges[d]) {
            a_conn[d] = a;
            dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        }
    }
    const uint64_t t_start = get_timestamp();
    for (uint32_t d = 0; d < 4; d++) {
        if (dirs[d].range) {
            dirs[d].conn.open();
            for (uint32_t k = 0; k < NHDR; k++) {
                dirs[d].data[k] = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
                dirs[d].data[k]->to_chip_multicast(
                    MulticastRoutingCommandHeader{1, static_cast<uint8_t>(dirs[d].range)});
            }
            dirs[d].flag = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
            dirs[d].flag->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(dirs[d].range)});
            dirs[d].flag->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                safe_get_noc_addr(lane_x, lane_y, sync + far::BARRIER * far::WORD, 0), 1});
        }
    }
    const uint32_t writers = (N1 - 1) + (N2 - 1);
    // one data packet: payload from local L1 `src`, landing at `dst` (same address on every receiver); every GRP-th
    // packet (and the last) also increments the receiver's counter word `cnt` after the write (and every earlier
    // write of the channel) landed (fused write + atomic inc, flush) -- the other packets are plain writes, so the
    // receiving router does not wait for a write acknowledgement per packet
    auto send = [&](Dir& d, uint32_t src, uint32_t dst, uint32_t len, uint32_t cnt, bool signal) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();  // the ring wraps: the headers' previous packets have left L1
            d.next = 0;
        }
        volatile tt_l1_ptr Hdr* h = d.data[d.next++];
        d.conn.wait_for_empty_write_slot();
        if (signal) {
            h->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{
                    safe_get_noc_addr(lane_x, lane_y, dst, 0),
                    safe_get_noc_addr(lane_x, lane_y, sync + cnt * far::WORD, 0),
                    1,
                    true},
                len);
        } else {
            h->to_noc_unicast_write(NocUnicastCommandHeader{safe_get_noc_addr(lane_x, lane_y, dst, 0)}, len);
        }
        d.conn.send_payload_without_header_non_blocking_from_address(src, len);
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    };
    auto is_signal = [](uint32_t i) { return (i + 1) % GRP == 0 || i + 1 == NPKT; };
    volatile tt_l1_ptr uint32_t* gen = far::word(sync, far::GEN_BRISC);
    uint32_t g = *gen;
    // the previous launch's last end-of-iteration inc (deferred: this launch's landing CBs are now reserved)
    auto send_barrier = [&]() {
        for (uint32_t d = 0; d < 4; d++) {
            if (dirs[d].range) {
                dirs[d].next = 0;
                dirs[d].conn.wait_for_empty_write_slot();
                dirs[d].conn.send_payload_flush_non_blocking_from_address((uint32_t)dirs[d].flag, sizeof(Hdr));
            }
        }
    };
    if (CREDIT_NEXT && g > 0) {
        send_barrier();
    }
#if defined(FAR_LATE_OPEN)
    // release the links until this lane's own round: the stream lanes (fs_send.cpp FS_AFTER_SEM) open theirs only
    // after this signal, and they have closed again before the producers of this phase can be ready (causal chain)
    noc_async_writes_flushed();
    for (uint32_t d = 0; d < 4; d++) {
        if (dirs[d].range) {
            far::drain_close(dirs[d].conn);
        }
    }
    for (uint32_t k = 0; k < n_lo; ++k) {
        const uint32_t x = get_arg_val<uint32_t>(lo0 + 2 * k), yy = get_arg_val<uint32_t>(lo0 + 2 * k + 1);
        noc_semaphore_inc(safe_get_noc_addr(x, yy, get_semaphore(lo_sem), 0), 1);
    }
    noc_async_atomic_barrier();
    bool opened = false;
#endif
    for (uint32_t it = 0; it < iters; it++) {
        g++;
        far::wait_ge(far::word(sync, far::BARRIER), (g - 1) * writers);
        far::wait_ge(far::word(sync, far::OWN_READY), g * n_prod);
#if defined(FAR_LATE_OPEN)
        if (!opened) {
            for (uint32_t d = 0; d < 4; d++) {
                if (dirs[d].range) {
                    size_t ai = a_conn[d];
                    dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(ai);
                    dirs[d].conn.open();
                    dirs[d].next = 0;
                }
            }
            opened = true;
        }
#endif
        if (it == 0) {
            far::probe(sync, far::PROBE + 0, t_start);
            far::probe(sync, far::PROBE + 1, get_timestamp());
        } else if (it == 1) {
            far::probe(sync, far::PROBE + 6, get_timestamp());  // the 2nd all-reduce of the launch: inputs ready
        }
        if constexpr (N1 > 1) {
            for (uint32_t i = 0; i < NPKT; i++) {
                const uint32_t off = i * PKT_T * TB;
                const uint32_t len = (i + 1 < NPKT ? PKT_T : PT - i * PKT_T) * TB;
                for (uint32_t d = 0; d < 2; d++) {
                    if (dirs[d].range) {
                        send(dirs[d], own + off, land1 + off, len, far::ARR1 + p1, is_signal(i));
                    }
                }
            }
        }
        if (it == 0) {
            far::probe(sync, far::PROBE + 2, get_timestamp());
        }
        if constexpr (N2 > 1) {
            for (uint32_t i = 0; i < NPKT; i++) {
                const uint32_t off = i * PKT_T * TB;
                const uint32_t nt = i + 1 < NPKT ? PKT_T : PT - i * PKT_T;
                uint32_t src = own + off;
                if constexpr (N1 > 1) {
                    cb_wait_front(CB_SEND, nt);
                    src = get_read_ptr(CB_SEND);
                }
                for (uint32_t d = 2; d < 4; d++) {
                    if (dirs[d].range) {
                        send(dirs[d], src, land2 + off, nt * TB, far::ARR2 + p2, is_signal(i));
                    }
                }
                if constexpr (N1 > 1) {
                    noc_async_writes_flushed();  // the payload reads of these CB pages are done before TRISC reuses
                                                 // them
                    cb_pop_front(CB_SEND, nt);
                }
            }
        }
        if (it == 0) {
            far::probe(sync, far::PROBE + 3, get_timestamp());
        }
        far::wait_ge(far::word(sync, far::RES_DONE), g);
        if (it == 0) {
            far::probe(sync, far::PROBE + 4, get_timestamp());
        } else if (it == 1) {
            far::probe(sync, far::PROBE + 7, get_timestamp());  // the 2nd all-reduce: result complete
        }
        noc_async_writes_flushed();  // (also: the flag header's previous send has left L1)
        if (!CREDIT_NEXT || it + 1 < iters) {
            send_barrier();
        }
        noc_async_writes_flushed();
    }
    *gen = g;
    far::probe(sync, far::PROBE + 5, get_timestamp());
#if defined(FAR_LATE_OPEN)
    if (opened) {
        for (uint32_t d = 0; d < 4; d++) {
            if (dirs[d].range) {
                far::drain_close(dirs[d].conn);
            }
        }
    }
#else
    for (uint32_t d = 0; d < 4; d++) {
        if (dirs[d].range) {
            far::drain_close(dirs[d].conn);
        }
    }
#endif
    noc_async_full_barrier();
}
