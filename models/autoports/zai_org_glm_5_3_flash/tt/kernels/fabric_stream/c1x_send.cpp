// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C1 / C3 on the 2x4 mesh (tt/fabric_stream.FabricStreamC1X), lane BRISC. Lane h of every chip
// owns ethernet link h in all four directions (axis 0 = the board pair up / down, axis 1 = the 4-chip row up / down).
// Per launch g:
//   0. (PMK H0) wait this chip's member XS-FREE credits (XSFREE >= g x xsfree_n; 0 = skip);
//   1. credit every chip that writes into this chip's XS: the partner (axis 0) and the row peers (axis 1 line
//      multicast) get +1 on CREDIT + me (me = this chip's mesh index); wait CREDIT + id >= g for each of them;
//   2. OWN blocks: 4-KiB packets into the same holder XS on the partner (axis 0) AND on every row peer (axis 1);
//      PARTNER blocks (landed here over axis 0, polled by c1x_read.cpp): the same packets on every row peer only.
//   Each producer slice (SLICE_T tiles) ends with a fused write + atomic +1 of the holder's arrive semaphore (address
//   in SEMADDR, the same on every chip) on each destination, so every holder sees SLICES per block, local or remote.
// After the two stages every chip holds all 64 blocks: own (local producers), partner (axis 0), and the row peers' own
// + partner blocks (axis 1).
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t SLICE_T = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t TB = 2048;
constexpr uint32_t PKT = 2 * TB;
constexpr uint32_t NHDR = 4;
static_assert(TPB % NB == 0 && NB % 2 == 0 && SLICE_T % 2 == 0, "packets never straddle a batch or a slice");

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct Dir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[NHDR];
    volatile tt_l1_ptr Hdr* credit;
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    PacketHeaderPool::reset();  // a composite runs several fabric phases on this RISC
#if defined(FS_HDR_CB)
    // the DATA headers live in a private L1 CB of this core (FS_HDR_CB, NHDR x directions x PACKET_HEADER_MAX_SIZE
    // bytes): the per-RISC PacketHeaderPool holds 12 headers (dev_mem_map.h: 144 B x 24 / 2 RISCs) and 3 active
    // directions x (NHDR 4 + 1 credit) = 15 overflowed it on the 2x4 (homelab-1 00:25Z, BRISC assert
    // packet_header_pool.h:51); the pool keeps only the credit headers (<= 1 per direction)
    const uint32_t fs_hdr_base = get_write_ptr(FS_HDR_CB);
    uint32_t fs_hdr_n = 0;
    auto fs_hdr_alloc = [&]() {
        return reinterpret_cast<volatile tt_l1_ptr Hdr*>(fs_hdr_base + (fs_hdr_n++) * PACKET_HEADER_MAX_SIZE);
    };
#else
    auto fs_hdr_alloc = []() { return reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header()); };
#endif
    size_t a = ph_rt0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t my_x = get_arg_val<uint32_t>(a++);
    const uint32_t my_y = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);  // this chip's mesh index (r C + c): its CREDIT word on the peers
    uint32_t ranges[4];
    for (uint32_t d = 0; d < 4; d++) {
        ranges[d] = get_arg_val<uint32_t>(a++);  // axis-0 up, axis-0 down, axis-1 up, axis-1 down (hops)
    }
    const uint32_t xsfree_n = get_arg_val<uint32_t>(a++);
    const uint32_t n_wait = get_arg_val<uint32_t>(a++);
    const uint32_t wait_ids = a;  // the chips that write into this chip's XS (their CREDIT words here)
    a += n_wait;
    const uint32_t n_own = get_arg_val<uint32_t>(a++);
    const uint32_t n_par = get_arg_val<uint32_t>(a++);
    const uint32_t blk_xy = a;  // (n_own + n_par) x holder NoC 0 (x, y)
    a += 2 * (n_own + n_par);
#if defined(FS_C1X_REL_SEM)  // the core that holds this lane's axis-1 link next in the same program (its NoC 0 x, y)
    const uint32_t rel_x = get_arg_val<uint32_t>(a++);
    const uint32_t rel_y = get_arg_val<uint32_t>(a++);
#endif
#if defined(FS_C1X_ACQ_SEM)  // W7 P11: RS8 stage 2's lane h (a tail) held link h in between and released it back
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FS_C1X_ACQ_SEM)), FS_C1X_ACQ_N);
#endif
    Dir dirs[4];
    a = get_arg_val<uint32_t>(a);  // fab_at: the ABSOLUTE index of the fabric args (shared by a composite's phases)
    for (uint32_t d = 0; d < 4; d++) {
        dirs[d].range = ranges[d];
        if (ranges[d]) {
            dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        }
    }
    for (uint32_t d = 0; d < 4; d++) {
        if (dirs[d].range) {
            dirs[d].conn.open();
            for (uint32_t k = 0; k < NHDR; k++) {
                dirs[d].data[k] = fs_hdr_alloc();
                dirs[d].data[k]->to_chip_multicast(
                    MulticastRoutingCommandHeader{1, static_cast<uint8_t>(dirs[d].range)});
            }
            dirs[d].credit = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
            dirs[d].credit->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(dirs[d].range)});
            dirs[d].credit->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                safe_get_noc_addr(my_x, my_y, sync + (fs::CREDIT + me) * fs::WORD, 0), 1});
        }
    }
#if !defined(FS_NO_ROW_BARRIER)
    volatile tt_l1_ptr Hdr* tok = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
#endif
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN_B);
    const uint32_t g = *gen + 1;
    if (xsfree_n) {
        fs::wait_ge(fs::word(sync, fs::XSFREE), g * xsfree_n);
    }
    for (uint32_t d = 0; d < 4; d++) {
        if (dirs[d].range) {
            dirs[d].conn.wait_for_empty_write_slot();
            dirs[d].conn.send_payload_flush_non_blocking_from_address((uint32_t)dirs[d].credit, sizeof(Hdr));
        }
    }
    for (uint32_t i = 0; i < n_wait; i++) {
        fs::wait_ge(fs::word(sync, fs::CREDIT + get_arg_val<uint32_t>(wait_ids + i)), g);
    }
    auto send = [&](Dir& d, uint32_t src, uint64_t dst, uint64_t sem, bool signal) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();
            d.next = 0;
        }
        volatile tt_l1_ptr Hdr* h = d.data[d.next++];
        d.conn.wait_for_empty_write_slot();
        if (signal) {
            h->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, PKT);
        } else {
            h->to_noc_unicast_write(NocUnicastCommandHeader{dst}, PKT);
        }
        d.conn.send_payload_without_header_non_blocking_from_address(src, PKT);
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    };
    for (uint32_t k = 0; k < n_own + n_par; k++) {
        const uint32_t hx = get_arg_val<uint32_t>(blk_xy + 2 * k), hy = get_arg_val<uint32_t>(blk_xy + 2 * k + 1);
        const uint32_t d0 = k < n_own ? 0 : 2;  // own blocks: both axes; the partner's: the row only
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_wait_front(CB_STAGE, NB);
            const uint32_t buf = get_read_ptr(CB_STAGE);
            // THE RING-ADDRESS RULE (fs_common.hpp): the holders' FS_ARRIVE = this lane's own; the published SEMADDR
            // only tells that XSADDR (published before it) is this launch's
            const uint32_t own_sem = get_semaphore(fs::SEM_ARRIVE_ID);
            fs::wait_eq(fs::word(sync, fs::SEMADDR), own_sem);
            const uint32_t xs = *fs::word(sync, fs::XSADDR);
            const uint64_t sem = safe_get_noc_addr(hx, hy, own_sem, 0);
            for (uint32_t i = 0; i < NB; i += 2) {
                const uint32_t t = b * NB + i;
                const uint64_t dst = safe_get_noc_addr(hx, hy, xs + t * TB, 0);
                const bool last = (t + 2) % SLICE_T == 0;
                for (uint32_t d = d0; d < 4; d++) {
                    if (dirs[d].range) {
                        send(dirs[d], buf + i * TB, dst, sem, last);
                    }
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(CB_STAGE, NB);
        }
    }
#if !defined(FS_NO_ROW_BARRIER)
    {  // THE PER-HOP DRAIN (fs_common.hpp): the row barrier of 1-hop tokens on this lane's own axis-1 connections
        auto token = [&](Dir& d, uint32_t word) {
            noc_async_writes_flushed();
            tok->to_chip_unicast(1);
            tok->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(my_x, my_y, sync + word * fs::WORD, 0), 1});
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)tok, sizeof(Hdr));
            noc_async_writes_flushed();
        };
        if (dirs[3].range || dirs[2].range) {
            fs::wait_ge(fs::word(sync, fs::LANDED), g);  // this lane's row-incoming packets all landed on this chip
            if (dirs[3].range) {
                fs::wait_ge(fs::word(sync, fs::BAR_L), g);  // chips below have landed everything
            }
            if (dirs[2].range) {
                token(dirs[2], fs::BAR_L);
                fs::wait_ge(fs::word(sync, fs::BAR_R), g);  // chips above have landed everything
            }
            if (dirs[3].range) {
                token(dirs[3], fs::BAR_R);
            }
        }
    }
#endif
    *gen = g;
    for (uint32_t d = 0; d < 4; d++) {
        if (dirs[d].range) {
            fs::drain_close(dirs[d].conn);
        }
    }
#if defined(FS_C1X_REL_SEM)
    // THE LINK HAND-OVER (fused A2F at TP8, fuse8 hang 2026-10-24 02:36Z): RS8 stage 2's lane h (rsl_send, a tail core)
    // sends on the same link h axis-1 EDM sender channels; tt-fabric serves one worker per channel at a time, so it
    // opens only after this +1 (FS_RSL_AFTER_SEM), i.e. after every connection here is drain-closed
    noc_semaphore_inc(safe_get_noc_addr(rel_x, rel_y, get_semaphore(FS_C1X_REL_SEM), 0), 1);
#endif
    noc_async_full_barrier();
}
