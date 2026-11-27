// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream line RS of fp32 partials (tt/fabric_stream.FabricStreamRSL; TP8 stage 2 along the 4-chip row),
// lane BRISC. Lane h of chip `me` on a line of NC chips owns ethernet link h both ways (dir 0 = toward lower positions,
// dir 1 = higher). Per launch g: credit every peer (+1 CREDIT + me, line multicast both ways), wait CREDIT + c >= g for
// every peer c (their programs run: the landing buffers are free). Then for each (destination c, block) of its list:
// the block's TPB fp32 tiles from CB_SEND, one tile per packet (4 KiB), unicast |c - me| hops toward c into the
// destination's landing buffer r2 (row = this source's slot there x B2 + b); the block's last packet is a fused write +
// atomic +1 of the destination lane h's ARR + slot x LANES + h (flush: the block landed, in order per source and lane).
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
constexpr uint32_t CB_SEND = get_compile_time_arg_val(PH_CT0 + 2);
#if defined(FS_R1_BF16)  // GLM53_PMK_RS8_S2_BF16 (tt/fabric_stream.FabricStreamRSL r1_dtype): r1 / r2 in bf16 tiles
constexpr uint32_t TB32 = 2048;
#else
constexpr uint32_t TB32 = 4096;
#endif
constexpr uint32_t NHDR = 4;
static_assert(TPB % NB == 0, "a batch never straddles a block");

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct LDir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[NHDR];
    volatile tt_l1_ptr Hdr* credit;
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr auto r2_args = TensorAccessorArgs<PH_CT0 + 3>();
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
    const uint32_t my_x = get_arg_val<uint32_t>(a++);  // this lane core, NoC 0 (the peers' lane h is at the same core)
    const uint32_t my_y = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t nc = get_arg_val<uint32_t>(a++);
    const uint32_t r2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t sends = a;  // n x (destination position, row tile in its r2, ARR word there)
    a += 3 * n;
    const auto r2 = TensorAccessor(r2_args, r2_addr);
    const uint32_t ranges[2] = {me, nc - 1 - me};
    LDir dirs[2];
#if defined(FS_RSL_REL_SEM)  // W7: the C1X lane h that takes link h back after this stage (its NoC 0 x, y)
    const uint32_t rel_x = get_arg_val<uint32_t>(a++);
    const uint32_t rel_y = get_arg_val<uint32_t>(a++);
#endif
    a = get_arg_val<uint32_t>(a);  // fab_at: the ABSOLUTE index of the fabric args (shared by a composite's phases)
    for (uint32_t d = 0; d < 2; d++) {
        dirs[d].range = ranges[d];
        if (ranges[d]) {
            dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        }
    }
#if defined(FS_RSL_AFTER_SEM)  // the link hand-over (c1x_send.cpp FS_C1X_REL_SEM): C1X lane h released link h
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FS_RSL_AFTER_SEM)),
#if defined(FS_RSL_AFTER_N)
        FS_RSL_AFTER_N);  // W7: the C1X lane's n-th release in this launch (P15 after P11)
#else
        1);
#endif
#endif
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            dirs[d].conn.open();
            for (uint32_t k = 0; k < NHDR; k++) {
                dirs[d].data[k] = fs_hdr_alloc();
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
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            dirs[d].conn.wait_for_empty_write_slot();
            dirs[d].conn.send_payload_flush_non_blocking_from_address((uint32_t)dirs[d].credit, sizeof(Hdr));
        }
    }
    for (uint32_t c = 0; c < nc; c++) {
        if (c != me) {
            fs::wait_ge(fs::word(sync, fs::CREDIT + c), g);
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t dst_c = get_arg_val<uint32_t>(sends + 3 * i);
        const uint32_t row = get_arg_val<uint32_t>(sends + 3 * i + 1);
        const uint32_t arr = get_arg_val<uint32_t>(sends + 3 * i + 2);
        LDir& d = dirs[dst_c < me ? 0 : 1];
        const uint8_t hops = static_cast<uint8_t>(dst_c < me ? me - dst_c : dst_c - me);
        const uint64_t sem = safe_get_noc_addr(my_x, my_y, sync + arr * fs::WORD, 0);
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_wait_front(CB_SEND, NB);
            const uint32_t buf = get_read_ptr(CB_SEND);
#if defined(FS_R1_BF16)  // bf16 tiles: 2-page scatter packets (4 KB of payload per packet, as rs_send.cpp)
            static_assert(NB % 2 == 0, "tile pairs per packet");
            for (uint32_t t = 0; t < NB; t += 2) {
                if (d.next == NHDR) {
                    noc_async_writes_flushed();
                    d.next = 0;
                }
                volatile tt_l1_ptr Hdr* hd = d.data[d.next++];
                hd->to_chip_unicast(hops);
                const uint32_t pg = row * TPB + b * NB + t;
                const uint64_t d0 = r2.get_noc_addr(pg, 0, 0), d1 = r2.get_noc_addr(pg + 1, 0, 0);
                d.conn.wait_for_empty_write_slot();
                if (b + 1 == TPB / NB && t + 2 == NB) {
                    hd->to_noc_fused_unicast_scatter_write_atomic_inc(
                        NocUnicastScatterAtomicIncFusedCommandHeader(
                            {d0, d1}, sem, {static_cast<uint16_t>(TB32)}, 1, true),
                        2 * TB32);
                } else {
                    hd->to_noc_unicast_scatter_write(
                        NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(TB32)}), 2 * TB32);
                }
                d.conn.send_payload_without_header_non_blocking_from_address(buf + t * TB32, 2 * TB32);
                d.conn.send_payload_flush_non_blocking_from_address((uint32_t)hd, sizeof(Hdr));
            }
#else
            for (uint32_t t = 0; t < NB; t++) {
                if (d.next == NHDR) {
                    noc_async_writes_flushed();
                    d.next = 0;
                }
                volatile tt_l1_ptr Hdr* hd = d.data[d.next++];
                hd->to_chip_unicast(hops);
                const uint64_t dst = r2.get_noc_addr(row * TPB + b * NB + t, 0, 0);
                d.conn.wait_for_empty_write_slot();
                if (b + 1 == TPB / NB && t + 1 == NB) {
                    hd->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, TB32);
                } else {
                    hd->to_noc_unicast_write(NocUnicastCommandHeader{dst}, TB32);
                }
                d.conn.send_payload_without_header_non_blocking_from_address(buf + t * TB32, TB32);
                d.conn.send_payload_flush_non_blocking_from_address((uint32_t)hd, sizeof(Hdr));
            }
#endif
            noc_async_writes_flushed();
            cb_pop_front(CB_SEND, NB);
        }
    }
#if !defined(FS_NO_ROW_BARRIER)
    {  // THE PER-HOP DRAIN (fs_common.hpp): the row barrier of 1-hop tokens on this lane's own axis-1 connections
        auto token = [&](LDir& d, uint32_t word) {
            noc_async_writes_flushed();
            tok->to_chip_unicast(1);
            tok->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(my_x, my_y, sync + word * fs::WORD, 0), 1});
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)tok, sizeof(Hdr));
            noc_async_writes_flushed();
        };
        if (dirs[0].range || dirs[1].range) {
            fs::wait_ge(fs::word(sync, fs::LANDED), g);  // this lane's row-incoming packets all landed on this chip
            if (dirs[0].range) {
                fs::wait_ge(fs::word(sync, fs::BAR_L), g);  // chips below have landed everything
            }
            if (dirs[1].range) {
                token(dirs[1], fs::BAR_L);
                fs::wait_ge(fs::word(sync, fs::BAR_R), g);  // chips above have landed everything
            }
            if (dirs[0].range) {
                token(dirs[0], fs::BAR_R);
            }
        }
    }
#endif
    *gen = g;
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            fs::drain_close(dirs[d].conn);
        }
    }
#if defined(FS_RSL_REL_SEM)
    // W7 (P2 -> P8 s2 -> P11 -> P15 s2 on link h): hand link h BACK to C1X lane h once every connection here closed
    noc_semaphore_inc(safe_get_noc_addr(rel_x, rel_y, get_semaphore(FS_RSL_REL_SEM), 0), 1);
#endif
    noc_async_full_barrier();
}
