// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream RS with the lane reduce (tt/fabric_stream.FabricStreamRSR; prefill_megakernel.md §4b H5 -> H6),
// lane BRISC. Per launch g: the launch credit (CREDIT + p on the peer's lane h, then wait the peer's: the peer's
// previous launch no longer reads its landing buffer r), then interleaved per step i in rs_read.cpp's order:
//   send i:  CB_SEND batches of peer-owned block send[i] -> 2-page scatter packets into the PEER's landing buffer r
//   (row
//            = the block's index in the peer's range); the block's last packet is a fused scatter write + atomic +1 of
//            the peer lane h's ARR + p LANES + h (flush: the block's writes landed);
//   reduce i: CB_Y fp32 batches of own block red[i] -> the y destination (unit: DRAM fp32 y rows; the layer program:
//            the members' CB_Y 21 + FS_REDUCED, the next step).
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
constexpr uint32_t CB_Y = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t TB = 2048;
constexpr uint32_t TB32 = 4096;
constexpr uint32_t PKT = 2 * TB;
constexpr uint32_t NHDR = 4;
static_assert(TPB % NB == 0 && NB % 2 == 0, "packets never straddle a batch");

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
    constexpr auto r_args = TensorAccessorArgs<PH_CT0 + 4>();
    constexpr auto y_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
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
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t q = get_arg_val<uint32_t>(a++);
    const uint32_t lanes = get_arg_val<uint32_t>(a++);
    const uint32_t h = get_arg_val<uint32_t>(a++);
    const uint32_t up = get_arg_val<uint32_t>(a++);
    const uint32_t dn = get_arg_val<uint32_t>(a++);
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_send = get_arg_val<uint32_t>(a++);
    const uint32_t n_red = get_arg_val<uint32_t>(a++);
#if defined(FS_RS_ARITH)  // the lanes-reduce lists are arithmetic: lane h's blocks h, h + lanes, .. (PMK P15: 32 args
    // fewer on the lane BRISC composite, which carries P2 + P8 + P15 sends under the 256-arg launch limit)
    const uint32_t send_row = 0, red_row = 0;
    (void)send_row;
    (void)red_row;
#else
    const uint32_t send_row = a;  // n_send: the block's row tile in the PEER's landing buffer
    a += n_send;
    const uint32_t red_row = a;  // n_red: the block's row tile in this chip's y
    a += n_red;
#endif
    const auto rr = TensorAccessor(r_args, r_addr);
    const auto yy = TensorAccessor(y_args, y_addr);
    const uint32_t ranges[2] = {up, dn};
    Dir dirs[2];
    a = get_arg_val<uint32_t>(a);  // fab_at: the ABSOLUTE index of the fabric args (shared by a composite's phases)
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
                dirs[d].data[k] = fs_hdr_alloc();
                dirs[d].data[k]->to_chip_multicast(
                    MulticastRoutingCommandHeader{1, static_cast<uint8_t>(dirs[d].range)});
            }
            dirs[d].credit = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
            dirs[d].credit->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(dirs[d].range)});
            dirs[d].credit->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                safe_get_noc_addr(my_x, my_y, sync + (fs::CREDIT + p) * fs::WORD, 0), 1});
        }
    }
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN_B);
    const uint32_t g = *gen + 1;
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            dirs[d].conn.wait_for_empty_write_slot();
            dirs[d].conn.send_payload_flush_non_blocking_from_address((uint32_t)dirs[d].credit, sizeof(Hdr));
        }
    }
    fs::wait_ge(fs::word(sync, fs::CREDIT + q), g);
    const uint64_t arr = safe_get_noc_addr(my_x, my_y, sync + (fs::ARR + p * lanes + h) * fs::WORD, 0);
    auto send2 = [&](Dir& d, uint32_t src, uint64_t d0, uint64_t d1, bool signal) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();
            d.next = 0;
        }
        volatile tt_l1_ptr Hdr* hd = d.data[d.next++];
        d.conn.wait_for_empty_write_slot();
        if (signal) {
            hd->to_noc_fused_unicast_scatter_write_atomic_inc(
                NocUnicastScatterAtomicIncFusedCommandHeader({d0, d1}, arr, {static_cast<uint16_t>(TB)}, 1, true), PKT);
        } else {
            hd->to_noc_unicast_scatter_write(
                NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(TB)}), PKT);
        }
        d.conn.send_payload_without_header_non_blocking_from_address(src, PKT);
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)hd, sizeof(Hdr));
    };
    const uint32_t n = n_send > n_red ? n_send : n_red;
#if defined(FS_SEND_FIRST)  // (rs_read.cpp FS_SEND_FIRST) every send, then every reduced block's y write
    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t i = 0; i < n; i++) {
            if (pass == 0 && i < n_send) {
#else
    for (uint32_t i = 0; i < n; i++) {
        if (i < n_send) {
#endif
#if defined(FS_RS_ARITH)
                const uint32_t row = h + i * lanes;
#else
            const uint32_t row = get_arg_val<uint32_t>(send_row + i);
#endif
                for (uint32_t b = 0; b < TPB / NB; b++) {
                    cb_wait_front(CB_SEND, NB);
                    const uint32_t buf = get_read_ptr(CB_SEND);
                    for (uint32_t t = 0; t < NB; t += 2) {
                        const uint32_t pg = row * TPB + b * NB + t;
                        const uint64_t d0 = rr.get_noc_addr(pg, 0, 0), d1 = rr.get_noc_addr(pg + 1, 0, 0);
                        const bool last = b + 1 == TPB / NB && t + 2 == NB;
                        for (uint32_t d = 0; d < 2; d++) {
                            if (dirs[d].range) {
                                send2(dirs[d], buf + t * TB, d0, d1, last);
                            }
                        }
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(CB_SEND, NB);
                }
            }
#if defined(FS_SEND_FIRST)
            if (pass == 1 && i < n_red) {
#else
        if (i < n_red) {
#endif
#if defined(FS_RS_ARITH)
                const uint32_t row = h + i * lanes;
#else
            const uint32_t row = get_arg_val<uint32_t>(red_row + i);
#endif
                for (uint32_t b = 0; b < TPB / NB; b++) {
                    cb_wait_front(CB_Y, NB);
                    const uint32_t buf = get_read_ptr(CB_Y);
                    for (uint32_t t = 0; t < NB; t++) {
                        noc_async_write_page(row * TPB + b * NB + t, yy, buf + t * TB32);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(CB_Y, NB);
                }
            }
        }
    }
    noc_async_write_barrier();
    *gen = g;
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            fs::drain_close(dirs[d].conn);
        }
    }
    noc_async_full_barrier();
}
