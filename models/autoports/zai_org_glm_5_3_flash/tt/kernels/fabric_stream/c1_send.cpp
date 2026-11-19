// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C1 / C3 (the prefill-megakernel all-gather of x̂ into the MM cores' XS, doc/ttnn/16-prefill-
// production/prefill_megakernel.md §4b H1 / H2), lane BRISC (tt/fabric_stream.FabricStreamC1). Lane h of every chip
// owns ethernet link h both ways. Per launch g:
//   1. credit: line-multicast an atomic +1 of CREDIT + p on lane h of every peer in range. It certifies that this
//   chip's
//      program runs, so its MM cores' XS (static CB) and FS_ARRIVE semaphores (re-initialized at launch) exist. The
//      remote packets of this launch must not land earlier.
//   2. wait CREDIT + q >= g for every peer q this lane writes to (their programs run).
//   3. for this lane's local blocks in order (pulled into the staging CB by c1_read.cpp): send 2-tile (4 KiB) packets
//   to
//      the SAME XS address of the block's MM core on every peer (line multicast up / down); each producer slice
//      (SLICE_T tiles) ends with a fused write + atomic +1 of that MM core's FS_ARRIVE (flush: the slice's writes
//      landed), so the peer MM core j waits FS_ARRIVE >= SLICES for remote and local blocks alike.
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

namespace c1s {
constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);  // tiles per block
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);   // tiles per staging batch
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t SLICE_T = get_compile_time_arg_val(PH_CT0 + 3);  // tiles per producer slice (one FS_ARRIVE +1 each)
// FS_C2 (the pair's RS exchange, TP2 DRAM form): the block lands in the OWNER's landing buffer r (DRAM, row tile = the
// block's index in the owner's range) as 2-page scatter packets, and the slice's fused +1 goes to the owner MM core j's
// FS_REDUCED (its address in SEMADDR); the owner's consumer adds own + landed in fixed chip order
constexpr uint32_t TB = 2048;
constexpr uint32_t PKT = 2 * TB;  // 4 KiB per packet (max payload 4,352 B)
constexpr uint32_t NHDR = 4;
static_assert(TPB % NB == 0 && NB % 2 == 0 && SLICE_T % 2 == 0, "packets never straddle a batch or a slice");
}  // namespace c1s

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct Dir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[c1s::NHDR];
    volatile tt_l1_ptr Hdr* credit;
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    using namespace c1s;
#if defined(FS_C2)
    constexpr auto r_args = TensorAccessorArgs<PH_CT0 + 4>();
#endif
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
    const uint32_t sync = get_arg_val<uint32_t>(a++);  // this lane's sync shard (the same address on every lane / chip)
    const uint32_t my_x = get_arg_val<uint32_t>(a++);  // this lane core, NoC 0 (the peers' lane h is at the same core)
    const uint32_t my_y = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);  // chips on the axis
    const uint32_t up = get_arg_val<uint32_t>(a++);
    const uint32_t dn = get_arg_val<uint32_t>(a++);
    const uint32_t n_blk = get_arg_val<uint32_t>(a++);
    const uint32_t blk_xy = a;  // n_blk x (MM core NoC 0 x, y)
    a += 2 * n_blk;
#if defined(FS_C2)
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blk_row = a;  // n_blk x the block's row tile in the owner's landing buffer
    a += n_blk;
    const auto rr = TensorAccessor(r_args, r_addr);
#endif
    // PMK H0 (XS-FREE): the member credits per use landed on this lane's XSFREE word (32 at 64 members); 0 = the launch
    // credit alone (the standalone unit: the XS is free at launch)
    const uint32_t xsfree_n = get_arg_val<uint32_t>(a++);
    a = get_arg_val<uint32_t>(a);  // fab_at: the ABSOLUTE index of the fabric connection args (shared by every C1 phase
                                   // of a composite: each phase opens and closes the same connection in turn)
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
    // 0. (PMK) this chip's XS is free for this use: every member credited (XSFREE >= g x xsfree_n)
    if (xsfree_n) {
        fs::wait_ge(fs::word(sync, fs::XSFREE), g * xsfree_n);
    }
    // 1. credit every peer in range, 2. wait for theirs
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            dirs[d].conn.wait_for_empty_write_slot();
            dirs[d].conn.send_payload_flush_non_blocking_from_address((uint32_t)dirs[d].credit, sizeof(Hdr));
        }
    }
    for (uint32_t q = 0; q < n; q++) {
        if (q != p) {
            fs::wait_ge(fs::word(sync, fs::CREDIT + q), g);
        }
    }
    auto send = [&](Dir& d, uint32_t src, uint64_t dst, uint64_t sem, bool signal) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();  // the ring wraps: the headers' previous packets have left L1
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
#if defined(FS_C2)
    auto send2 = [&](Dir& d, uint32_t src, uint64_t d0, uint64_t d1, uint64_t sem, bool signal) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();
            d.next = 0;
        }
        volatile tt_l1_ptr Hdr* h = d.data[d.next++];
        d.conn.wait_for_empty_write_slot();
        if (signal) {
            h->to_noc_fused_unicast_scatter_write_atomic_inc(
                NocUnicastScatterAtomicIncFusedCommandHeader({d0, d1}, sem, {static_cast<uint16_t>(TB)}, 1, true), PKT);
        } else {
            h->to_noc_unicast_scatter_write(NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(TB)}), PKT);
        }
        d.conn.send_payload_without_header_non_blocking_from_address(src, PKT);
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    };
#endif
    // 3. the blocks (XSADDR / SEMADDR were written before the producers' first LREADY increment, i.e. before the first
    //    batch reached the staging CB)
    for (uint32_t k = 0; k < n_blk; k++) {
        const uint32_t bx = get_arg_val<uint32_t>(blk_xy + 2 * k), by = get_arg_val<uint32_t>(blk_xy + 2 * k + 1);
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_wait_front(CB_STAGE, NB);
            const uint32_t buf = get_read_ptr(CB_STAGE);
            invalidate_l1_cache();  // XSADDR / SEMADDR came over the NoC
#if !defined(FS_C2)
            const uint32_t xs = *fs::word(sync, fs::XSADDR);
#endif
            const uint64_t sem = safe_get_noc_addr(bx, by, *fs::word(sync, fs::SEMADDR), 0);
            for (uint32_t i = 0; i < NB; i += 2) {
                const uint32_t t = b * NB + i;  // tile of the block
                const bool last = (t + 2) % SLICE_T == 0;
#if defined(FS_C2)
                const uint32_t pg = get_arg_val<uint32_t>(blk_row + k) * TPB + t;
                const uint64_t d0 = rr.get_noc_addr(pg, 0, 0), d1 = rr.get_noc_addr(pg + 1, 0, 0);
                for (uint32_t d = 0; d < 2; d++) {
                    if (dirs[d].range) {
                        send2(dirs[d], buf + i * TB, d0, d1, sem, last);
                    }
                }
#else
                const uint64_t dst = safe_get_noc_addr(bx, by, xs + t * TB, 0);
                for (uint32_t d = 0; d < 2; d++) {
                    if (dirs[d].range) {
                        send(dirs[d], buf + i * TB, dst, sem, last);
                    }
                }
#endif
            }
            noc_async_writes_flushed();  // the packets' payload reads have left the batch
            cb_pop_front(CB_STAGE, NB);
        }
    }
    *gen = g;
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            fs::drain_close(dirs[d].conn);
        }
    }
    noc_async_full_barrier();
}
