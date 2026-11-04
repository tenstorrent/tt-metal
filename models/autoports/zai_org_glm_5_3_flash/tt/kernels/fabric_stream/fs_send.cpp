// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a prefill-megakernel collectives (doc/ttnn/07a-fused-streaming-layer/prefill_mk_collectives.md; ported into
// the DSA shard tree with FS_CONS: the waiter signals consumer cores once every block landed): FabricStream block send,
// lane BRISC (tt/fabric_stream.py): the ALL-GATHER (local block -> every chip, local write on) and the pair
// REDUCE-SCATTER exchange (the peer's block of the partial -> its landing buffer, local write off). The
// sequence-parallel input x [R_loc, H] (this chip's rows, DRAM interleaved tiles) is gathered into y [n R_loc, H]
// (DRAM, the same buffer address on every chip) along one mesh axis (a line of n chips, this chip at position p; on the
// 2x4 mesh every row / column is its own line, tt/fabric_stream.py axis=). Lane h of every chip owns ethernet link
// link_base + h in both line directions (the host's link table: tt/shard_plan.py). The lane's NCRISC (fs_read.cpp)
// streams its share of the local 32-row chunks (PPC pages each) into the staging CB in NB-page batches; here every PAIR
// of pages is written into the local y and line-multicast "up" (n - 1 - p hops) and "down" (p hops) as ONE fabric
// scatter packet (2 x 2 KiB to the two y pages: half the packets of one page per packet); the pair that ends a chunk is
// a fused scatter write + atomic increment (flush: every earlier write of the connection landed) of the receiver's
// counter ARR + p LANES + h on its lane-0 core (one counter per source lane: its chunks land in order). After its last
// batch the lane increments DONE on this chip's lane 0; lane 0 (waiter) then waits for every peer's chunks and every
// local lane (DONE). FS_SCATTER (tt/fabric_stream.py FabricStreamScatter, the 2x4 RS round 2): no local write; the
// block for destination q (fs_read.cpp streams them in the destination order) goes to q only: a UNICAST packet
// (to_chip_unicast(hops)) on this chip's connection in q's direction, into q's y at this chip's block (row_base), the
// chunk-end packet a fused write + atomic inc of q's counter ARR + p LANES + h (one counter per source: lesson of the
// credit-ahead hang). Runtime args after the FS_CONS block: n_dst, then (direction 0 up / 1 down, hops) per destination
// (count-prefixed). MK_PROBE_ARG (tt/fabric_stream.py probe=, the shard phase probe: kernels/mk/probe.hpp): the probe
// pair after the scatter block (before the fabric connections); marks: start, the first batch staged + the link opened,
// the sends done, closed (DONE), the waiter's completion (lane 0) -> one row of the MkMla probe tensor.
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "fs_common.hpp"

constexpr uint32_t PPC = get_compile_time_arg_val(0);  // pages (tiles) per chunk = H / 32
constexpr uint32_t NB = get_compile_time_arg_val(1);   // pages per staging batch
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(2);
constexpr uint32_t TB = 2048;
constexpr uint32_t NHDR = 4;  // data-header ring per direction: flush only when a header is reused
static_assert(PPC % NB == 0 && NB % 2 == 0, "page pairs never straddle a batch or a chunk");

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct Dir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[NHDR];
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t l0_x = get_arg_val<uint32_t>(a++);  // this chip's lane-0 core (the counters), NoC 0
    const uint32_t l0_y = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t up = get_arg_val<uint32_t>(a++);
    const uint32_t dn = get_arg_val<uint32_t>(a++);
    a++;  // (x address: fs_read.cpp's)
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t c_lo = get_arg_val<uint32_t>(a++);  // this lane's first local chunk
    const uint32_t c_n = get_arg_val<uint32_t>(a++);   // its chunk count
    const uint32_t row_base =
        get_arg_val<uint32_t>(a++);                      // the block's first row tile in y (on the peers: same address)
    const uint32_t waiter = get_arg_val<uint32_t>(a++);  // lane 0: the standalone completion
    a++;                                                 // (chips on the axis: the host's)
    const uint32_t lanes = get_arg_val<uint32_t>(a++);
    const uint32_t lane_h = get_arg_val<uint32_t>(a++);      // this lane
    const uint32_t local = get_arg_val<uint32_t>(a++);       // also write the block into this chip's y (all-gather)
    a++;                                                     // (the source chunk base: fs_read.cpp's)
    const uint32_t first_src = get_arg_val<uint32_t>(a++);   // waiter: the peers this chip receives from: [first_src,
    const uint32_t n_src = get_arg_val<uint32_t>(a++);       //   first_src + n_src) minus p
    a++;                                                     // (reserved)
    const uint32_t n_lane_ch0 = get_arg_val<uint32_t>(a++);  // waiter: chunks per lane (lane 0 / others) per source
    const uint32_t n_lane_ch1 = get_arg_val<uint32_t>(a++);
#if defined(FS_CONS)  // tt/fabric_stream.py cons=: the waiter's consumers (count, semaphore id, (x, y) pairs)
    const uint32_t ncons = get_arg_val<uint32_t>(a++);
    const uint32_t cons_sem = get_arg_val<uint32_t>(a++);
    const uint32_t cons0 = a;
    a += 2 * ncons;
#endif
#if defined(FS_SCATTER)
    const uint32_t n_dst = get_arg_val<uint32_t>(a++);
    const uint32_t dst0 = a;
    a += 2 * n_dst;
#endif
#if defined(MK_PROBE_ARG)
    a += 2;  // (the probe pair: kernels/mk/probe.hpp reads it at MK_PROBE_ARG)
#endif
    MK_MARK();  // probe 0: start
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
#if !defined(FS_LATE_OPEN)
            dirs[d].conn.open();
#endif
            for (uint32_t k = 0; k < NHDR; k++) {
                dirs[d].data[k] = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
#if !defined(FS_SCATTER)  // (the scatter routes every packet to its destination: send())
                dirs[d].data[k]->to_chip_multicast(
                    MulticastRoutingCommandHeader{1, static_cast<uint8_t>(dirs[d].range)});
#endif
            }
        }
    }
    const uint64_t cnt_noc = safe_get_noc_addr(l0_x, l0_y, sync + (fs::ARR + p * lanes + lane_h) * fs::WORD, 0);
    auto send = [&](Dir& d, uint32_t src, uint64_t dst0, uint64_t dst1, bool signal, uint32_t hops) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();  // the ring wraps: the headers' previous packets have left L1
            d.next = 0;
        }
        volatile tt_l1_ptr Hdr* h = d.data[d.next++];
#if defined(FS_SCATTER)
        h->to_chip_unicast(static_cast<uint8_t>(hops));
#else
        (void)hops;
#endif
        d.conn.wait_for_empty_write_slot();
        if (signal) {
            h->to_noc_fused_unicast_scatter_write_atomic_inc(
                NocUnicastScatterAtomicIncFusedCommandHeader(
                    {dst0, dst1}, cnt_noc, {static_cast<uint16_t>(TB)}, 1, true),
                2 * TB);
        } else {
            h->to_noc_unicast_scatter_write(
                NocUnicastScatterCommandHeader({dst0, dst1}, {static_cast<uint16_t>(TB)}), 2 * TB);
        }
        d.conn.send_payload_without_header_non_blocking_from_address(src, 2 * TB);
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    };
    const uint32_t nbat = c_n * PPC / NB;
#if defined(FS_LATE_OPEN)  // tt/fabric_stream.py after=: the link is held only around the sends, opened once the
    // lane that owns the link signalled its release (FS_AFTER_SEM on this core)
    bool opened = false;
#endif
#if defined(FS_SCATTER)
    for (uint32_t k = 0; k < n_dst; k++) {
        const uint32_t sdir = get_arg_val<uint32_t>(dst0 + 2 * k), shops = get_arg_val<uint32_t>(dst0 + 2 * k + 1);
        ASSERT(sdir < 2 && shops > 0 && dirs[sdir].range >= shops);
#else
    {
        constexpr uint32_t shops = 0;
#endif
        for (uint32_t b = 0; b < nbat; b++) {
            cb_wait_front(CB_STAGE, NB);
#if defined(FS_LATE_OPEN)
            if (!opened) {
#if defined(FS_AFTER_SEM)
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FS_AFTER_SEM)), 1);
#endif
                for (uint32_t d = 0; d < 2; d++) {
                    if (dirs[d].range) {
                        dirs[d].conn.open();
                    }
                }
                opened = true;
                MK_MARK();  // probe 1: the first batch staged (the producers done) + the link released / opened
            }
#endif
            const uint32_t buf = get_read_ptr(CB_STAGE);
            for (uint32_t i = 0; i < NB; i += 2) {
                const uint32_t idx = b * NB + i;  // this lane's page index
                const uint32_t chunk = c_lo + idx / PPC, col = idx % PPC;
                const uint32_t g0 = (row_base + chunk) * PPC + col;
                const uint32_t src = buf + i * TB;
                if (local) {
                    noc_async_write_page(g0, y, src);
                    noc_async_write_page(g0 + 1, y, src + TB);
                }
                const uint64_t d0 = y.get_noc_addr(g0, 0, 0), d1 = y.get_noc_addr(g0 + 1, 0, 0);
                const bool last = col + 2 == PPC;
#if defined(FS_SCATTER)
                send(dirs[sdir], src, d0, d1, last, shops);
#else
                for (uint32_t d = 0; d < 2; d++) {
                    if (dirs[d].range) {
                        send(dirs[d], src, d0, d1, last, shops);
                    }
                }
#endif
            }
            noc_async_writes_flushed();  // the local writes and the packets' payload reads have left the batch
            cb_pop_front(CB_STAGE, NB);
        }
    }  // (FS_SCATTER: the destinations)
    noc_async_write_barrier();
    MK_MARK();             // probe 2: every packet / local write of this lane sent
#if defined(FS_LATE_OPEN)  // closed before DONE / the consumers' signal: the next holder of the link may open
#if defined(FS_ROW_BARRIER)
    const bool keep_open = waiter != 0;  // THE PER-HOP DRAIN: the waiter keeps its line connections for the row barrier
#else
    constexpr bool keep_open = false;
#endif
    if (opened && !keep_open) {
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                fs::drain_close(dirs[d].conn);
            }
        }
    }
#endif
    noc_semaphore_inc(safe_get_noc_addr(l0_x, l0_y, sync + fs::DONE * fs::WORD, 0), 1);
    noc_async_atomic_barrier();
    MK_MARK();  // probe 3: closed, DONE
    if (waiter) {
        volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN);
        const uint32_t g = *gen + 1;
        for (uint32_t q = first_src; q < first_src + n_src; q++) {
            if (q != p) {
                for (uint32_t k = 0; k < lanes; k++) {
                    fs::wait_ge(fs::word(sync, fs::ARR + q * lanes + k), g * (k == 0 ? n_lane_ch0 : n_lane_ch1));
                }
            }
        }
        fs::wait_ge(fs::word(sync, fs::DONE), g * lanes);
        *gen = g;
#if defined(FS_ROW_BARRIER)
        {  // THE PER-HOP DRAIN (fs_common.hpp; coordinator 2026-10-23 23:16Z: every multi-hop lane stream): every
           // peer's chunks of this LINE stream landed here (the waits above); a row barrier of 1-hop tokens on this
           // stream's own line connections (dirs[0] = up / higher positions, dirs[1] = down), then the close, then the
           // consumers: no packet of this stream is in transit when any chip of the line ends its program, and the
           // link's next holder (the AR lane's own round) still opens only after the consumers' signal
#if defined(FS_LATE_OPEN)
            if (!opened && (dirs[0].range || dirs[1].range)) {  // nothing was sent: open behind the same release
#if defined(FS_AFTER_SEM)
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FS_AFTER_SEM)), 1);
#endif
                for (uint32_t d = 0; d < 2; d++) {
                    if (dirs[d].range) {
                        dirs[d].conn.open();
                    }
                }
                opened = true;
            }
#endif
            auto token = [&](Dir& d, uint32_t word) {  // a data header reused: every data send has been flushed
                noc_async_writes_flushed();
                volatile tt_l1_ptr Hdr* th = d.data[0];
                th->to_chip_unicast(1);
                th->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(l0_x, l0_y, sync + word * fs::WORD, 0), 1});
                d.conn.wait_for_empty_write_slot();
                d.conn.send_payload_flush_non_blocking_from_address((uint32_t)th, sizeof(Hdr));
                noc_async_writes_flushed();
            };
            if (dirs[1].range) {
                fs::wait_ge(fs::word(sync, fs::BAR_L), g);  // the chips below landed everything
            }
            if (dirs[0].range) {
                token(dirs[0], fs::BAR_L);
                fs::wait_ge(fs::word(sync, fs::BAR_R), g);  // the chips above landed everything
            }
            if (dirs[1].range) {
                token(dirs[1], fs::BAR_R);
            }
#if defined(FS_LATE_OPEN)
            if (opened) {
                for (uint32_t d = 0; d < 2; d++) {
                    if (dirs[d].range) {
                        fs::drain_close(dirs[d].conn);
                    }
                }
            }
#endif
        }
#endif
#if defined(FS_CONS)
        for (uint32_t k = 0; k < ncons; ++k) {
            const uint32_t x = get_arg_val<uint32_t>(cons0 + 2 * k), yy = get_arg_val<uint32_t>(cons0 + 2 * k + 1);
            noc_semaphore_inc(safe_get_noc_addr(x, yy, get_semaphore(cons_sem), 0), 1);
        }
        noc_async_atomic_barrier();
#endif
    }
#if !defined(FS_LATE_OPEN)
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            fs::drain_close(dirs[d].conn);
        }
    }
#endif
    MK_MARK();  // probe 4: (lane 0) every peer's chunks landed, the consumers signalled
    noc_async_full_barrier();
#if defined(MK_PROBE_ARG)  // the 64-B flush scratch: sync words PROBE_FLUSH .. of this lane-0 core (fs_common.hpp)
    MK_FLUSH((sync + fs::PROBE_FLUSH * fs::WORD + 63) & ~63u);
#endif
}
