// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream v3 (phase-ready), STREAMER BRISC (prefill_megakernel.md §4c; tt/pf_stream.PfStreamMM). Stream st
// holds NW output column tiles: its weight is the K x NW tiles of those columns, K-row-major, contiguous in DRAM bank
// st mod 8 ("mk-stream-v1", tt/mm_stream.py, cores = streams). It is served by TWO streamers, lane L = 0 / 1: for each
// of its NH = NW / NB N-blocks h, lane L sends the K-blocks j = L, L + 2, .. (block = KB rows x NB columns: KB
// contiguous NB-tile reads) on NoC L. The members consume the phase's blocks q = (st NH + h) NBK + j in order; with the
// phase's base QBASE (even) the launch-global block Q = QBASE + q goes to their PARK slot Q mod P, followed on the same
// NoC by a multicast SET of the members' lane-L VALID = Q / 2 + 1 (ordered behind the data: same source, NoC, VC). A
// 3-slot ring runs ahead (the slot of block i - 1 refills while block i multicasts). Credits: block i waits
// READY[i mod 2] of the phase's sync slot >= NDEST ((g - 1) NI / 2 + i / 2 + 1) (parity words: pfs_common.hpp; the
// protocol is simulated host-side, tt/pfs_credit_model.py). Lane chains: (st, L) waits TOKEN >= g before its first
// multicast (st > 0); after its last, a write barrier, then +1 TOKEN of (st + 1, L). CT / RT maps: pfs_common.hpp.
// pmk/proj (SPC > 1): the core serves the SPC streams RT_ST .. RT_ST + SPC - 1 of ITS rectangle in order (one list of
// SPC x NH x NBK / 2 blocks: i -> stream i / NIS, then (h, j) as before), the rectangle = RT_LX..HY, NDEST its members;
// no TOKEN chain (RT_NX = 0): every rectangle's two lanes run concurrently.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfs_common.hpp"

namespace pfs_s {
constexpr uint32_t NBK = get_compile_time_arg_val(PH_CT0 + pfs::CT_NBK);
constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + pfs::CT_KB);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + pfs::CT_NB);
constexpr uint32_t NW = get_compile_time_arg_val(PH_CT0 + pfs::CT_NW);
constexpr uint32_t TB = get_compile_time_arg_val(PH_CT0 + pfs::CT_TB);
constexpr uint32_t P = get_compile_time_arg_val(PH_CT0 + pfs::CT_P);
constexpr uint32_t NDEST = get_compile_time_arg_val(PH_CT0 + pfs::CT_NDEST);
constexpr uint32_t CB_RING = get_compile_time_arg_val(PH_CT0 + pfs::CT_RING);
constexpr uint32_t QBASE = get_compile_time_arg_val(PH_CT0 + pfs::CT_QBASE);
constexpr uint32_t SLOT = get_compile_time_arg_val(PH_CT0 + pfs::CT_SLOT);
constexpr uint32_t SPC = get_compile_time_arg_val(PH_CT0 + pfs::CT_SPC);
constexpr uint32_t R = 3;
constexpr uint32_t NH = NW / NB;
constexpr uint32_t NIS = NH * NBK / 2;  // blocks per stream per lane
constexpr uint32_t NI = SPC * NIS;      // this streamer's blocks per launch
constexpr uint32_t BB = KB * NB * TB;
constexpr uint32_t RB = NB * TB;  // one K row of a block: one read transaction
static_assert(NW % NB == 0 && NBK % 2 == 0 && NIS % 2 == 0 && QBASE % 2 == 0 && RB <= NOC_MAX_BURST_SIZE, "geometry");
static_assert(SPC >= 1, "streams per core");
}  // namespace pfs_s

void kernel_main() {
    using namespace pfs_s;
    const uint32_t a = ph_rt0;
    if (get_arg_val<uint32_t>(a + pfs::RT_ACTIVE) == 0) {
        return;  // no stream of this phase on this core (no NoC transaction issued)
    }
    const uint32_t sync = get_arg_val<uint32_t>(a + pfs::RT_SYNC) + SLOT * pfs::SYNC_WORDS * pfs::WORD;
    const uint32_t st = get_arg_val<uint32_t>(a + pfs::RT_ST);
    const uint32_t L = get_arg_val<uint32_t>(a + pfs::RT_L);
    const uint32_t nst = get_arg_val<uint32_t>(a + pfs::RT_NST);
    const uint32_t w_addr = get_arg_val<uint32_t>(a + pfs::RT_W);
    const uint32_t vc = get_arg_val<uint32_t>(a + pfs::RT_VC);
    const uint32_t lx = get_arg_val<uint32_t>(a + pfs::RT_LX), ly = get_arg_val<uint32_t>(a + pfs::RT_LY);
    const uint32_t hx = get_arg_val<uint32_t>(a + pfs::RT_HX), hy = get_arg_val<uint32_t>(a + pfs::RT_HY);
    const uint32_t nx = get_arg_val<uint32_t>(a + pfs::RT_NX), ny = get_arg_val<uint32_t>(a + pfs::RT_NY);
#if defined(PFS_PROBE)
    uint64_t t_tok = 0, t_read = 0, t_ready = 0, t_flush = 0, t0 = get_timestamp(), tq;
#define PFS_T0() tq = get_timestamp()
#define PFS_ACC(v) v += get_timestamp() - tq
#else
#define PFS_T0()
#define PFS_ACC(v)
#endif
    const uint8_t noc = static_cast<uint8_t>(get_arg_val<uint32_t>(a + pfs::RT_NOC));  // 7a: L
    volatile tt_l1_ptr uint32_t* gen = pfs::word(sync, pfs::GEN);
    const uint32_t g = *gen + 1;
    // the core's streams: (bank, m0) pairs at RT_SB; each stream's weight is contiguous in its bank, so the read state
    // (bank base, RB bytes, VC) is re-armed when the list moves to the next stream
    uint32_t cur_s = 0xffffffffu;
    uint64_t dram = 0;
    uint32_t src0 = 0;
    const uint32_t ring = get_write_ptr(CB_RING);
    auto arm = [&](uint32_t si) {
        if (si != cur_s) {
            cur_s = si;
            const uint32_t bank = get_arg_val<uint32_t>(a + pfs::RT_SB + 2 * si);
            const uint32_t m0 = get_arg_val<uint32_t>(a + pfs::RT_SB + 2 * si + 1);
            dram = get_noc_addr_from_bank_id<true>(bank, 0);
            noc_async_read_one_packet_set_state<true>(dram, RB, vc);
            src0 = w_addr + m0 * TB;
        }
    };
    auto issue = [&](uint32_t i) {  // this streamer's block i -> ring slot i mod R (transaction id = slot + 1)
        const uint32_t si = i / NIS, ii = i % NIS;
        arm(si);
        const uint32_t h = ii / (NBK / 2), j = 2 * (ii % (NBK / 2)) + L;
        const uint32_t trid = i % R + 1;
        noc_async_read_set_trid(trid);
        uint32_t src = src0 + ((j * KB) * NW + h * NB) * TB, dst = ring + (i % R) * BB;
        for (uint32_t r = 0; r < KB; r++) {
            noc_async_read_one_packet_with_state_with_trid<false, false>(dram, src, dst, trid);
            src += NW * TB;
            dst += RB;
        }
    };
    for (uint32_t i = 0; i < R && i < NI; i++) {
        issue(i);
    }
    if (st > 0) {
        PFS_T0();
        pfs::wait_ge(pfs::word(sync, pfs::TOKEN), g);
        PFS_ACC(t_tok);
    }
    volatile tt_l1_ptr uint32_t* val = pfs::word(sync, L ? pfs::VAL1 : pfs::VAL0);
    for (uint32_t i = 0; i < NI; i++) {
        const uint32_t si = i / NIS, ii = i % NIS;
        const uint32_t h = ii / (NBK / 2), j = 2 * (ii % (NBK / 2)) + L;
        const uint32_t q =
            QBASE + ((st + si) * NH + h) * NBK + j;  // launch-global block (st: the rectangle's local stream)
        PFS_T0();
        noc_async_read_barrier_with_trid(i % R + 1);
        PFS_ACC(t_read);
        PFS_T0();
        pfs::wait_ge(pfs::word(sync, (i & 1) ? pfs::READY1 : pfs::READY), NDEST * ((g - 1) * (NI / 2) + i / 2 + 1));
        PFS_ACC(t_ready);
        PFS_T0();
        noc_async_writes_flushed(noc);  // block i - 1 left: its ring slot and the VAL word are free
        PFS_ACC(t_flush);
        if (i >= 1 && i - 1 + R < NI) {
            issue(i - 1 + R);
        }
        // THE RING-ADDRESS RULE (fabric_stream/fs_common.hpp): the members' VALID semaphore = this streamer's own
        // (the members' id from the shared compile-time list); the published VALIDADDR is only the freshness token of
        // PARKADDR (pfs_member_rd.cpp publishes PARKADDR, VALIDADDR0 / 1, then a write barrier, then its READY credit)
        const uint32_t valid = get_semaphore(
            L ? get_compile_time_arg_val(PH_CT0 + pfs::CT_V1)
              : get_compile_time_arg_val(PH_CT0 + pfs::CT_V0));  // fmt: skip
        {
            volatile tt_l1_ptr uint32_t* pv = pfs::word(sync, L ? pfs::VALIDADDR1 : pfs::VALIDADDR0);
            do {
                invalidate_l1_cache();
            } while (*pv != valid);
        }
        const uint32_t park = *pfs::word(sync, pfs::PARKADDR);
        // the publisher (a member of this rectangle, pfs_member_rd.cpp RT_FIRST) wrote them before its first credit;
        // a 0 here would multicast into the members' mailboxes (d0 card2 18:26Z): stop under the watcher instead
        ASSERT(park != 0 && valid != 0);
        // NoC 0 multicasts go low -> high corner, NoC 1 high -> low (ttnn's 2D mcast matmul convention)
        const uint64_t dst = noc ? get_noc_multicast_addr(hx, hy, lx, ly, park + (q % P) * BB, noc)
                                 : get_noc_multicast_addr(lx, ly, hx, hy, park + (q % P) * BB, noc);
        const uint64_t vdst = noc ? get_noc_multicast_addr(hx, hy, lx, ly, valid, noc)
                                  : get_noc_multicast_addr(lx, ly, hx, hy, valid, noc);
        *val = q / 2 + 1;
        noc_async_write_multicast(ring + (i % R) * BB, dst, BB, NDEST, false, noc);
        noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(val), vdst, NDEST, false, noc);
    }
    noc_async_write_barrier(noc);  // every member acked this streamer's blocks and VALID sets
    if (st + 1 < nst) {
        noc_semaphore_inc(get_noc_addr(nx, ny, sync + pfs::TOKEN * pfs::WORD, noc), 1, noc);
        noc_async_atomic_barrier(noc);
    }
    noc_async_read_set_trid(0);
    noc_async_read_one_packet_set_state<true>(dram, RB, 1);  // firmware default VC (NOC_CTRL persists)
    (void)nst;
#if defined(PFS_PROBE)
    *pfs::word(sync, pfs::P_TOKEN) = static_cast<uint32_t>(t_tok);
    *pfs::word(sync, pfs::P_READ) = static_cast<uint32_t>(t_read);
    *pfs::word(sync, pfs::P_READY) = static_cast<uint32_t>(t_ready);
    *pfs::word(sync, pfs::P_FLUSH) = static_cast<uint32_t>(t_flush);
    *pfs::word(sync, pfs::P_SPAN) = static_cast<uint32_t>(get_timestamp() - t0);
#endif
    *gen = g;
}
