# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""se_engine3 - the shared expert of the fused MoE head, ROW-CHUNK-MAJOR with its weights RESIDENT in L1.

y [rows, H] = (glu(x [rows, H] @ W_gu [H, 2 I])) @ W_down [I, H] on a GX x GY core rectangle, chunk by chunk (CR = GY MT
row tiles a chunk: core row cy owns the chunk's row tiles cy MT .. cy MT + MT - 1), every buffer in one L1 arena:
  - weights: column cx owns its gate / up tile pairs and its down output tiles; the column's W blocks (gate / up
    [KB x pass width], then down [KBD x pass width], one chunk's stream order) are spread over the column's cores
    (block b on row b % GY), read from DRAM ONCE at the call's start; each chunk streams them down the column from
    their holders (L1 -> L1 multicast, per-slot credits back to the next holder of the slot);
  - x: all of it resident ("x own": K step j of every chunk on its owner column, the gather lands it there or the
    engine loads it), multicast along the row per gate / up pass by two kernel_lib rotating PerRow families (the two
    x slots: steps of even / odd index; each family's owner subset divides its steps a pass, so a step's owner is the
    same every pass and chunk);
  - gate / up: per pass all of K in DEST (MT x pass width <= 8 fp32 tiles), the GLU, h packed once;
  - h: all-gathered along the row by a kernel_lib rotating PerRow family (round (row tile, column): every core's h
    rows into every row core's "h all" [MT][I/32]);
  - down: per pass (<= 8 / MT output tiles) all of K in DEST over h all and the streamed W_down blocks, y packed once
    and written (NCRISC) - while the next chunk's x is still arriving."""

import os

import torch
from ttnn.bringup.combine_rf import persist_l1
from ttnn.bringup.combine_rf.lin2d import column_split, glu_weight  # noqa: F401

import ttnn

NWS = int(__import__("os").environ.get("SE_NWS", "3"))  # (W slots; SE_NWS: A/B, <= 4 - the source words rotate over 4)
XREADY, WV, WFREE = 0, 1, 1 + NWS  # (hand-written semaphores; the kernel_lib families' come after)
XLD = 1 + 2 * NWS  # (my own-shard x is loaded: the NCRISC's W load may start)
HCNT, HFREE = 1 + 2 * NWS, 3 + 2 * NWS  # (unused: the h exchange is the kernel_lib family again)
NSEM_OWN = 2 + 2 * NWS
# (the router logits on the engine, SE_RTR: words of a zeroed L1 tensor - a K chunk of x rows / W_r^T landed (the
# round), the leads' acks - 16 B apart; the semaphore ids are all taken by the kernel_lib families)
RXR, RWR, RXS = 0, 16, 32  # (the router's ew words: RX / RW ready (a round), the x lead's set source)
RGATE = 64  # (every engine core's partials written: the W streams' gate, SE_RTR_GATE)
XRL = 96  # (a row lead's count of its row's own-x loads: the router's x read from x own, SE_RTRX_L1)
XACK, WACK = 128, 144  # (more than two logits rounds: the receivers' "round consumed" counts on the x / W leads)
XIN, WIN, HOWN, HALL, OUT, ACC = 0, 1, 2, 3, 4, 5  # CB ids
NOUT = 2  # (down passes OUT holds)
RS_CB = dict(hold=6, finb=7, arrf=8, arrb=9, finc=10, outf=11, psync=12, ffree=20)  # (down + reduce-scatter CBs)
LAT_CB = 13  # (the latent block [MT][LWMAX], bf16)
PREP_CB = dict(pu=14, po=15, lrdy=16)
XFREE_CB = 21  # (x own's ring: a chunk's prep read its slot - the compute's credit to the BRISC's reload)
RTR_CB = dict(
    rx=17, rw=18, ro=19
)  # (the router logits on the engine: x rows [KTO][MT], W_r^T [ETC][KTO], out fp32)  # (dispatch prep: the units' source view, the untilized rows, latent ready)
ARENA_MAX = (
    1344 * 1024
)  # (L1 a core the arena may take: Blackhole's 1461248-byte bank less the movers' static CBs (~177 KB
# end on a port core) - the arena range is carved on them too - and what the head keeps below it)


def dest_tiles():
    """fp32 DEST tiles a block: 8 with full sync, 4 with half sync (SE_HALFSYNC=1, A/B) - the down passes' width and
    the reduce-scatter plan's pwmax follow it."""
    return 4 if os.environ.get("SE_HALFSYNC") == "1" else 8


def arena_cap(mesh_device, kb):
    """L1 the arena may take on this mesh now: from where it lands (the top of the free L1 - placed top-down) down to the
    movers' static circular buffers (the x gather's port CB at K step kb: the arena's range is carved on the port cores
    too), ARENA_MAX at most. (GLM-5.3's larger fabric payload - 6 pages a packet - grows the port CB past ARENA_MAX's
    allowance: its arena then clashed with it.)"""
    return arena_cap_of(l1_free_bytes(mesh_device), kb)


def arena_cap_of(l1_free, kb, payload=None):
    """arena_cap from a free-L1 figure (host planning: no device; payload the fabric's packet payload bytes)."""
    from models.demos.moe_block.tt.x_gather3 import port_cb_bytes

    return min(ARENA_MAX, (l1_free - port_cb_bytes(kb, payload=payload) - 1024) // 64 * 64)


def l1_free_bytes(mesh_device):
    """L1 a core from the allocator's base up to the lowest L1 allocated now (a probe: L1 is placed top-down), at most
    the largest free block."""
    probe = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
    )
    top = int(probe.buffer_address())
    ttnn.deallocate(probe)
    # (and no more than the largest free block: words a closed block's ops left mid-L1 split it - the probe lands above
    # them, a large arena below; an earlier block's leftovers moved a K3 head's arena into the movers' CBs)
    mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    free = top - ttnn._ttnn.device.GetAllocatorBaseAddress(mesh_device, ttnn.BufferType.L1)
    return min(free, int(mv.largest_contiguous_bytes_free_per_bank))


_BRISC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "api/debug/kernel_profiler.hpp"
using namespace dataflow_kernel_lib;
constexpr uint32_t GX = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1), KB = get_compile_time_arg_val(2);
constexpr uint32_t NSG = get_compile_time_arg_val(3), NPG = get_compile_time_arg_val(4), NCH = get_compile_time_arg_val(5);
constexpr uint32_t HT = get_compile_time_arg_val(6), POMAX = get_compile_time_arg_val(7), CR = get_compile_time_arg_val(8);
constexpr uint32_t S0 = get_compile_time_arg_val(9), S1 = get_compile_time_arg_val(10), KT = get_compile_time_arg_val(11);
constexpr uint32_t XREADY = get_compile_time_arg_val(12), HCNT = get_compile_time_arg_val(13), HFREE = get_compile_time_arg_val(14);
constexpr auto x_args = TensorAccessorArgs<15>();
#ifdef SE_NODOWN
constexpr auto h_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
#endif
#ifdef SE_RTR
constexpr auto brx_args = TensorAccessorArgs<SE_RTR_BCT0>();  // (the router's x: the chip's x shard [rows, H / TP])
#endif
constexpr uint32_t CB_XIN = 0, CB_HOWN = 2, CB_HALL = 3;
constexpr uint32_t NF = NSG / 2, NOWN0 = NF / S0, NOWN1 = NF / S1;  // (steps a family a pass; per owner)
#ifdef SE_XPIPE
// (SE_XPIPE: my owned x slots load under transaction ids 13 .. 15 by slot, and a slot's send waits only on its id - the
// rounds start as the first steps land, not after the whole load; the W reads keep ids 1 .. 12)
#define XTRID(m, j) (13 + (((m) * nown + ((j) & 1 ? (i0 < GX ? NOWN0 : 0) + own_slot(j) : own_slot(j))) % 3))
#endif
// x position j (the j-th K step this chip processes: its own shard's first, then each shard as it lands): family j % 2,
// owner = the family's subset[(j / 2) % S] (S divides a shard's steps / 2, so the rotation over (shard, pass, step)
// rounds keeps every position on one owner), its local index (j / 2) / S
FORCE_INLINE uint32_t own_slot(uint32_t j) { return (j & 1) ? (j / 2) / S1 : (j / 2) / S0; }
// (the args equal on every core of the chip go once, as common args [S, NC, those NC values]: arg i is the core's
// own arg i below S, common below S + NC, the core's arg i - NC past them - every index stays the full list's)
FORCE_INLINE uint32_t se_ra(uint32_t i) {
    const uint32_t s = get_common_arg_val<uint32_t>(0), nc = get_common_arg_val<uint32_t>(1);
    return i < s ? get_arg_val<uint32_t>(i) : i < s + nc ? get_common_arg_val<uint32_t>(2 + i - s) : get_arg_val<uint32_t>(i - nc);
}
void kernel_main() {
    size_t a = 0;
    // (mine first - then from x_addr to y's address the chip's: se_ra's common span)
    const uint32_t cx = se_ra(a++), cy = se_ra(a++);
    const uint32_t po = se_ra(a++), h0 = se_ra(a++);
    const uint32_t i0 = se_ra(a++), i1 = se_ra(a++);  // my index in family 0 / 1 (or GX)
    const uint32_t ry = se_ra(a++);
    const uint32_t x_addr = se_ra(a++), x_ct = se_ra(a++), x_k0 = se_ra(a++);
    const uint32_t xown = se_ra(a++), xin = se_ra(a++), hown = se_ra(a++);
    const uint32_t hall = se_ra(a++), lx = se_ra(a++), rx = se_ra(a++);
    const uint32_t rtr_x = se_ra(a++), rtr_xa = se_ra(a++);  // (the router's x; RX)
    const uint32_t ft0 = a;  // (the x families: family 0 / 1 columns [S0] [S1], per column (nown, family-1 base slot))
    a += S0 + S1 + 2 * GX;
    const uint32_t vx0 = a, sps = se_ra(vx0 + GX), g = se_ra(vx0 + GX + 1);
    const uint32_t ord0 = vx0 + GX + 2;  // (the row's NoC x; steps a shard, shards; the shard order: own, then arrival)
    auto step_of = [&](uint32_t j) { return se_ra(ord0 + j / sps) * sps + j % sps; };  // (position j)
    const uint32_t nown = (i0 < GX ? NOWN0 : 0) + (i1 < GX ? NOWN1 : 0);  // (my steps a chunk)
    auto mine = [&](uint32_t j) { return (j & 1) ? ((j / 2) % S1 == i1) : ((j / 2) % S0 == i0); };
    // my steps' x-own slot: [chunk][family 0 slots, then family 1][MT][KB]
#ifndef SE_XRING
#define SE_XRING NCH  // (x own: every chunk resident)
#endif
    auto own_addr = [&](uint32_t m, uint32_t j) {  // (chunk m in slot m % SE_XRING: the DRAM fallback's ring)
        const uint32_t slot = (j & 1) ? (i0 < GX ? NOWN0 : 0) + own_slot(j) : own_slot(j);
        return xown + (((m % SE_XRING) * nown + slot) * MT) * KB * 2048;
    };
#ifdef SE_RTR
    auto rtrx = [&]() {  // (called after my own-shard x load: the gather's ports wait on every matmul core's - a row
        // lead's x load behind its ~20 us of router x held the gather back that long: GLM-5.3 ring 640 235.8 -> 219.2 us)
    if (cx == 0) {  // the router's x: the row's first core reads the row's x (MT row tiles, the chip's K slice) a K chunk
        // a round into RX and multicasts it along the row on NOC0 (the NCRISC's W_r^T goes down the columns on NOC1),
        // then sets the round on the row's RX-ready word (from my own source word RXS: the NCRISC re-arms the ready word)
        DeviceZoneScopedN("SE-RTRX");
        const auto rxg = TensorAccessor(brx_args, rtr_x, 2048);
        constexpr uint32_t XS = SE_KC * SE_MTR * 2048;  // (SE_MTR: my MT rows of every chunk - row rr of chunk rr / MT)
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + RXS_ID);
        constexpr uint32_t NRD = SE_KTO / SE_KC;
        volatile tt_l1_ptr uint32_t* xack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + XACK_ID);
#ifdef SE_RTRX_L1
        {  // (the row's x is the own shard the row's cores just loaded into x own: read from their L1, not the 2 MB again
           // from DRAM beside the own-x loads and W_r^T - once every row core counted its load on my XRL word)
            volatile tt_l1_ptr uint32_t* xrl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + XRL_ID);
            while (*xrl < GX) invalidate_l1_cache();
            *xrl = 0;  // (re-armed: one count a row core a call, all in)
        }
#endif
        for (uint32_t rd = 0; rd < NRD; ++rd) {
            const uint32_t xa = rtr_xa + (rd & 1) * XS;  // (two buffers; past two rounds the row's acks first)
            if (NRD > 2 && rd >= 2) while (*xack < (rd - 1) * GX) invalidate_l1_cache();
            for (uint32_t r = 0; r < SE_MTR; ++r)
                for (uint32_t k = 0; k < SE_KC; ++k) {
#ifdef SE_RTRX_L1
                    // (K tile kt of the chip's slice: own-shard position kt / KB - its owner column and x-own slot as
                    // x_gather3's dst_of; row r: chunk r / MT, local row r % MT)
                    const uint32_t kt = rd * SE_KC + k, j = kt / KB, hh = j / 2;
                    const uint32_t col = (j & 1) ? se_ra(ft0 + S0 + hh % S1) : se_ra(ft0 + hh % S0);
                    const uint32_t nw = se_ra(ft0 + S0 + S1 + 2 * col), b1 = se_ra(ft0 + S0 + S1 + 2 * col + 1);
                    const uint32_t slot = (j & 1) ? b1 + hh / S1 : hh / S0;
                    const uint32_t xa_ = xown + ((((r / MT) * nw + slot) * MT + r % MT) * KB + kt % KB) * 2048;
                    noc_async_read(get_noc_addr(se_ra(vx0 + col), ry, xa_), xa + (k * SE_MTR + r) * 2048, 2048);
#else
                    noc_async_read(rxg.get_noc_addr(GROW(r / MT, cy * MT + r % MT) * SE_KTO + rd * SE_KC + k), xa + (k * SE_MTR + r) * 2048, 2048);
#endif
                }
            noc_async_read_barrier();
            noc_async_write_multicast_loopback_src(xa, get_noc_multicast_addr(lx, ry, rx, ry, xa), XS, GX);
            noc_async_writes_flushed();
            *src = rd + 1;
            noc_semaphore_set_multicast_loopback_src(SE_EW + RXS_ID, get_noc_multicast_addr(lx, ry, rx, ry, SE_EW + RXR_ID), GX);
            noc_async_writes_flushed();
        }
        if (NRD > 2) {  // (re-arm: every ack landed - the last were waited for)
            noc_semaphore_inc(get_noc_addr(SE_EW + XACK_ID), 0u - (NRD - 2) * GX);
            noc_async_atomic_barrier();
        }
    }
    };
#endif
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t xports = se_ra(ord0 + g);  // (the gather's ports read my own shard from here)
#ifdef SE_XN1
    // (SE_XN1: chunk 0's odd (family 1) steps are loaded by the NCRISC on its NOC1 - both NoCs' read bandwidth for the
    // first pass, which the x load binds. Table at SE_WS word 512 (count, then NOC1 address lo / hi, L1 destination);
    // word 10: table ready, 11: the NCRISC issued it all (transaction id 15 on NOC1, polled here read-only))
    volatile tt_l1_ptr uint32_t* xsy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_WS);
    if (x_ct && !xports) {
        uint32_t n = 0;
        for (uint32_t j = 1; j < NSG; j += 2) {
            const uint32_t s = step_of(j);
            if (!(mine(j) && s * KB >= x_k0 && s * KB < x_k0 + x_ct)) continue;
            for (uint32_t r = 0; r < MT; ++r)
                for (uint32_t kk = 0; kk < KB; ++kk) {
                    const uint64_t a = xg.get_noc_addr(GROW(0, cy * MT + r) * x_ct + s * KB + kk - x_k0, 0, 1);
                    xsy[513 + 3 * n] = static_cast<uint32_t>(a), xsy[514 + 3 * n] = static_cast<uint32_t>(a >> 32);
                    xsy[515 + 3 * n] = own_addr(0, j) + (r * KB + kk) * 2048;
                    ++n;
                }
        }
        xsy[512] = n;
        xsy[10] = 1;
    }
#endif
    auto xload_step = [&](uint32_t m, uint32_t j) {  // (one owned step of chunk m from DRAM x [rows, x_ct])
        const uint32_t s = step_of(j);
        if (!(mine(j) && s * KB >= x_k0 && s * KB < x_k0 + x_ct)) return;
#ifdef SE_XN1
        if (m == 0 && (j & 1) && !xports) return;  // (the NCRISC's)
#endif
#ifdef SE_XPIPE
        noc_async_read_set_trid(XTRID(m, j));  // (its slot's transaction id: its send waits only on that)
#endif
        for (uint32_t r = 0; r < MT; ++r)
            for (uint32_t kk = 0; kk < KB; ++kk)
                noc_async_read(xg.get_noc_addr(GROW(m, cy * MT + r) * x_ct + s * KB + kk - x_k0),
                               own_addr(m, j) + (r * KB + kk) * 2048, 2048);
    };
#ifdef SE_XPIPE
    // (no ports: chunk 0's steps load in need order, SE_XL positions ahead of the rounds - all at once, every step's
    // load lands at the end of the whole load; with ports, all of it first as before)
    uint32_t xj = 0;
    auto xload_upto = [&](uint32_t jmax) {
        if (!x_ct) return;
        for (; xj < NSG && xj <= jmax; ++xj) xload_step(0, xj);
        noc_async_read_set_trid(0);
    };
#endif
    if (x_ct) {  // (alone) my steps of every chunk from DRAM x [rows, x_ct]: columns x_k0 ..
        DeviceZoneScopedN("SE-XLOAD");
#ifdef SE_XPIPE
        for (uint32_t m = 1; m < NCH; ++m)
            for (uint32_t j = 0; j < NSG; ++j) xload_step(m, j);
        xload_upto(xports ? NSG : SE_XL);
        if (xports) noc_async_read_barrier();
#else
        for (uint32_t m = 0; m < SE_XRING; ++m)  // (the ring's first chunks; the rest as their slots free)
            for (uint32_t j = 0; j < NSG; ++j) xload_step(m, j);
        noc_async_read_barrier();
#endif
    }
    {  // (my own-shard x is loaded: count it on the gather's ports - they send it from here - and let my W load start)
        const uint32_t np0 = ord0 + g, nports = se_ra(np0), xl_addr = se_ra(np0 + 1);
        for (uint32_t i = 0; i < nports; ++i)
            noc_semaphore_inc(get_noc_addr(se_ra(np0 + 2 + 2 * i), se_ra(np0 + 3 + 2 * i), xl_addr), 1);
#if defined(SE_RTR) && defined(SE_RTRX_L1)
        noc_semaphore_inc(get_noc_addr(se_ra(vx0), ry, SE_EW + XRL_ID), 1);  // (and on my row's lead: the router's x)
#endif
        noc_async_atomic_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(se_ra(np0 + 2 + 2 * nports))) = 1;
    }
#ifdef SE_RTR
    rtrx();  // (the router's x after my own x)
#endif
#ifdef SE_BRW
    volatile tt_l1_ptr uint32_t* wsy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_WS);
    uint32_t wn = 0, wbk = 0, wnext = 0;
    {  // (SE_BRW: the W DRAM reads on my NOC0, all issued now - the NCRISC (NOC1) wrote a table of its blocks in its
       // stream order at SE_WS (words: 0 ready | count, 1 issued, 3 bank; from 16: DRAM offset, L1 destination,
       // bytes); block k under transaction id min(k, 13) + 1, which the NCRISC polls on NOC0's status before sending
       // it. Nothing here waits on the stream later, so the x / h rounds can block freely)
#ifdef SE_BRW_BAR
        {  // (a grid barrier first: no weight burst while any core still loads its x - it starved the x loads.
           // Words on core SE_BAR: 8 x loaded, 9 done (its end re-arms both); polled with 4-byte NOC0 reads)
            const uint64_t bar = get_noc_addr(SE_BAR_X, SE_BAR_Y, SE_WS + 32);
            noc_semaphore_inc(bar, 1);
            noc_async_atomic_barrier();
            volatile tt_l1_ptr uint32_t* pr = wsy + 12;
            do {
                noc_async_read(bar, reinterpret_cast<uint32_t>(pr), 4);
                noc_async_read_barrier();
                invalidate_l1_cache();
            } while (*pr < SE_BAR_N);
        }
#endif
        uint32_t r;
        do {
            invalidate_l1_cache();
            r = wsy[0];
        } while (!(r >> 31));
        wn = r & 0xFFFF, wbk = wsy[3];
    }
    // (paced: the block of W round w goes out once my x round reaches w - SE_WPD; every block once my x rounds end.
    // No wait on anyone: a holder issues up to its own position + SE_WPD and nobody needs rounds past its own)
    auto wissue = [&](uint32_t lim) {
        while (wnext < wn && wsy[19 + 4 * wnext] <= lim) {
            noc_async_read_set_trid((wnext < 11 ? wnext : 11) + 1);
            noc_async_read(get_noc_addr_from_bank_id<true>(wbk, wsy[16 + 4 * wnext]), wsy[17 + 4 * wnext], wsy[18 + 4 * wnext]);
            wsy[1] = ++wnext;  // (issued count: the NCRISC polls a block's transaction id only once it was issued)
        }
        noc_async_read_set_trid(0);
    };
    wissue(SE_WPD);
#define WISSUE(lim) wissue(lim)
#else
#define WISSUE(lim)
#endif
    constexpr auto mx0 = McastArgs<get_named_compile_time_arg_val("x0_ct_offset"), get_named_compile_time_arg_val("x0_rt_offset")>();
    constexpr auto mx1 = McastArgs<get_named_compile_time_arg_val("x1_ct_offset"), get_named_compile_time_arg_val("x1_rt_offset")>();
#ifndef SE_NODOWN
    constexpr auto mh = McastArgs<get_named_compile_time_arg_val("h_ct_offset"), get_named_compile_time_arg_val("h_rt_offset")>();
#endif
    Noc noc;
#ifndef SE_NODOWN
    auto sh = mh.optional_sender(noc);
    auto rh = mh.optional_receiver(noc);
#endif
    auto s0 = mx0.optional_sender(noc);
    auto r0 = mx0.optional_receiver(noc);
    auto s1 = mx1.optional_sender(noc);
    auto r1 = mx1.optional_receiver(noc);
    volatile tt_l1_ptr uint32_t* xready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(XREADY));
    for (uint32_t m = 0; m < NCH; ++m) {
        if (SE_XRING < NCH && m >= SE_XRING && x_ct) {  // (the ring: chunk m's x own, issued at chunk m - 1's end)
            DeviceZoneScopedN("SE-XRELOAD");
            noc_async_read_barrier();
        }
        {
        DeviceZoneScopedN("SE-XROUNDS");
        // shard-major: each shard (own first, then as it lands) through every pass before the next shard
        for (uint32_t k = 0; k < g; ++k)
            for (uint32_t pg = 0; pg < NPG; ++pg)
            for (uint32_t st = 0; st < sps; ++st) {
                const uint32_t j = k * sps + st;  // (position; its family j % 2 = st % 2: a block's rounds are even)
                const uint32_t round = ((m * g + k) * NPG + pg) * (sps / 2) + st / 2;  // (the family's)
                WISSUE(m ? 0xFFFFFFFFu : (k * NPG + pg) * sps + st + SE_WPD);
#ifdef SE_XPIPE
                if (m == 0 && k == 0 && pg == 0) xload_upto(j + SE_XL);
#endif
                cb_reserve_back(CB_XIN, MT * KB);
                const uint32_t dst = xin + (j & 1) * MT * KB * 2048;
#ifdef SE_PROBE_NOXR
                (void)dst, (void)round;  // (probe: no x rounds at all - XIN pushed as if it landed)
#else
                if (mine(j)) {  // (position j: the shard of rank j / sps must be in - the own shard always is)
#ifndef SE_PROBE_NOXWAIT
                    if (j >= sps) noc_semaphore_wait_min(xready, m * (g - 1) + j / sps);
#endif
#ifdef SE_XPIPE
#ifdef SE_XN1
                    if (j < sps && x_ct && m == 0 && (j & 1) && !xports) {  // (the NCRISC's NOC1 load of it landed)
                        do {
                            invalidate_l1_cache();
                        } while (xsy[11] == 0);
                        while (!ncrisc_noc_read_with_transaction_id_flushed(1, 15)) {
                        }
                    } else
#endif
                    if (j < sps && x_ct)
                        while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, XTRID(m, j))) {
                        }  // (my own step's load landed)
#endif
#ifdef SE_PROBE_NOX
                    constexpr uint32_t XBYTES = 16;  // (timing probe: no x data)
#else
                    constexpr uint32_t XBYTES = MT * KB * 2048;
#endif
                    if (j & 1) s1->send(own_addr(m, j), dst, XBYTES); else s0->send(own_addr(m, j), dst, XBYTES);
                } else {
                    if (j & 1) r1->receive(round); else r0->receive(round);
                }
#endif
                cb_push_back(CB_XIN, MT * KB);
            }
        }
        WISSUE(0xFFFFFFFFu);  // (the rest - the down's blocks - before the h rounds, which block)
#ifdef SE_LAT
        {  // (the latent's all-to-all: once it is packed, +1 on each destination's port word of mine)
            DeviceZoneScopedN("SE-LHAND");
            const uint32_t np0 = ord0 + g, nports = se_ra(np0), r0 = np0 + 5 + 2 * nports;
            const uint32_t l0 = r0 + 2 + 2 * se_ra(r0), nl = se_ra(l0);
            const uint32_t lw = se_ra(l0 + 1);
            cb_wait_front(SE_LAT, (m + 1) * SE_LATP);  // (chunk m's block; the words count up over the chunks)
            for (uint32_t j = 0; j < nl; ++j)
                noc_semaphore_inc(get_noc_addr(se_ra(l0 + 2 + 2 * j), se_ra(l0 + 3 + 2 * j), lw), 1);
        }
#endif
#ifdef SE_NODOWN
        {  // (no down here: my h block [MT][po] to DRAM h [rows, HT] - the down + reduce-scatter reads it)
        DeviceZoneScopedN("SE-HOUT");
        const uint32_t np0 = ord0 + g, nports = se_ra(np0);
        const auto hg = TensorAccessor(h_args, se_ra(np0 + 4 + 2 * nports), 2048);
        cb_wait_front(CB_HOWN, MT * POMAX);
        for (uint32_t r = 0; r < MT; ++r)
            for (uint32_t t = 0; t < po; ++t)
                noc_async_write(hown + (r * POMAX + t) * 2048, hg.get_noc_addr(GROW(m, cy * MT + r) * HT + h0 + t), 2048);
        noc_async_write_barrier();
        cb_pop_front(CB_HOWN, MT * POMAX);
        }
#else
        {  // (its own zone: the hand-offs below are not h exchange)
        DeviceZoneScopedN("SE-HX");
        // h: every core's h block [MT][POMAX] into every row core's h-all slot (m % 2) as block cx ([GX][MT][POMAX]):
        // one kernel_lib rotating round a column; the receivers' readiness ack is the slot's credit
        constexpr uint32_t HA = GX * MT * POMAX;
        cb_wait_front(CB_HOWN, MT * POMAX);
        cb_reserve_back(CB_HALL, HA);
        const uint32_t hs = hall + (SE_HSL > 1 ? (m & 1) : 0) * HA * 2048;  // (h-all slots: SE_HSL)
        for (uint32_t c = 0; c < GX; ++c) {
            if (c == cx) sh->send(hown, hs + c * MT * POMAX * 2048, MT * POMAX * 2048);
            else rh->receive(m * GX + c);
#if defined(SE_RS) || defined(SE_HXINC)
            cb_push_back(CB_HALL, MT * POMAX);  // (column c's block: the down's first pass starts on its K tiles)
#endif
        }
#if !defined(SE_RS) && !defined(SE_HXINC)
        cb_push_back(CB_HALL, HA);
#endif
        cb_pop_front(CB_HOWN, MT * POMAX);
        }
#endif
#ifdef SE_RS
    {  // (down + reduce-scatter) each finished sub-block's hand-off: +1 on its port's mailbox word of mine (the words
       // count up over the chunks: the port's targets do)
        DeviceZoneScopedN("SE-HAND");
        const uint32_t np0 = ord0 + g, nports = se_ra(np0), r0 = np0 + 5 + 2 * nports;
        const uint32_t nh = se_ra(r0), mbox = se_ra(r0 + 1);
        for (uint32_t j = 0; j < nh; ++j) {
            cb_wait_front(SE_FINC, j + 1);
            noc_semaphore_inc(get_noc_addr(se_ra(r0 + 2 + 2 * j), se_ra(r0 + 3 + 2 * j), mbox), 1);
        }
        if constexpr (NCH > 1) cb_pop_front(SE_FINC, nh);  // (the next chunk's hand-offs from the front again)
    }
#endif
        if (SE_XRING < NCH && m + SE_XRING < NCH && x_ct) {
            // (the ring: chunk m + SE_XRING into chunk m's slot - its rounds sent, its prep done (my hand-offs above
            // waited for the chunk's fins, after its prep op): the load lands under the chunks between)
#ifdef SE_XFREE
            cb_wait_front(SE_XFREE, 1);
            cb_pop_front(SE_XFREE, 1);
#endif
            noc_async_writes_flushed();
            for (uint32_t j = 0; j < NSG; ++j) xload_step(m + SE_XRING, j);
        }
    }
#ifdef SE_XFREE
    {  // (the last chunks' credits, unused: popped - a call ends with the CB empty, the next one's credits its own)
        constexpr uint32_t left = SE_XRING < NCH ? SE_XRING : NCH;
        cb_wait_front(SE_XFREE, left);
        cb_pop_front(SE_XFREE, left);
    }
#endif
#ifdef SE_BRW
    noc_async_read_barrier();
#ifdef SE_BRW_BAR
    {  // (done; core SE_BAR re-arms the barrier once every core is past it)
        volatile tt_l1_ptr uint32_t* wsy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_WS);
        noc_semaphore_inc(get_noc_addr(SE_BAR_X, SE_BAR_Y, SE_WS + 36), 1);
        noc_async_atomic_barrier();
        if (cx == 0 && cy == 0) {
            do {
                invalidate_l1_cache();
            } while (wsy[9] < SE_BAR_N);
            wsy[8] = 0, wsy[9] = 0;
        }
    }
#endif
#endif
    noc_async_atomic_barrier();
}
"""

_NCRISC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t GY = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1), KB = get_compile_time_arg_val(2);
constexpr uint32_t NSG = get_compile_time_arg_val(3), NPG = get_compile_time_arg_val(4), NCH = get_compile_time_arg_val(5);
constexpr uint32_t KBD = get_compile_time_arg_val(6), NSD = get_compile_time_arg_val(7), NPD = get_compile_time_arg_val(8);
constexpr uint32_t NWS = get_compile_time_arg_val(9), WSLOT = get_compile_time_arg_val(10), WTB = get_compile_time_arg_val(11);
constexpr uint32_t NTH = get_compile_time_arg_val(12), CR = get_compile_time_arg_val(13), PWDMAX = get_compile_time_arg_val(14);
constexpr uint32_t WV = get_compile_time_arg_val(15), WFREE = get_compile_time_arg_val(16), NOUT = get_compile_time_arg_val(17);
constexpr auto w_args = TensorAccessorArgs<18>();
constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
#ifdef SE_LAT
constexpr auto l_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
#endif
#ifdef SE_PREP
constexpr auto p_args = TensorAccessorArgs<SE_PREP_CT0>();  // (the dispatch buffer [N S, W] row-major)
#endif
#ifdef SE_RTR
FORCE_INLINE volatile tt_l1_ptr uint32_t* ew(uint32_t o) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_EW + o); }
constexpr auto rx_args = TensorAccessorArgs<SE_RTR_CT0>();  // (the chip's x shard [rows, H / TP], DRAM)
constexpr auto rw_args = TensorAccessorArgs<rx_args.next_compile_time_args_offset()>();  // (W_r^T [E, H / TP])
#endif
constexpr uint32_t CB_WIN = 1, CB_OUT = 4, NB = NPG * NSG + NPD * NSD;  // (W blocks a chunk)
constexpr uint32_t NPD1 = NPD ? NPD : 1, NSD1 = NSD ? NSD : 1;  // (no down: 0, never divided by)
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i)); }
// the column's W stream: block b of a chunk (gate / up pass b / NSG, step b % NSG; then down) held by row b % GY,
// multicast down the column into W slot (round % NWS); every core pushes what lands and credits the slot's next holder
// (the args equal on every core of the chip go once, as common args [S, NC, those NC values]: arg i is the core's
// own arg i below S, common below S + NC, the core's arg i - NC past them - every index stays the full list's)
FORCE_INLINE uint32_t se_ra(uint32_t i) {
    const uint32_t s = get_common_arg_val<uint32_t>(0), nc = get_common_arg_val<uint32_t>(1);
    return i < s ? get_arg_val<uint32_t>(i) : i < s + nc ? get_common_arg_val<uint32_t>(2 + i - s) : get_arg_val<uint32_t>(i - nc);
}
void kernel_main() {
#ifdef SE_XN1
    {  // (the BRISC's family-1 x steps, on my NOC1, first thing: transaction id 15, then the issued flag)
        volatile tt_l1_ptr uint32_t* xsy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_WS);
        {
            do {
                invalidate_l1_cache();
            } while (xsy[10] == 0);
            const uint32_t n = xsy[512];
            noc_async_read_set_trid(15);
            for (uint32_t i = 0; i < n; ++i)
                noc_async_read((static_cast<uint64_t>(xsy[514 + 3 * i]) << 32) | xsy[513 + 3 * i], xsy[515 + 3 * i], 2048);
            noc_async_read_set_trid(0);
            xsy[11] = 1;
        }
    }
#endif
    size_t a = 0;
    // (mine first; from ty to the tail the chip's: se_ra's common span)
    const uint32_t cy = se_ra(a++), cxv = se_ra(a++);
    const uint32_t wbase = se_ra(a++), n0d = se_ra(a++);
    const uint32_t wbank = se_ra(a++);  // (the DRAM bank holding my W-own region)
#ifdef SE_RTR
    const uint32_t R0 = a;  // (the router logits' args)
    a += 12 + SE_RD * SE_MTR;
#endif
    const uint32_t wg0 = a, wd0 = wg0 + NPG;  // pass widths (gate / up, down)
    a = wd0 + NPD;
    const uint32_t ty = se_ra(a++), by = se_ra(a++);
    const uint32_t w_addr = se_ra(a++), wown = se_ra(a++), win = se_ra(a++);
    const uint32_t words = se_ra(a++), y_addr = se_ra(a++);
    const uint32_t ys0 = a;  // (the column's NoC y)
    const uint32_t sps = se_ra(ys0 + GY), ord0 = ys0 + GY + 2;  // (steps a shard; the shard order)
    // round w's block (natural index): a gate / up pass's steps in this chip's processing order, then down's
    auto bsize = [&](uint32_t b) {  // (tiles)
        return b < NPG * NSG ? KB * se_ra(wg0 + b / NSG) : KBD * se_ra(wd0 + (b - NPG * NSG) / NSD1);
    };
    // (a chunk's round -> block, and my blocks' offsets in my W-own, tabled once: computed in place, the round's block
    // (4 runtime divisions - sps is a runtime arg) and the offset's walk over my blocks ran on every pass of the
    // stream's loop below (the prefetch scan, the send's skip, the credit) - ~340 cycles a pass, a W round ~2.7
    // passes; tabled ~285 (K3 head handshake-only gate / up 48.8 -> 41.1 us, traced head ~-4 us; GLM-5.3 -5.6 us))
    static_assert(NB <= 1024, "W rounds a chunk");
    uint16_t rblk[NB], woffs[NB / GY + 1];
    for (uint32_t c = 0; c < NB; ++c) {
        uint32_t b = c;
        if (c < NPG * NSG) {
            const uint32_t k = c / (NPG * sps), rem = c % (NPG * sps), pg = rem / sps, st = rem % sps;
            b = pg * NSG + se_ra(ord0 + k) * sps + st;  // (shard-major: shard, pass, step)
        }
        rblk[c] = b;
    }
    for (uint32_t kb = 0, o = 0; cy + kb * GY < NB; ++kb) {  // (my blocks: cy, cy + GY, ..)
        woffs[kb] = o;
        o += bsize(cy + kb * GY);
    }
    auto blk = [&](uint32_t w) -> uint32_t { return rblk[w % NB]; };
    const auto yg = TensorAccessor(y_args, y_addr, 2048);
    // my blocks from DRAM, once: my W-own is one contiguous region in one bank (per-bank layout, wbase its offset),
    // read block by block in large bursts, every read issued now - the k-th block's under transaction id
    // min(k, 14) + 1; a block is sent once its reads landed (the column streams while the weights are still arriving)
#ifdef SE_RTR
    // the router's logits on the engine: the row's first core reads the row's x (MT row tiles, the chip's K slice) and
    // multicasts it along the row; the column's first core reads the column's W_r^T expert tiles and multicasts them
    // down the column; then the compute's partial logits go to the router cores' HOLD (the main loop below)
    // (args at R0: x, W, my column's expert tiles (valid count, first), RX / RW addresses, the row's NoC x (far end,
    // first core), my row's NoC y, the router cores' NoC [RB], their HOLD, their counter, my column has experts)
    const uint32_t etc = se_ra(R0 + 2), re0 = se_ra(R0 + 3);
    const uint32_t rxa = se_ra(R0 + 4), rwa = se_ra(R0 + 5);
    {
        // K chunks of SE_KC tiles, at most two rounds, a buffer each (no slot reused: no acks): the row's first core's
        // BRISC multicasts the row's x along the row (NOC0), the column's first core multicasts its W_r^T expert tiles
        // down the column (here, NOC1 - same-NoC row and column multicasts stall each other); a lead reads round
        // rd + 1 while round rd's multicast goes; every core pushes a round once its words say it landed
        DeviceZoneScopedN("SE-RTRLD");
#ifdef SE_RWL
        // (SE_RWL: the column's W_r^T lead on row NoC x % GY, not row 0: the NOC1 read responses come back along the
        // lead's row (-y, then -x), and 11 leads on one row shared its links - ~3 us a round's reads vs ~0.6 for the
        // x leads, who sit on different rows)
        const uint32_t wly = se_ra(ys0 + cxv % GY);
        const bool lw = etc && cy == cxv % GY;
#else
        const uint32_t wly = ty;
        const bool lw = etc && cy == 0;
#endif
        const bool hw = se_ra(R0 + 9);
        const auto wg2 = TensorAccessor(rw_args, se_ra(R0 + 1), 2048);
        constexpr uint32_t NRD = SE_KTO / SE_KC, WS = SE_ETC * SE_KC * 2048;
        // (two buffers: past two rounds a buffer is reused - every receiver acks a round once its compute popped it
        // (its next reserve went through): +1 on the row's x lead and the column's W lead, who wait for all of them)
        const uint32_t xlx = se_ra(R0 + 7), xly = se_ra(R0 + 8);
        for (uint32_t rd = 0; rd < NRD; ++rd) {
            cb_reserve_back(SE_RX, SE_KC * SE_MTR);
            cb_reserve_back(SE_RW, SE_ETC * SE_KC);
            if (NRD > 2 && rd >= 2) {
                noc_semaphore_inc(get_noc_addr(xlx, xly, SE_EW + XACK_ID), 1);
                if (hw) noc_semaphore_inc(get_noc_addr(cxv, wly, SE_EW + WACK_ID), 1);
            }
            if (lw) {  // (the column's first core: the chunk of its expert tiles of W_r^T down the column)
                const uint32_t wa = rwa + (rd & 1) * WS;
                if (NRD > 2 && rd >= 2) while (*ew(WACK_ID) < (rd - 1) * GY) invalidate_l1_cache();
                for (uint32_t e = 0; e < etc; ++e)
                    for (uint32_t k = 0; k < SE_KC; ++k)
                        noc_async_read(wg2.get_noc_addr((re0 + e) * SE_KTO + rd * SE_KC + k), wa + (e * SE_KC + k) * 2048, 2048);
                noc_async_read_barrier();
                noc_async_write_multicast_loopback_src(wa, get_noc_multicast_addr(cxv, by, cxv, ty, wa), etc * SE_KC * 2048, GY);
                noc_async_writes_flushed();
                *ew(RWR_ID) = rd + 1;
                noc_semaphore_set_multicast_loopback_src((SE_EW + RWR_ID), get_noc_multicast_addr(cxv, by, cxv, ty, (SE_EW + RWR_ID)), GY);
            }
            while (*ew(RXR_ID) < rd + 1) invalidate_l1_cache();
            cb_push_back(SE_RX, SE_KC * SE_MTR);
            while (hw && *ew(RWR_ID) < rd + 1) invalidate_l1_cache();
            cb_push_back(SE_RW, SE_ETC * SE_KC);
        }
        noc_async_writes_flushed();  // (my sets' source words read: re-arm them)
        *ew(RXR_ID) = 0;  // (re-armed: no more sets come this call - the x lead sets from its own word, RXS)
        *ew(RWR_ID) = 0;
        if (NRD > 2 && lw) {  // (every ack landed: the last were waited for)
            noc_semaphore_inc(get_noc_addr(SE_EW + WACK_ID), 0u - (NRD - 2) * GY);
            noc_async_atomic_barrier();
        }
    }
    {  // my partial logits (the compute's, a few us later) to their router cores' HOLD - before my W stream starts
        DeviceZoneScopedN("SE-RTROUT");
        cb_wait_front(SE_RO, SE_ETC * SE_MTR);
        // (row tile r of mine: my own owner's block into its router core's HOLD, another owner's straight into the
        // port sub-block it leaves from - (NoC, base, count) a row tile; the count ETC tiles, after they landed)
        const uint32_t ro = get_read_ptr(SE_RO);
        for (uint32_t r = 0; r < SE_MTR; ++r) {
#ifdef SE_RTR_SPLIT
            // (the row tile's experts over two router cores: tile g to half g / SE_RTR_SPLIT's destination, at
            // g % SE_RTR_SPLIT; each half counted its own tiles)
            const uint32_t a0 = R0 + 10 + SE_RD * r;
            uint32_t n[2] = {0, 0};
            for (uint32_t e = 0; e < etc; ++e) {
                const uint32_t g = re0 + e, hh = g >= SE_RTR_SPLIT ? 1 : 0;
                const uint32_t dw = se_ra(a0 + 3 * hh), base = se_ra(a0 + 3 * hh + 1);
                noc_async_write(ro + (e * SE_MTR + r) * 4096, get_noc_addr(dw & 0xFFFF, dw >> 16, base + (g - hh * SE_RTR_SPLIT) * 4096), 4096);
                ++n[hh];
            }
            noc_async_write_barrier();  // (landed before the counts)
            for (uint32_t hh = 0; hh < 2; ++hh) {
                const uint32_t dw = se_ra(a0 + 3 * hh), ctr = se_ra(a0 + 3 * hh + 2);
                if (n[hh]) noc_semaphore_inc(get_noc_addr(dw & 0xFFFF, dw >> 16, ctr), n[hh]);
            }
#else
            const uint32_t dw = se_ra(R0 + 10 + 3 * r), dx = dw & 0xFFFF, dy = dw >> 16;
            const uint32_t base = se_ra(R0 + 11 + 3 * r), ctr = se_ra(R0 + 12 + 3 * r);
            for (uint32_t e = 0; e < etc; ++e)
                noc_async_write(ro + (e * SE_MTR + r) * 4096, get_noc_addr(dx, dy, base + (re0 + e) * 4096), 4096);
            noc_async_write_barrier();  // (landed before the count)
            if (etc) noc_semaphore_inc(get_noc_addr(dx, dy, ctr), etc);
#endif
        }
        cb_pop_front(SE_RO, SE_ETC * SE_MTR);
#ifdef SE_RTR_GATE
        {  // (my partials are written: router core 0 counts every engine core's, then opens the W gate)
            const uint32_t gw = se_ra(R0 + 10 + SE_RD * SE_MTR);
            noc_semaphore_inc(get_noc_addr(gw & 0xFFFF, gw >> 16, se_ra(R0 + 11 + SE_RD * SE_MTR)), 1);
        }
#endif
    }
#ifdef SE_RDONE
    {  // my router phase is over - x rows / W_r^T read, the partials out: +1 on every port's "router done" word (the
       // ports credit this chip's arrival zones / tell the latent's senders "started" once every core's is: the router's
       // buffers overlay them - rtr_overlay)
        const uint32_t t0 = ord0 + se_ra(ys0 + GY + 1), np_ = se_ra(t0), rdw = se_ra(t0 + 3 + 2 * np_);
        for (uint32_t i = 0; i < np_; ++i) noc_semaphore_inc(get_noc_addr(se_ra(t0 + 2 + 2 * i), se_ra(t0 + 3 + 2 * i), rdw), 1);
    }
#endif
#ifdef SE_RTR_GATE
    {  // my W stream waits until every engine core's partials are written (RGATE, multicast by router core 0): the
       // streams' DRAM reads on NOC1 crowd the partials' writes into the mover column (GLM: in at ~47 us, not ~22)
        while (*ew(RGATE_ID) == 0) invalidate_l1_cache();
        *ew(RGATE_ID) = 0;  // (re-armed: the next call's comes after every core's next partials)
    }
#endif
    constexpr bool rout = true;
#else
    constexpr bool rout = true;
#endif
    {
        {  // (after my core's x load: the DRAM serves the x first - the gather is waiting on it)
            const uint32_t t0 = ord0 + se_ra(ys0 + GY + 1);  // (past the shard order: the tail)
            const uint32_t xld = se_ra(t0 + 2 + 2 * se_ra(t0));
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xld)) == 0) invalidate_l1_cache();
        }
    }
    // the weights' DRAM reads go in stream order a window ahead of the sends (WPF of my blocks in flight): spread
    // over gate / up at the rate the stream uses them - a burst of all 35 MB at the start starved the gather
    constexpr uint8_t WNOC = noc_index;  // (the W DRAM reads: the NCRISC's own NoC)
#ifdef SE_WPF
    constexpr uint32_t WPF = SE_WPF;  // (probe: the prefetch window)
#else
    constexpr uint32_t WPF = 3;
#endif
    uint8_t trid_of[NB / GY + 1];  // (a core's blocks: every GY-th of the chunk's NB)
    uint32_t wi = 0, ni = 0, pending = 0;  // (prefetch cursor over the first chunk's rounds; issued; issued unsent)
#ifdef SE_WRING
    // (one chunk: my W-own a ring of SE_WRING block slots - a block read into slot (its issue index % SE_WRING) once
    // the block SE_WRING before it was sent (multicast flushed); the DRAM region keeps every block)
    uint8_t slot_of[NB / GY + 1];
    uint32_t nsent = 0;
#endif
#ifdef SE_WRING
    constexpr uint32_t WPN = NCH * NB;  // (rounds the prefetch walks: every chunk reads its blocks again)
#else
    constexpr uint32_t WPN = NB;  // (the first chunk's: resident after it)
#endif
    auto offset_of = [&](uint32_t b) { return static_cast<uint32_t>(woffs[b / GY]); };  // (b: one of mine)
    uint32_t dn_go = 0;  // (SE_BRW_DN1: the down's NOC1 reads start once the stream passed round SE_DN1_AT)
    uint32_t wla = 0;  // (SE_WLA: rounds landed here - the window's base)
    auto prefetch = [&](uint32_t upto) {  // (issue my blocks of rounds < upto, and the window)
#if defined(SE_BRW) && !defined(SE_BRW_DN1)
        return;  // (the BRISC reads them on NOC0)
#endif
#ifdef SE_WLA
        // (SE_WLA: the window in the column's rounds, not my blocks - a block only once its round is within SE_WLA of
        // the stream; 3 of mine in flight on every core put ~30 rounds of the whole grid in the DRAM queues at once,
        // and the first-needed block landed ~50 us after the burst was issued)
        while (wi < WPN && (wi < upto || (pending < WPF && wi < wla + SE_WLA))) {
#else
        while (wi < WPN && (wi < upto || pending < WPF)) {
#endif
            const uint32_t b = blk(wi);
#ifdef SE_BRW_DN1
            if (wi % NB >= NPG * NSG && !dn_go && wi >= upto) break;  // (not yet: the first x loads go first)
            if (b % GY != cy || wi % NB < NPG * NSG) {  // (SE_BRW_DN1: the gate / up blocks are the BRISC's, on NOC0)
#else
            if (b % GY != cy) {
#endif
                ++wi;
                continue;
            }
#ifdef SE_WRING
            if (ni >= nsent + SE_WRING) break;  // (its slot still sending)
            const uint32_t lo = (ni % SE_WRING) * WSLOT;
            slot_of[b / GY] = ni % SE_WRING;
#endif
            ++wi;
            const uint32_t t = (ni++ % 14) + 1, o = offset_of(b);
#ifndef SE_WRING
            const uint32_t lo = o;
#endif
            trid_of[b / GY] = t;
            noc_async_read_set_trid(t, WNOC);
#if defined(SE_PROBE_NOWREAD)
            (void)lo;  // (probe: no W DRAM read at all - the block "lands" at once)
#elif defined(SE_PROBE_NOWL)
            noc_async_read(get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + o * WTB, WNOC), wown + lo * WTB, 16, WNOC);  // (probe)
#else
#ifdef SE_WPKT
            {  // (A/B: the block as single-packet reads of SE_WPKT bytes, one transaction id - no software split)
                const uint64_t src0 = get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + o * WTB, WNOC);
                const uint32_t nb = bsize(b) * WTB;
                for (uint32_t q = 0; q < nb; q += SE_WPKT)
                    noc_async_read(src0 + q, wown + lo * WTB + q, nb - q < SE_WPKT ? nb - q : SE_WPKT, WNOC);
            }
#else
            noc_async_read(get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + o * WTB, WNOC), wown + lo * WTB, bsize(b) * WTB, WNOC);
#endif
#endif
            ++pending;
        }
    };
    const uint32_t NR = NCH * NB;  // (W rounds a call)
    uint32_t ws = 0, woff = 0, wr = 0, wc = NWS, p = 0, pm = 0;  // (send, my offset, pushed, credited; y pass / chunk)
#ifdef SE_PROBE_NOWR
    constexpr bool NOWR_ = true;  // (probe: no W rounds - no multicast, no credits; WIN pushed as if every round landed)
    wc = NR;
#else
    constexpr bool NOWR_ = false;
#endif
    uint32_t nf[NWS] = {};
    while (ws < NR && blk(ws) % GY != cy) ++ws;  // (skip rounds I do not hold)
#ifdef SE_BRW
    volatile tt_l1_ptr uint32_t* wsy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_WS);
    uint32_t nmy = 0;  // (my blocks sent)
    {  // (my blocks in stream order for the BRISC's NOC0 reads: DRAM offset, L1 destination, bytes)
        uint32_t n = 0;
        for (uint32_t w = 0; w < WPN; ++w) {
            const uint32_t b = blk(w);
            if (b % GY != cy) continue;
#ifdef SE_BRW_DN1
            if (w % NB >= NPG * NSG) continue;  // (the down's: mine, on NOC1)
#endif
            const uint32_t o = offset_of(b);
            wsy[16 + 4 * n] = w_addr + wbase + o * WTB;
            wsy[17 + 4 * n] = wown + o * WTB;
            wsy[18 + 4 * n] = bsize(b) * WTB;
            // (its issue position in x rounds: a gate / up block its own round; the down's spread over the last SE_WDD
            // x rounds of the gate / up (they are needed right after it) - the BRISC's pacing)
            constexpr uint32_t NG = NPG * NSG, ND = NPD * NSD1, DD = SE_WDD < NG ? SE_WDD : NG;
            wsy[19 + 4 * n] = w < NG ? w : NG - DD + (w - NG) * DD / ND;
            ++n;
        }
        wsy[1] = 0, wsy[3] = wbank;
        wsy[0] = n | 0x80000000u;
    }
#endif
#ifdef SE_RS
    // (down + reduce-scatter) arrivals per direction in zone-slot order: slot i credited (its pages pushed - the arg's
    // high half: P, or with compact zones my tiles of its block) once my arrival counter reached its threshold (the
    // low half; 0: nothing comes - a block I hold no tile of); the fin sub-block to y
    const uint32_t t0r = ord0 + se_ra(ys0 + GY + 1);
    const uint32_t q0 = t0r + 4 + 2 * se_ra(t0r);  // (past the tail)
    const uint32_t naf = se_ra(q0), af0 = q0 + 1, nab = se_ra(af0 + naf), ab0 = af0 + naf + 1;
    const uint32_t f0 = ab0 + nab;
    const uint32_t fcount = se_ra(f0), fy0 = se_ra(f0 + 1);
    const uint32_t fstride = se_ra(f0 + 2), fw = se_ra(f0 + 3);
    volatile tt_l1_ptr uint32_t* arrf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(se_ra(f0 + 4));
    volatile tt_l1_ptr uint32_t* arrb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(se_ra(f0 + 5));
    uint32_t af = 0, ab = 0, fo = 0;
    const uint32_t NPY = 0;  // (no plain y writes: the fin sub-block is the output)
    // (chunks: the counters count up over them - a chunk's thresholds + its index x the arrivals a chunk; args at
    // f0 + 6: the fin's y0 of chunks 1 .., the ports facing my fwd / bwd upstream (NoC x | y << 16), their zone-free
    // word, my FINB-free word, the ports reading my FINB)
    #ifdef SE_FIN_ROWS  // (a core's MT rows straddle owner blocks: each (chunk, row)'s own y base, at f0 + 6 + NCH + 4)
    constexpr uint32_t CHX = NCH + 4 + NCH * MT, NFF = NCH - 1;
#else
    constexpr uint32_t CHX = NCH > 1 ? NCH + 4 : 0, NFF = NCH > 1 ? NCH - 1 : 0;
#endif
    uint32_t nfmax = 0, nbmax = 0, zuf = 0, zub = 0;  // (the thresholds' max; the zone pages my slots take)
    for (uint32_t i = 0; i < naf; ++i) {
        const uint32_t a = se_ra(af0 + i);
        nfmax = (a & 0xFFFF) > nfmax ? (a & 0xFFFF) : nfmax;
        zuf += a >> 16;
    }
    for (uint32_t i = 0; i < nab; ++i) {
        const uint32_t a = se_ra(ab0 + i);
        nbmax = (a & 0xFFFF) > nbmax ? (a & 0xFFFF) : nbmax;
        zub += a >> 16;
    }
    uint32_t zsf = 0, zsb = 0, ff = 0;  // (zone-free signals sent a direction; FINB-free credits pushed)
    const uint32_t zpf = NCH > 1 ? se_ra(f0 + 6 + NCH - 1) : 0, zpb = NCH > 1 ? se_ra(f0 + 6 + NCH) : 0;
    const uint32_t zfa = NCH > 1 ? se_ra(f0 + 6 + NCH + 1) : 0;
    volatile tt_l1_ptr uint32_t* ffw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(NCH > 1 ? se_ra(f0 + 6 + NCH + 2) : 0);
    const uint32_t fnr = NCH > 1 ? se_ra(f0 + 6 + NCH + 3) : 0;
#ifdef SE_LAT
    // (the latent: my rows of my own block - mine and every other chip's slice of them - to the owner's latent [RB, G LT])
    const uint32_t L0 = f0 + 6 + CHX, lnr = se_ra(L0), lr0 = se_ra(L0 + 1);
    const uint32_t lnsrc = se_ra(L0 + 2), lzone = se_ra(L0 + 4);
    volatile tt_l1_ptr uint32_t* lctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(se_ra(L0 + 3));
    const uint32_t LT = se_ra(L0 + 5), lc0 = se_ra(L0 + 6), llw = se_ra(L0 + 7);
    const uint32_t lp = se_ra(L0 + 8), lorow = se_ra(L0 + 9);
    const auto lg = TensorAccessor(l_args, se_ra(L0 + 10), 2048);
    // (chunks done: my own rows written; the arrivals in - chunk m's latent block at LAT + m LATP pages, its zone
    // LATZ + the same; the owner's rows of chunk m at m SE_LRB)
#ifdef SE_PREP
    const uint32_t NLO = 0, NLI = lnr == 0 || lnsrc == 0 ? 0 : NCH;  // (dispatch prep: the compute untilizes my rows' latent)
#else
    const uint32_t NLO = lnr == 0 ? 0 : NCH, NLI = lnr == 0 || lnsrc == 0 ? 0 : NCH;
#endif
    uint32_t lo = 0, li = 0;
    auto lwrite = [&](uint32_t src, uint32_t q, uint32_t m) {  // (rows lr0 .. of a [MT][LWMAX] block, chip q's columns)
        for (uint32_t i = 0; i < lnr; ++i)
            for (uint32_t t = 0; t < llw; ++t)
                noc_async_write(src + ((lr0 + i) * SE_LWMAX + t) * 2048, lg.get_noc_addr((lorow + m * SE_LRB + i) * G_LT + q * LT + lc0 + t), 2048);
    };
#else
    constexpr uint32_t NLO = 0, NLI = 0, lo = 0, li = 0;
#endif
#ifdef SE_PREP  // (dispatch prep: per unit the dispatch buffer's row, byte offset in it, bytes a row)
#ifdef SE_LAT
    const uint32_t P0 = f0 + 6 + 11 + CHX;
#else
    const uint32_t P0 = f0 + 6 + CHX;
#endif
    const uint32_t pnu = se_ra(P0);
    const auto pg = TensorAccessor(p_args, se_ra(P0 + 1), SE_PREP_ROW);
    uint32_t fsrc = 0, frow = 0, fco = 0;  // (the last prep unit: the fence's source / rows - mine: nothing else writes them)
#else
    constexpr uint32_t pnu = 0;
#endif
    uint32_t pu = 0;
#ifdef SE_HW
    uint32_t hw_it = 0, hw_last[4] = {};
#endif
#ifdef SE_PROBE_WLOOP
    uint32_t wl_it = 0, wl_t0 = 0, wl_t1 = 0;  // (the loop's passes while the first chunk's gate / up W rounds land; cycles)
#endif
    while (ws < NR || wr < NR || wc < NR || af < NCH * naf || ab < NCH * nab || fo < NCH * fcount || ff < NFF || lo < NLO || li < NLI || pu < pnu || !rout) {
        invalidate_l1_cache();
#ifdef SE_PROBE_WLOOP
        if (wr < NPG * NSG) {
            if (wl_it++ == 0) wl_t0 = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
            wl_t1 = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
        }
#endif
#ifdef SE_HW
        if ((++hw_it & 0x3FF) == 0) {  // (a snapshot every 1024 polls: per-poll stores hid the hang)
        HWW(0, 0xE0000000u | (hw_it & 0xFFFFFF)); HWW(1, ws); HWW(2, wr); HWW(3, wc); HWW(4, af); HWW(5, ab); HWW(6, fo);
        HWW(7, lo | li << 8 | (uint32_t)rout << 16); HWW(8, pu); HWW(9, *arrf); HWW(10, *arrb);
        HWW(11, af < NCH * naf ? se_ra(af0 + af % naf) : 0xFFFF); HWW(12, ab < NCH * nab ? se_ra(ab0 + ab % nab) : 0xFFFF);
#ifdef SE_LAT
        HWW(13, *lctr);
#endif
        HWW(14, NR); HWW(15, naf | nab << 8 | fcount << 16);
        for (uint32_t i = 0; i < NWS && i < 4; ++i) { HWW(19 + i, *sem(WV + i)); HWW(23 + i, *sem(WFREE + i)); }
        HWW(27, get_semaphore(WV)); HWW(28, get_semaphore(WFREE));
        }
        for (uint32_t i = 0; i < NWS && i < 4; ++i) {  // (a slot's W-ready word (its latest round + 1) going backwards,
            // or past the rounds that may have landed (< wr + NWS: they need my credit) - a stray write: record, halt)
            const uint32_t v = *sem(WV + i), lim = wr + NWS;
            if (v < hw_last[i] || v > lim) {
                HWW(29, (v < hw_last[i] ? 0xDEC00000u : 0xA4E00000u) | i << 16 | wr); HWW(30, hw_last[i]); HWW(31, v);
                while (true) {}
            }
            hw_last[i] = v;
        }
#endif
#ifdef SE_LAT
        if (lo < NLO && cb_pages_available_at_front(SE_LAT, (lo + 1) * SE_LATP)) {
            lwrite(get_read_ptr(SE_LAT) + lo * SE_LATP * 2048, lp, lo);
            ++lo;
        }
        if (li < NLI && *lctr >= (li + 1) * lnsrc) {
#ifdef SE_PREP
            cb_reserve_back(SE_LRDY, 1);  // (the compute's prep may untilize the arrivals)
            cb_push_back(SE_LRDY, 1);
#else
            for (uint32_t sl = 0; sl < lnsrc; ++sl)
                lwrite(lzone + li * SE_LATP * 2048 + sl * MT * SE_LWMAX * 2048, sl < lp ? sl : sl + 1, li);
#endif
            ++li;
        }
#endif
#ifdef SE_PREP
        if (pu < pnu && cb_pages_available_at_front(SE_PO, SE_PREP_FULL)) {  // a prep unit's 32 rows to the buffer
            const uint32_t src = get_read_ptr(SE_PO), q = P0 + 2 + 3 * pu;
            const uint32_t row = se_ra(q), co = se_ra(q + 1), nb = se_ra(q + 2);
            // (posted: non-posted DRAM writes on NOC1 - their acks - under the W stream's NOC1 DRAM reads wedged a chip's
            // NOC1, ~1 in 200 calls with the prep in every chunk; landed by the fence at the end: test_posted_fence)
            for (uint32_t i = 0; i < 32; ++i) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(src + i * SE_PREP_FULL * 64, pg.get_noc_addr(row + i) + co, nb);
            }
            noc_async_posted_writes_flushed();  // (PO may be refilled once they left L1)
            fsrc = src, frow = row, fco = co;
            cb_pop_front(SE_PO, SE_PREP_FULL);
            ++pu;
        }
#endif
        if (af < NCH * naf) {
            const uint32_t sl = af % naf, mc = af / naf;
            if (NCH > 1 && sl == 0 && mc > zsf) {  // (chunk mc's first slot: chunk mc - 1's zone all read (popped) -
                // +1 on the zone-free word of the port facing my fwd upstream: it credits the upstream's sends)
                if (cb_pages_reservable_at_back(SE_ARRF, SE_ZF)) {
                    // (only a core something lands on signals: one with only "nothing comes" slots runs ahead of
                    // the credit - its signals would count for a slower one's)
                    if (nfmax) noc_semaphore_inc(get_noc_addr(zpf & 0xFFFF, zpf >> 16, zfa), 1);
                    ++zsf;
                }
            } else {
                const uint32_t a = se_ra(af0 + sl), thr = a & 0xFFFF, pz = a >> 16;
                if (thr == 0 || *arrf >= mc * nfmax + thr) {
                    if (pz) {
                        cb_reserve_back(SE_ARRF, pz);
                        cb_push_back(SE_ARRF, pz);
                    }
                    ++af;
                    if (NCH > 1 && sl + 1 == naf && SE_ZF > zuf) {  // (the chunk padded to the whole zone: the
                        // compute pops it all, the next chunk's slots from the zone's start again)
                        cb_reserve_back(SE_ARRF, SE_ZF - zuf);
                        cb_push_back(SE_ARRF, SE_ZF - zuf);
                    }
                }
            }
        }
        if (ab < NCH * nab) {
            const uint32_t sl = ab % nab, mc = ab / nab;
            if (NCH > 1 && sl == 0 && mc > zsb) {
                if (cb_pages_reservable_at_back(SE_ARRB, SE_ZB)) {
                    if (nbmax) noc_semaphore_inc(get_noc_addr(zpb & 0xFFFF, zpb >> 16, zfa), 1);
                    ++zsb;
                }
            } else {
                const uint32_t a = se_ra(ab0 + sl), thr = a & 0xFFFF, pz = a >> 16;
                if (thr == 0 || *arrb >= mc * nbmax + thr) {
                    if (pz) {
                        cb_reserve_back(SE_ARRB, pz);
                        cb_push_back(SE_ARRB, pz);
                    }
                    ++ab;
                    if (NCH > 1 && sl + 1 == nab && SE_ZB > zub) {
                        cb_reserve_back(SE_ARRB, SE_ZB - zub);
                        cb_push_back(SE_ARRB, SE_ZB - zub);
                    }
                }
            }
        }
        if (fo < NCH * fcount && cb_pages_available_at_front(SE_OUTF, 1)) {
            const uint32_t mc = fo / fcount, i = fo % fcount;
#ifdef SE_FIN_ROWS
            const uint32_t ya = se_ra(f0 + 6 + NCH + 4 + mc * MT + i / fw) + i % fw;
#else
            const uint32_t y0 = mc == 0 ? fy0 : se_ra(f0 + 6 + mc - 1);  // (chunk mc's rows)
            const uint32_t ya = y0 + (i / fw) * fstride + i % fw;
#endif
            cb_wait_front(SE_OUTF, 1);
            noc_async_write(get_read_ptr(SE_OUTF), yg.get_noc_addr(ya), 2048);
            noc_async_writes_flushed();  // (the page read out: free for the next fin tiles - its ack is the kernel end's
                                         // barrier; a barrier a tile serialized ~35 DRAM round trips behind the last fin)
            cb_pop_front(SE_OUTF, 1);
            ++fo;
        }
        if (ff < NFF && *ffw >= (ff + 1) * fnr) {  // (the ports read chunk ff's FINB: the compute may write it again)
            cb_reserve_back(SE_FFREE, 1);
            cb_push_back(SE_FFREE, 1);
            ++ff;
        }
#else
    const uint32_t NPY = NCH * NPD;
    while (ws < NR || wr < NR || wc < NR || p < NPY) {
        invalidate_l1_cache();
#endif
#ifdef SE_BRW_DN1
        if (!dn_go && wr >= SE_DN1_AT) dn_go = 1;
#endif
        if (wi < WPN) prefetch(0);  // (keep the window full)
        if (ws < NR && (NOWR_ || ws < NWS || *sem(WFREE + ws % NWS) >= GY * (nf[ws % NWS] + 1))) {  // my block: down the column
            const uint32_t b = blk(ws), n = bsize(b), kb = b / GY;  // (kb: its index among my blocks)
            woff = 0;  // (its offset in my W-own: my blocks before it)
#ifdef SE_WRING
            woff = slot_of[kb] * WSLOT;
#else
            woff = offset_of(b);
#endif
            if (ws < WPN) {  // (first chunk, or every chunk with the ring: its reads issued and landed)
#ifdef SE_BRW
#ifdef SE_BRW_DN1
              if (ws % NB >= NPG * NSG) {  // (a down block: my own NOC1 read)
                prefetch(ws + 1);
                noc_async_read_barrier_with_trid(trid_of[kb], WNOC);
                --pending;
              } else {
#endif
                do {
                    invalidate_l1_cache();
                } while (wsy[1] <= nmy);  // (the BRISC issued it)
                while (!ncrisc_noc_read_with_transaction_id_flushed(0, (nmy < 11 ? nmy : 11) + 1)) {
                }  // (and its NOC0 read landed: NOC0's status, read only)
                ++nmy;
#ifdef SE_BRW_DN1
              }
#endif
#else
                prefetch(ws + 1);
                noc_async_read_barrier_with_trid(trid_of[kb], WNOC);
                --pending;
#endif
            }
            const uint32_t dst = win + (ws % NWS) * WSLOT * WTB, vw = words + (ws & 3) * 16;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vw) = ws + 1;
#ifdef SE_PROBE_NOWR
            (void)dst;
#elif defined(SE_PROBE_NOW)
            noc_async_write_multicast_loopback_src(wown + woff * WTB, get_noc_multicast_addr(cxv, by, cxv, ty, dst), 16, GY);  // (probe)
#else
            noc_async_write_multicast_loopback_src(wown + woff * WTB, get_noc_multicast_addr(cxv, by, cxv, ty, dst), n * WTB, GY);
#endif
#ifndef SE_PROBE_NOWR
            noc_async_writes_flushed();
            noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(cxv, by, cxv, ty, get_semaphore(WV + ws % NWS)), GY);
            noc_async_writes_flushed();
#endif
            if (ws >= NWS) ++nf[ws % NWS];
            ++ws;
#ifdef SE_WRING
            ++nsent;  // (flushed: its slot may take a later block)
#endif
            while (ws < NR && blk(ws) % GY != cy) ++ws;
        }
        if (wr < NR && (NOWR_ || *sem(WV + wr % NWS) >= wr + 1)) {  // W round wr landed here
            cb_reserve_back(CB_WIN, WSLOT);
            cb_push_back(CB_WIN, WSLOT);
            wla = ++wr;
        }
        if (!NOWR_ && wc < NR && wc < wr + NWS && cb_pages_reservable_at_back(CB_WIN, WSLOT * (wc - wr + 1))) {  // slot of wc free
            noc_semaphore_inc(get_noc_addr(cxv, se_ra(ys0 + blk(wc) % GY), get_semaphore(WFREE + wc % NWS)), 1);
            ++wc;
        }
        if (p < NPY && cb_pages_available_at_front(CB_OUT, MT * PWDMAX)) {  // down pass p's y (chunk p / NPD)
            cb_wait_front(CB_OUT, MT * PWDMAX);
            const uint32_t m = p / NPD1, pd = p % NPD1, l1 = get_read_ptr(CB_OUT);
            uint32_t c0 = n0d;
            for (uint32_t i = 0; i < pd; ++i) c0 += se_ra(wd0 + i);
            const uint32_t w = se_ra(wd0 + pd);
            for (uint32_t r = 0; r < MT; ++r)
                for (uint32_t t = 0; t < w; ++t)
                    noc_async_write(l1 + (r * PWDMAX + t) * 2048, yg.get_noc_addr(GROW(m, cy * MT + r) * NTH + c0 + t), 2048);
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, MT * PWDMAX);
            ++p;
        }
    }
#ifdef SE_PREP
    if (pu) {  // (the prep's fence: a non-posted rewrite of 4 B in every DRAM bank - the last unit's rows, one a bank -
        // after the posted writes on the same VC: its ack means they all landed)
        for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) noc_async_write(fsrc + i * SE_PREP_FULL * 64, pg.get_noc_addr(frow + i) + fco, 4);
    }
#endif
    noc_async_write_barrier();
#ifdef SE_PROBE_WLOOP
    DeviceTimestampedData("SE-WLIT", wl_it);
    DeviceTimestampedData("SE-WLCY", wl_t1 - wl_t0);
#endif
#ifdef SE_BRW
    wsy[0] = 0, wsy[1] = 0;  // (re-arm: every block landed and sent)
#endif
#ifdef SE_XN1
    {
        volatile tt_l1_ptr uint32_t* xsy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SE_WS);
        noc_async_read_barrier();
        xsy[10] = 0, xsy[11] = 0;  // (re-arm)
    }
#endif
#ifdef SE_RS
    {  // re-arm my arrival counters (every increment of the call landed: each was waited to its final count)
        if (nfmax) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrf)), 0u - NCH * nfmax);
        if (nbmax) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrb)), 0u - NCH * nbmax);
        if (NFF && fnr) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(ffw)), 0u - NFF * fnr);
    }
#endif
#ifdef SE_LAT
    if (lnr && lnsrc) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(lctr)), 0u - NCH * lnsrc);  // (re-arm)
#endif
    noc_async_atomic_barrier();
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#ifdef SE_PREP
#include "api/compute/pack_untilize.h"
#endif
#include "api/debug/kernel_profiler.hpp"
#if defined(SE_GLU) && defined(TRISC_PACK)
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if SE_GLU == 2
#include "ckernel_sfpu_situ_glu.h"
#elif SE_GLU == 3
#include "ckernel_sfpu_clamped_silu_glu.h"
#elif SE_GLU == 4
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#endif
namespace ckernel::sfpu {
// (the model's activation constants, compile-time: SiTU's betas, the clamp limit, SwiGLU-OAI's alpha / limit; their
// defines come with the kernel source - se_engine3.glu_defines)
struct SeGluCfg {
    static constexpr float beta_gate = SE_GLU_BETA, beta_up = SE_GLU_LBETA;
    static constexpr float limit = SE_GLU_LIMIT, clamp_limit = SE_GLU_LIMIT, alpha = SE_GLU_ALPHA;
};
}  // namespace ckernel::sfpu
#endif
constexpr uint32_t MT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1), NSG = get_compile_time_arg_val(2);
constexpr uint32_t NPG = get_compile_time_arg_val(3), NCH = get_compile_time_arg_val(4), HT = get_compile_time_arg_val(5);
constexpr uint32_t POMAX = get_compile_time_arg_val(6), KBD = get_compile_time_arg_val(7), NSD = get_compile_time_arg_val(8);
constexpr uint32_t NPD = get_compile_time_arg_val(9), WSLOT = get_compile_time_arg_val(10), PWDMAX = get_compile_time_arg_val(11);
constexpr uint32_t G = get_compile_time_arg_val(12), SPS = get_compile_time_arg_val(13);  // (shards, steps a shard)
constexpr uint32_t GX = get_compile_time_arg_val(14), HA = GX * MT * POMAX;  // (h-all: [column][MT][POMAX])
// Words [0, NC) - the per-group block (gate / down tables, h column offsets, HT) and the RS / prep tables' prefix every
// core of the group shares - are this kernel's common args (sent once a chip, not once a core); the rest of a core's
// tables are its runtime args from NC on. Indices stay absolute (the host's tables point at them).
constexpr uint32_t SE_NC = get_compile_time_arg_val(15);
FORCE_INLINE uint32_t se_arg(uint32_t i) {
    return i < SE_NC ? get_common_arg_val<uint32_t>(i) : get_arg_val<uint32_t>(i - SE_NC);
}
constexpr uint32_t XIN = 0, WIN = 1, HOWN = 2, HALL = 3, OUT = 4, ACC = 5;
// (runtime args: the column's gate / up pass widths [NPG], then down pass widths [NPD])
#ifdef SE_RTR
// the router's matmul at HiFi3 (the engine runs HiFi2: the routing picks need HiFi3) - matmul_block's LLK calls with
// the fidelity fixed
ALWI void rmm_init(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 1, ct, 1, kt)));
    MATH((llk_math_matmul_init<MathFidelity::HiFi3, MM_THROTTLE>(in0, in1, 1, ct, 1)));
}
ALWI void rmm(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t ct, uint32_t kt) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, 1, kt)));
    MATH((llk_math_matmul<MathFidelity::HiFi3, MM_THROTTLE>(idst, ct, 1)));
}
#endif
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(XIN, WIN, HOWN);
#ifdef SE_RTR
    {  // the router's logits first (every chain starts here): my expert tiles x my MT rows over the chip's K slice
        DeviceZoneScopedN("SE-RTRMM");
        constexpr uint32_t etc = SE_ETC;  // (every core the same count; past the experts: unused tiles)
        {  // (a K chunk's ETC x MT partials in DEST, then fp32 into RO with the packer's L1 accumulation)
            reconfig_data_format(SE_RX, SE_RW);
            pack_reconfig_data_format(SE_RO);
            // (row groups of SE_RG rows: ETC x SE_RG tiles in DEST - all rows at once where they fit)
            rmm_init(SE_RW, SE_RX, SE_RG, SE_KC);
            cb_reserve_back(SE_RO, etc * SE_MTR);
            for (uint32_t rd = 0; rd < SE_KTO / SE_KC; ++rd) {
                cb_wait_front(SE_RX, SE_KC * SE_MTR);
                cb_wait_front(SE_RW, etc * SE_KC);
                PACK((llk_pack_reconfig_l1_acc(rd > 0 ? 1 : 0)));
                for (uint32_t g = 0; g < SE_MTR; g += SE_RG) {
                    tile_regs_acquire();
                    for (uint32_t e = 0; e < etc; ++e)
                        for (uint32_t k = 0; k < SE_KC; ++k) rmm(SE_RW, SE_RX, e * SE_KC + k, k * SE_MTR + g, e * SE_RG, SE_RG, SE_KC);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t e = 0; e < etc; ++e)
                        for (uint32_t r = 0; r < SE_RG; ++r) pack_tile<true>(e * SE_RG + r, SE_RO, e * SE_MTR + g + r);
                    tile_regs_release();
                }
                cb_pop_front(SE_RX, SE_KC * SE_MTR);
                cb_pop_front(SE_RW, etc * SE_KC);
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_push_back(SE_RO, etc * SE_MTR);
            reconfig_data_format(WIN, XIN);  // (back to gate / up's formats)
        }
    }
#endif
#if defined(SE_GLU)
#if SE_GLU == 2
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
#elif SE_GLU == 3
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
#elif SE_GLU == 4
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_init)));
#else
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
#endif
#ifdef SE_PROBE_GUWAIT
    uint32_t gu_wx = 0, gu_ww = 0;  // (the unpacker's cycles waiting for x / W in gate / up: SE-GUWX / SE-GUWW)
    uint32_t gu_ns = 0, gs_x = 0, gs_w = 0;
#endif
    uint32_t opos = 0;  // (OUTF tiles pushed: its blocks end at multiples of 8 - the NCRISC reads it page by page,
                        // pack_tile does not wrap; the chunks' fins continue where the last one left it)
    for (uint32_t m = 0; m < NCH; ++m) {
        // gate / up: per pass all of K in DEST, the GLU, h packed once
        {
        DeviceZoneScopedN("SE-GU");
        if (NCH > 1 && m > 0) reconfig_data_format(WIN, XIN);  // (the last chunk's down / RS left other formats)
        cb_reserve_back(HOWN, MT * POMAX);
        // shard-major: shard k (own first, then as each lands) through every pass; a pass's fp32 partial is packed
        // onto its fixed ACC slot (MT x pw tiles at the pass's offset) with the packer's L1 accumulation - shard 0
        // writes it, shards 1 .. G - 2 add onto it in L1 - and only the last shard brings the sum back into DEST
        // (unpacked straight to DEST, fp32) under its own matmuls; then the GLU and h. Shard G - 2 pushes each pass's
        // slot as it is done (the write pointer then sits at the next pass's slot); the chunk's tail is pushed and
        // popped as padding, so every chunk starts at the region's start
#if SE_DT == 4
        // (half-sync DEST: a pass keeps its width - the same x / W rounds as full sync - but each step's MT x pw block
        // goes through DEST as sub-blocks of <= 4 tiles (rows), each packed onto the pass's ACC slot with the
        // packer's L1 accumulation (the first step of shard 0 writes it). The packer, and the last step's GLU, run on
        // one half while the matmuls fill the other. The last step brings the slot's sum back into DEST first; the
        // step before it pushes the slot (its sum complete but for the last step's), in pass order)
        constexpr uint32_t LAST = G * SPS - 1;
        if (LAST > 0) cb_reserve_back(ACC, SE_ACCP);
        uint32_t apushed = 0;
        for (uint32_t k = 0; k < G; ++k) {
            uint32_t hoff = 0, loff = 0, aoff = 0;
            for (uint32_t pg = 0; pg < NPG; ++pg) {
                const uint32_t pw = se_arg(pg), n = MT * pw;
                const uint32_t sbr = n > 4 ? 1 : MT, nsb = MT / sbr, sbn = sbr * pw;  // (rows a sub-block)
                matmul_block_init(XIN, WIN, 0, pw, sbr, KB);
                for (uint32_t st = 0; st < SPS; ++st) {
                    const uint32_t sx = k * SPS + st;
#ifdef SE_PROBE_GUWAIT
                    UNPACK(({
                        const uint32_t t0_ = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
                        cb_wait_front(XIN, MT * KB);
                        const uint32_t t1_ = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
                        cb_wait_front(WIN, WSLOT);
                        const uint32_t t2_ = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
                        gu_wx += t1_ - t0_, gu_ww += t2_ - t1_;
                        gs_x = t1_ - t0_, gs_w = t2_ - t1_;
                    }));
#ifdef SE_PROBE_GUSTEP
                    UNPACK(({  // (GUSTEP: the first 48 steps' x / W waits - the profiler's buffer)
                        if (gu_ns++ < 48) {
                            DeviceTimestampedData("SE-GUSX", gs_x);
                            DeviceTimestampedData("SE-GUSW", gs_w);
                        }
                    }));
#endif
#endif
                    cb_wait_front(XIN, MT * KB);
                    cb_wait_front(WIN, WSLOT);
                    const bool fin = sx == LAST;
                    if (fin && LAST > 0) cb_wait_front(ACC, n);
                    for (uint32_t sb = 0; sb < nsb; ++sb) {
                        tile_regs_acquire();
                        if (fin && LAST > 0) {
                            reconfig_data_format_srca(WIN, ACC);
                            copy_init(ACC);
                            for (uint32_t t = 0; t < sbn; ++t) copy_tile(ACC, sb * sbn + t, t);
                            reconfig_data_format_srca(ACC, WIN);
                            matmul_block_init(XIN, WIN, 0, pw, sbr, KB);
                        }
#ifndef SE_PROBE_NOMATH
                        for (uint32_t kk = 0; kk < KB; ++kk)
                            matmul_block(XIN, WIN, sb * sbr * KB + kk, kk * pw, 0, false, pw, sbr, KB);
#endif
                        tile_regs_commit();
                        tile_regs_wait();
                        if (!fin) {
                            pack_reconfig_data_format(ACC);
                            PACK((llk_pack_reconfig_l1_acc(sx > 0 ? 1 : 0)));
#ifndef SE_PROBE_NOACCPACK
                            for (uint32_t t = 0; t < sbn; ++t) pack_tile<true>(t, ACC, aoff + sb * sbn + t - apushed);
#endif
                            PACK((llk_pack_reconfig_l1_acc(0)));
                            tile_regs_release();
                            continue;
                        }
#ifdef SE_LAT
                        if (pg >= SE_NGU) {  // (a latent-down pass: no GLU, the chip's latent columns packed bf16 to LAT)
                            pack_reconfig_data_format(SE_LAT);
                            for (uint32_t r = 0; r < sbr; ++r)
                                for (uint32_t t = 0; t < pw; ++t)
                                    pack_tile<true>(r * pw + t, SE_LAT, (sb * sbr + r) * SE_LWMAX + loff + t);
                            tile_regs_release();
                            continue;
                        }
#endif
                        pack_reconfig_data_format(HOWN);
#if defined(SE_GLU) && !defined(SE_PROBE_NOGLU)
                        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if SE_GLU == 2
                        for (uint32_t t = 0; t < sbn; t += 2)
                            PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_situ_glu, (DST_ACCUM_MODE, 8, sfpu::SeGluCfg), t, t + 1, t, VectorMode::RC)));
#elif SE_GLU == 3  // (silu(min(gate, 10)) x clamp(up, +-10): GLM-5.3's swiglu_limit, DeepSeek V4's)
                        for (uint32_t t = 0; t < sbn; t += 2)
                            PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_clamped_silu_glu, (DST_ACCUM_MODE, 8, sfpu::SeGluCfg), t, t + 1, t, VectorMode::RC)));
#elif SE_GLU == 4  // (SwiGLU-OAI: (clamp(up, +-limit) + 1) x g sigmoid(alpha g), g = min(gate, limit) - MiniMax M3)
                        for (uint32_t t = 0; t < sbn; t += 2)
                            PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_swiglu, (DST_ACCUM_MODE, 8, sfpu::SeGluCfg, true), t, t + 1, t, VectorMode::RC)));
#else
                        for (uint32_t t = 0; t < sbn; t += 2) silu_tile_pack(t);
                        for (uint32_t t = 0; t < sbn; t += 2)
                            PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul, (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE), t, t + 1, t, VectorMode::RC)));
#endif
                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
                        for (uint32_t r = 0; r < sbr; ++r)
                            for (uint32_t t = 0; t < pw; t += 2)
                                pack_tile<true>(r * pw + t, HOWN, (sb * sbr + r) * POMAX + hoff + t / 2);
                        tile_regs_release();
                    }
                    if (fin && LAST > 0) cb_pop_front(ACC, n);
                    cb_pop_front(XIN, MT * KB);
                    cb_pop_front(WIN, WSLOT);
                    if (sx + 1 == LAST) {  // (the slot's sum but for the last step: the last step may read it)
                        cb_push_back(ACC, n);
                        apushed += n;
                    }
                }
#ifdef SE_LAT
                if (pg >= SE_NGU) loff += pw; else
#endif
                hoff += pw / 2;
                aoff += n;
            }
        }
#else
        if (G > 1) cb_reserve_back(ACC, SE_ACCP);
        uint32_t apushed = 0;  // (ACC pages pushed this chunk - the pack offsets are relative to the write pointer)
        for (uint32_t k = 0; k < G; ++k) {
            uint32_t hoff = 0, loff = 0, aoff = 0;
            for (uint32_t pg = 0; pg < NPG; ++pg) {
                const uint32_t pw = se_arg(pg), n = MT * pw;
                tile_regs_acquire();
                if (k + 1 == G && k > 0) {
                    cb_wait_front(ACC, n);
                    reconfig_data_format_srca(WIN, ACC);
                    copy_init(ACC);
                    for (uint32_t t = 0; t < n; ++t) copy_tile(ACC, t, t);
                    cb_pop_front(ACC, n);
                    reconfig_data_format_srca(ACC, WIN);
                }
                matmul_block_init(XIN, WIN, 0, pw, MT, KB);
                for (uint32_t st = 0; st < SPS; ++st) {
#ifdef SE_PROBE_GUWAIT
                    UNPACK(({
                        const uint32_t t0_ = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
                        cb_wait_front(XIN, MT * KB);
                        const uint32_t t1_ = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
                        cb_wait_front(WIN, WSLOT);
                        const uint32_t t2_ = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
                        gu_wx += t1_ - t0_, gu_ww += t2_ - t1_;
                        gs_x = t1_ - t0_, gs_w = t2_ - t1_;
                    }));
#ifdef SE_PROBE_GUSTEP
                    UNPACK(({  // (GUSTEP: the first 48 steps' x / W waits - the profiler's buffer)
                        if (gu_ns++ < 48) {
                            DeviceTimestampedData("SE-GUSX", gs_x);
                            DeviceTimestampedData("SE-GUSW", gs_w);
                        }
                    }));
#endif
#endif
                    cb_wait_front(XIN, MT * KB);
                    cb_wait_front(WIN, WSLOT);
#ifndef SE_PROBE_NOMATH
                    for (uint32_t kk = 0; kk < KB; ++kk) matmul_block(XIN, WIN, kk, kk * pw, 0, false, pw, MT, KB);
#endif
                    cb_pop_front(XIN, MT * KB);
                    cb_pop_front(WIN, WSLOT);
                }
                tile_regs_commit();
                if (k + 1 < G) {
                    tile_regs_wait();
                    pack_reconfig_data_format(ACC);
                    PACK((llk_pack_reconfig_l1_acc(k > 0 ? 1 : 0)));
#ifndef SE_PROBE_NOACCPACK
                    for (uint32_t t = 0; t < n; ++t) pack_tile<true>(t, ACC, aoff + t - apushed);
#endif
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    tile_regs_release();
                    aoff += n;
                    if (k + 2 == G) {  // (its sum is complete but for the last shard's: the last shard may read it)
                        cb_push_back(ACC, n);
                        apushed += n;
                    }
                    continue;
                }
                tile_regs_wait();
#ifdef SE_LAT
                if (pg >= SE_NGU) {  // (a latent-down pass: no GLU, the chip's latent columns packed bf16 to LAT)
                    pack_reconfig_data_format(SE_LAT);
                    for (uint32_t r = 0; r < MT; ++r)
                        for (uint32_t t = 0; t < pw; ++t) pack_tile<true>(r * pw + t, SE_LAT, r * SE_LWMAX + loff + t);
                    tile_regs_release();
                    loff += pw;
                    continue;
                }
#endif
                pack_reconfig_data_format(HOWN);
#if defined(SE_GLU) && !defined(SE_PROBE_NOGLU)
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if SE_GLU == 2
                for (uint32_t t = 0; t < MT * pw; t += 2)
                    PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_situ_glu, (DST_ACCUM_MODE, 8, sfpu::SeGluCfg), t, t + 1, t, VectorMode::RC)));
#elif SE_GLU == 3  // (silu(min(gate, 10)) x clamp(up, +-10): GLM-5.3's swiglu_limit, DeepSeek V4's)
                for (uint32_t t = 0; t < MT * pw; t += 2)
                    PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_clamped_silu_glu, (DST_ACCUM_MODE, 8, sfpu::SeGluCfg), t, t + 1, t, VectorMode::RC)));
#elif SE_GLU == 4  // (SwiGLU-OAI: (clamp(up, +-limit) + 1) x g sigmoid(alpha g), g = min(gate, limit) - MiniMax M3)
                for (uint32_t t = 0; t < MT * pw; t += 2)
                    PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_swiglu, (DST_ACCUM_MODE, 8, sfpu::SeGluCfg, true), t, t + 1, t, VectorMode::RC)));
#else
                for (uint32_t t = 0; t < MT * pw; t += 2) silu_tile_pack(t);
                for (uint32_t t = 0; t < MT * pw; t += 2)
                    PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul, (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE), t, t + 1, t, VectorMode::RC)));
#endif
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
                for (uint32_t r = 0; r < MT; ++r)
                    for (uint32_t t = 0; t < pw; t += 2) pack_tile<true>(r * pw + t, HOWN, r * POMAX + hoff + t / 2);
                tile_regs_release();
                hoff += pw / 2;
            }
        }
#endif
        cb_push_back(HOWN, MT * POMAX);
#if SE_DT == 4
        if (G * SPS > 1 && apushed < SE_ACCP) {  // (the region's tail: padding, so the next chunk starts at its start)
#else
        if (G > 1 && apushed < SE_ACCP) {  // (the region's tail: padding, so the next chunk starts at its start)
#endif
            cb_push_back(ACC, SE_ACCP - apushed);
            cb_wait_front(ACC, SE_ACCP - apushed);
            cb_pop_front(ACC, SE_ACCP - apushed);
        }
#ifdef SE_PROBE_GUWAIT
        UNPACK((DeviceTimestampedData("SE-GUWX", gu_wx)));
        UNPACK((DeviceTimestampedData("SE-GUWW", gu_ww)));
#endif
#ifdef SE_LAT
        cb_reserve_back(SE_LAT, SE_LATP);  // (the latent: the BRISC hands it to the ports, the NCRISC writes mine;
        cb_push_back(SE_LAT, SE_LATP);     // with chunks a chunk's LAT + LATZ, the next chunk's packed past them)
#endif
        }
#ifdef SE_RS
        {
        DeviceZoneScopedN("SE-DOWNRS");
        // (down + reduce-scatter) the host's program: pass i - all of K in DEST over h-all and the streamed W_down;
        // row r's tiles packed at base_r: into HOLD (fp32: an entry that adds an arrival here) or FINB (bf16: a start
        // entry, exchanged as is); hand e - entry e's start sub-block is complete (a FINC page: the BRISC's hand-off);
        // fin e - HOLD + the arrival(s) in fp32 DEST, packed bf16 into FINB (relay: handed off) or OUTF (fin: y).
        // Partials accumulate fp32 in L1; only the exchanged sub-blocks are bf16 (matmul_reduce_scatter's rule).
        const uint32_t r0 = NPG + NPD + GX + 1, nops = se_arg(r0), pt0 = r0 + 1 + nops;
        const uint32_t et0 = pt0 + NPD * (1 + MT);
        uint32_t npass = 0, hin = 0;  // (h-all columns waited for: the first pass follows the exchange's rounds)
        if (NCH > 1 && m > 0) {  // (the ports read the last chunk's FINB: mine to write again - the NCRISC's credit)
            cb_wait_front(SE_FFREE, 1);
            cb_pop_front(SE_FFREE, 1);
        }
        for (uint32_t o = 0; o < nops; ++o) {
            const uint32_t op = se_arg(r0 + 1 + o), kind = op & 0xFF, idx = op >> 8;
            if (kind == 0) {
#ifdef SE_PROBE_RSSTAT
                DeviceZoneScopedN("SE-DPASS");
#endif
                const uint32_t pd = idx, pw = se_arg(pt0 + pd * (1 + MT));
                reconfig_data_format(WIN, HALL);
                matmul_block_init(HALL, WIN, 0, pw, MT, POMAX);
                uint32_t c = 0;
                tile_regs_acquire();
                for (uint32_t i = 0; i < NSD; ++i) {
                    cb_wait_front(WIN, WSLOT);
                    for (uint32_t k = 0; k < KBD; ++k) {
                        const uint32_t kt = i * KBD + k;
                        while (kt >= se_arg(NPG + NPD + c + 1)) ++c;
                        if (c >= hin) {
                            hin = c + 1;
                            cb_wait_front(HALL, hin * MT * POMAX);
                        }
                        const uint32_t a0 = c * MT * POMAX + kt - se_arg(NPG + NPD + c);
#ifndef SE_PROBE_NOMATH
                        matmul_block(HALL, WIN, a0, k * pw, 0, false, pw, MT, POMAX);
#endif
                    }
                    cb_pop_front(WIN, WSLOT);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < MT; ++r) {
                    const uint32_t bw = se_arg(pt0 + pd * (1 + MT) + 1 + r);
                    const uint32_t ocb = (bw >> 31) ? SE_HOLD : SE_FINB, base = bw & 0x7FFFFFFF;
                    pack_reconfig_data_format(ocb);
                    for (uint32_t t = 0; t < pw; ++t) pack_tile<true>(r * pw + t, ocb, base + t);
                }
                tile_regs_release();
                cb_reserve_back(SE_PSYNC, 1);  // (pass done: the UNPACK of a later finish may read its HOLD tiles)
                cb_push_back(SE_PSYNC, 1);
                ++npass;
            } else if (kind == 1) {
                cb_reserve_back(SE_FINC, 1);
                cb_push_back(SE_FINC, 1);
#ifdef SE_PREP
            } else if (kind == 3) {
                // dispatch prep: my rows of the chip's owned tokens (x own / the latent) untilized into PO, a unit (n
                // tiles of one row tile, contiguous in L1 at PU + off) at a time; the NCRISC writes PO's rows into
                // the dispatch buffer. After the last pass: under the reduce-scatter's waits for arrivals.
                DeviceZoneScopedN("SE-PREP");
#ifdef SE_PREP_LAST
                // (with chunks: every chunk's units in the last chunk's op - no prep writes under the W re-reads)
                const uint32_t nu = m + 1 == NCH ? se_arg(idx) : 0, u0 = 0, u1 = NCH * nu;
#else
                const uint32_t nu = se_arg(idx), u0 = m * nu, u1 = (m + 1) * nu;
#endif
                uint32_t pu0 = 0;
                UNPACK((pu0 = get_local_cb_interface(SE_PU).fifo_rd_ptr));
                uint32_t lrdy = 0;  // (chunks' latent arrivals waited for)
                if (nu) {
                    reconfig_data_format_srca(SE_PU);  // (the last pass left srcA at the weights' format)
                    pack_reconfig_data_format(SE_PO);
                    pack_untilize_init<SE_PREP_BCT, SE_PREP_FULL>(SE_PU, SE_PO);
                }
                for (uint32_t u = u0; u < u1; ++u) {  // (chunk u / nu's units)
                    const uint32_t off = se_arg(idx + 1 + 2 * u), fl = se_arg(idx + 2 + 2 * u);
                    const uint32_t n = fl & 0xFFFF;
#ifdef SE_LRDY
                    if ((fl >> 16) && lrdy < u / nu + 1) {  // (a unit of the latent's arrivals: once they all landed)
                        lrdy = u / nu + 1;
                        cb_wait_front(SE_LRDY, lrdy);
                    }
#endif
                    cb_reserve_back(SE_PO, SE_PREP_FULL);
                    for (uint32_t c = 0; c * SE_PREP_BCT < n; ++c) {
                        UNPACK((get_local_cb_interface(SE_PU).fifo_rd_ptr = pu0 + (off + c * SE_PREP_BCT * 2048) / 16));
                        pack_untilize_block<SE_PREP_BCT, SE_PREP_FULL>(SE_PU, 1, SE_PO, c);
                    }
                    cb_push_back(SE_PO, SE_PREP_FULL);
                }
                if (nu) pack_untilize_uninit(SE_PO);
                UNPACK((get_local_cb_interface(SE_PU).fifo_rd_ptr = pu0));
                (void)lrdy;
#ifdef SE_XFREE
                cb_reserve_back(SE_XFREE, 1);  // (x own's ring: this chunk's slot read - the BRISC may reload it)
                cb_push_back(SE_XFREE, 1);
#endif
#endif
            } else {
                const uint32_t e6 = et0 + 6 * idx, cnt = se_arg(e6), mode = se_arg(e6 + 1);
                const uint32_t sf = se_arg(e6 + 2), sb = se_arg(e6 + 3);
                const uint32_t ho = se_arg(e6 + 4), fbo = se_arg(e6 + 5);
                cb_wait_front(SE_PSYNC, npass);
                // (sf / sb: the arrivals' zone offsets, tiles - every slot before them pushed with them)
                if (mode & 1) cb_wait_front(SE_ARRF, sf + cnt);
                if (mode & 2) cb_wait_front(SE_ARRB, sb + cnt);
                const uint32_t a1 = (mode & 1) ? SE_ARRF : SE_ARRB, s1 = (mode & 1) ? sf : sb;
                const uint32_t ocb = (mode & 4) ? SE_OUTF : SE_FINB;
                pack_reconfig_data_format(ocb);
                for (uint32_t i0 = 0, n = 0; i0 < cnt; i0 += n) {
                    const uint32_t lim = (mode & 4) ? SE_DT - opos % SE_DT : SE_DT;
                    n = cnt - i0 < lim ? cnt - i0 : lim;
                    if (mode & 4) opos += n;
                    tile_regs_acquire();
                    reconfig_data_format(SE_HOLD, a1);
                    add_init(SE_HOLD, a1);
                    for (uint32_t j = 0; j < n; ++j) add_tiles(SE_HOLD, a1, ho + i0 + j, s1 + i0 + j, j);
                    if ((mode & 3) == 3) {  // (the second arrival: DEST -> srcA at fp32 width, + the bwd tile)
                        reconfig_data_format(SE_HOLD, SE_ARRB);
                        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(SE_ARRB);
                        for (uint32_t j = 0; j < n; ++j)
                            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(SE_ARRB, sb + i0 + j, j);
                    }
                    tile_regs_commit();
                    if (mode & 4) cb_reserve_back(SE_OUTF, n);
                    tile_regs_wait();
                    for (uint32_t j = 0; j < n; ++j) {
                        if (mode & 4) pack_tile(j, SE_OUTF);
                        else pack_tile<true>(j, SE_FINB, fbo + i0 + j);
                    }
                    tile_regs_release();
                    if (mode & 4) cb_push_back(SE_OUTF, n);
                }
                if (!(mode & 4)) {
                    cb_reserve_back(SE_FINC, 1);
                    cb_push_back(SE_FINC, 1);
                }
            }
        }
        cb_pop_front(HALL, HA);
        if constexpr (NCH > 1) {  // (the chunk's syncs and arrival zones popped: the next chunk's from their start)
            cb_wait_front(SE_PSYNC, npass);
            cb_pop_front(SE_PSYNC, npass);
            uint32_t mf = 0;
            for (uint32_t e = 0; e < SE_NENT; ++e) mf |= se_arg(et0 + 6 * e + 1);  // (the RS's entries: TP of them)
            if (mf & 1) {
                cb_wait_front(SE_ARRF, SE_ZF);
                cb_pop_front(SE_ARRF, SE_ZF);
            }
            if (mf & 2) {
                cb_wait_front(SE_ARRB, SE_ZB);
                cb_pop_front(SE_ARRB, SE_ZB);
            }
        }
        }
#elif !defined(SE_NODOWN)
        DeviceZoneScopedN("SE-DOWN");
        // down: per pass all of K in DEST over h-all and the streamed W_down blocks, y packed once
#ifndef SE_HXINC
        cb_wait_front(HALL, HA);
#endif
        uint32_t hin = 0;  // (SE_HXINC: h-all columns waited for - the first pass follows the exchange's rounds)
        for (uint32_t pd = 0; pd < NPD; ++pd) {
            const uint32_t pw = se_arg(NPG + pd);
            matmul_block_init(HALL, WIN, 0, pw, MT, POMAX);  // (h rows strided POMAX within a column's block)
            uint32_t c = 0;  // (the column block of K tile k: k in [h0[c], h0[c + 1]))
            tile_regs_acquire();
            for (uint32_t i = 0; i < NSD; ++i) {
                cb_wait_front(WIN, WSLOT);
                for (uint32_t k = 0; k < KBD; ++k) {
                    const uint32_t kt = i * KBD + k;
                    while (kt >= se_arg(NPG + NPD + c + 1)) ++c;
                    const uint32_t a0 = c * MT * POMAX + kt - se_arg(NPG + NPD + c);
#ifdef SE_HXINC
                    if (c >= hin) {
                        hin = c + 1;
                        cb_wait_front(HALL, hin * MT * POMAX);
                    }
#endif
#ifndef SE_PROBE_NOMATH
                    matmul_block(HALL, WIN, a0, k * pw, 0, false, pw, MT, POMAX);
#endif
                }
                cb_pop_front(WIN, WSLOT);
            }
            tile_regs_commit();
            cb_reserve_back(OUT, MT * PWDMAX);
            tile_regs_wait();
            for (uint32_t r = 0; r < MT; ++r)
                for (uint32_t t = 0; t < pw; ++t) pack_tile<true>(r * pw + t, OUT, r * PWDMAX + t);
            tile_regs_release();
            cb_push_back(OUT, MT * PWDMAX);
        }
        cb_pop_front(HALL, HA);
#endif
    }
}
"""


def _probes():
    """(timing only) SE_PROBE=NOX,NOW: skip the x / W data multicasts (semaphores kept); NOGLU: no gate / up activation (h = the
    gate tiles); NOXR / NOWR: no x / W rounds at
    all (no multicast, no handshake / credits: the CB pushed as if the round landed - the bottom-up build's stages); NOMATH: no matmuls;
    NOACCPACK: gate / up's shard partials not packed (the full-sync pass boundary's pack cost); NOWL: no W
    DRAM reads (16 B a block); NOWREAD: no W DRAM reads at all; NOXWAIT: the x rounds do not wait for a shard's arrival (gate / up at its own pace in the
    head: GU_ALONE at the head's shapes) - the results are wrong. GUWAIT (results right): gate / up's unpacker cycles
    waiting for x / W a core, SE-GUWX / SE-GUWW; GUSTEP (with GUWAIT): the first 48 steps' waits, one data marker each
    (unnamed in the csv: their source lines sit inside a macro argument). WLOOP: the NCRISC stream loop's passes and
    cycles while the first chunk's gate / up W rounds land (SE-WLIT / SE-WLCY, a core)."""
    import os

    out = "".join(f"#define SE_PROBE_{p}\n" for p in os.environ.get("SE_PROBE", "").split(",") if p)
    out += "\n" * int(os.environ.get("SE_PAD", "0"))  # (profiling: shift the source lines - a zone hash collision)
    if os.environ.get(
        "SE_WPF"
    ):  # (timing probe: the W prefetch window, blocks in flight a core; <= 13 transaction ids)
        out += f"#define SE_WPF {int(os.environ['SE_WPF'])}u\n"
    # (the router on the engine: its W_r^T leads spread over the rows - see SE-RTRLD; K3 head traced 397 -> 389 us,
    # the logits 46 -> 37. SE_RWL=0: all on row 0)
    if os.environ.get("SE_RWL", "1") == "1":
        out += "#define SE_RWL\n"
    wla = int(os.environ.get("SE_WLA", "6"))  # (the W prefetch window in the column's rounds - see prefetch; 0: off,
    # 3 of my blocks in flight a core: K3 head traced 440 -> 397 us at 6, flat over 5-8)
    if wla:
        out += f"#define SE_WLA {wla}u\n"
    # (HW: hang words - a kernel's state written into a persistent L1 tensor (fused_head sets HW_ADDR) as it waits;
    # tools: moe_block_runs' triage add-on reads them after a hang)
    if HW_ADDR and "HW" in os.environ.get("SE_PROBE", "").split(",") and not os.environ.get("HW_ALLOC_ONLY"):
        out += f"#define SE_HW {HW_ADDR}u\n#define HWW(i, v) (reinterpret_cast<volatile uint32_t*>(SE_HW)[i] = (v))\n"
    else:
        out += "#define HWW(i, v)\n"
    return out


HW_ADDR = 0  # (the hang words' L1 address - SE_PROBE=HW, set by fused_head)


GLU_CODES = {"silu": 1, "situ": 2, "clamped_silu": 3, "swiglu_oai": 4}


def _f(v):
    return f"{float(v)!r}f"


def glu_defines(act):
    """(SE_GLU code, its constants' defines) of the shared expert's activation: a kind name (the models' defaults - SiTU
    beta 4 / 25, clamped SiLU limit 10, SwiGLU-OAI limit 7 / alpha 1.702) or an ExpertActivation (its own)."""
    kind = act if isinstance(act, str) else act.kind
    if kind not in GLU_CODES:
        raise ValueError(f"se_engine3: shared activation {kind}: one of {', '.join(GLU_CODES)}")
    g = lambda k, d: d if isinstance(act, str) or getattr(act, k, None) is None else getattr(act, k)  # noqa: E731
    lim = g("limit", 7.0 if kind == "swiglu_oai" else 10.0)
    vals = dict(
        SE_GLU_BETA=g("beta", 4.0),
        SE_GLU_LBETA=g("linear_beta", 25.0),
        SE_GLU_LIMIT=lim,
        SE_GLU_ALPHA=g("alpha", 1.702),
    )
    return GLU_CODES[kind], "".join(f"#define {k} {_f(v)}\n" for k, v in vals.items())


class ArenaTooLarge(ValueError):
    """The matmul cores' arena does not fit the L1 cap at any W ring / h slot count: .need (bytes, the last try's), .cap."""


def _ring_pages(pages, pushes):
    """A FIFO CB's region (pages): pack_tile / copy_tile address a block from the CB's pointer without wrapping (the
    push / pop wrap after), so where a block may straddle the CB's end - blocks of differing sizes, or one size not
    dividing the CB - the region needs the largest block - 1 pages of room past the CB (else the tail overwrites the
    next region). Room is a fix only where both ends address blocks so (compute <-> compute: ACC); a dataflow end
    reading page by page wraps at the end - such a CB's blocks must not straddle (SE_OUTF: one fin a launch, 8-tile
    blocks from the reset pointer, the tail last)."""
    ps = set(int(p) for p in pushes)
    return pages if len(ps) == 1 and pages % min(ps) == 0 else pages + max(ps) - 1


def _split(n, parts, unit):
    """n tiles into `parts` widths, multiples of unit, as even as possible (each >= unit)."""
    assert n % unit == 0 and n // unit >= parts, (n, parts, unit)
    base, rem = divmod(n // unit, parts)
    return [(base + (1 if i < rem else 0)) * unit for i in range(parts)]


def _split_pad(n, parts, unit):
    """_split, or (fewer than `parts` units) a unit a part with the tail parts repeating the first unit (dummy passes:
    a column with fewer gate / up pairs than the passes every row's x stream walks - their h lands past the column's
    pairs, unread)."""
    return _split(n, parts, unit) if n // unit >= parts else [unit] * parts


def _subsets(nf, GX):
    """Two owner-subset sizes (families 0 / 1) dividing nf, together covering as many of GX columns as possible."""
    divs = [d for d in range(1, GX + 1) if nf % d == 0]
    best = max(((a, b) for a in divs for b in divs), key=lambda ab: (min(ab[0] + ab[1], GX), -abs(ab[0] - ab[1])))
    return best


def _common_span(lists, start, n):
    """lists {core (x, y): its full arg list} -> (ttnn.RuntimeArgs without args [start, start + n), common args [start,
    n, those n values]) when every core holds the same span; else (the full lists, [0, 0]) - se_ra reads either."""
    spans = {tuple(a[start : start + n]) for a in lists.values()}
    shared = n > 0 and len(spans) == 1 and all(len(a) >= start + n for a in lists.values())
    shared = shared and os.environ.get("SE_COMMON_SPAN", "1") != "0"  # (A/B: 0 sends every arg per core)
    rt = ttnn.RuntimeArgs()
    for (x, y), a in lists.items():
        rt[x][y] = a[:start] + a[start + n :] if shared else a
    return rt, ([start, n] + list(next(iter(spans))) if shared else [0, 0])


def next_semaphore_id(prog):
    """The first semaphore id past the program's (a multicast family takes its ids from there)."""
    return max((int(sd.id) for sd in prog.semaphores), default=-1) + 1


class SeEngine3:
    """The shared expert a chip, row-chunk-major, weights resident: x [rows, H] (the gathered x) -> y [rows, H] (this
    chip's partial over its I slice). glu "silu" / "situ" / "clamped_silu" (limit 10)."""

    def __init__(
        self,
        mesh_device,
        *,
        rows,
        H,
        I,
        glu,
        cx0=0,
        GX=11,
        GY=10,
        MT=1,
        KB=None,
        KBD=None,
        fidelity=ttnn.MathFidelity.HiFi2,
        w_dtype=ttnn.bfloat8_b,
        shards=1,
        down=True,
        rs=None,
        latent=None,
        prep=False,
        router=None,
        arena_max=None,
        xring=0,
    ):
        """arena_max: host planning (mesh_device None) - size the arena against this cap and stop (no device state).
        prep: the dispatch prep's owned rows (pieces(prep=...)): x own's rows of the chip's owned tokens (or, with a
        latent, their latent) untilized by the compute after its last pass, written row-major by the NCRISC.
        latent (tiles): K3's latent down on the gathered x - the chip's column slice [H, latent tiles], extra plain
        passes after gate / up (fp32 partials between shards, bf16 out) - and its all-to-all to the owners (with rs).
        rs (se_rs.RsPlan): down + the SE reduce-scatter in this program (tiling, schedule; the ports are the
        gather's movers) - y is then this chip's reduced block. down False: no down here - h [rows, I] goes to DRAM (y) for the down + reduce-scatter stage.
        shards: the K shards gate / up walks shard-major (the TP size in the fused head: the gather's shards)."""
        self.mesh = mesh_device
        self.shards = shards
        self.rows, self.H, self.I, self.glu = rows, H, I, glu
        self.cx0, self.GX, self.GY, self.MT = cx0, GX, GY, MT
        self.CR = GY * MT
        assert (
            rows % (32 * self.CR) == 0
        ), f"se_engine3: {rows} rows are not whole chunks of CR = GY {GY} x MT {MT} = {self.CR} row tiles"
        self.NCH = rows // 32 // self.CR
        self.KT, self.HT, self.NTH = H // 32, I // 32, H // 32
        self.KB = KB or 16  # (fewer, larger rounds: K3 403 -> 341 us at 640 rows; GLM-Flash neutral)
        self.NSG = self.KT // self.KB
        assert (
            self.KT % self.KB == 0 and self.NSG % 2 == 0
        ), f"se_engine3: K step {self.KB} must divide H's {self.KT} tiles into an even count of steps"
        self.arena_max = arena_max if arena_max is not None else arena_cap(mesh_device, self.KB)
        assert self.NSG % shards == 0 and (self.NSG // shards) % 2 == 0, "even K steps a shard"
        self.SPS = self.NSG // shards
        nf = self.NSG // 2
        # (a family's owners must divide its rounds a (shard, pass) block - SPS / 2 - so a position keeps one owner)
        self.S0, self.S1 = _subsets(self.SPS // 2, GX)
        cols = list(range(GX))
        self.fam0 = cols[: self.S0]
        self.fam1 = cols[GX - self.S1 :]  # (overlaps family 0 only when S0 + S1 > GX)
        # gate / up: the column's tile pairs, NPG passes of even widths <= 8 / MT
        pairs = column_split(self.HT, GX, 1)
        assert min(pairs) >= 1, f"se_engine3: I's {self.HT} tiles leave a column of the {GX} without a gate / up pair"
        self.po, self.h0 = pairs, [sum(pairs[:c]) for c in range(GX)]
        self.POMAX = max(pairs)
        # (DEST tiles a pass: 8 fp32 with full sync; SE_HALFSYNC=1 (A/B, engine alone): 4 a half - the GLU and pack of
        # one pass overlap the next pass's matmuls)
        self.DT = dest_tiles()
        # (gate / up / latent passes: up to 8 tiles; with half sync they go through DEST as row sub-blocks of <= 4 -
        # a row at most 4 wide)
        self.PGW = min(8 // MT, 4) if self.DT == 4 else 8 // MT
        pwmax = self.PGW // 2 * 2
        self.NPG = max(-(-2 * p // pwmax) for p in pairs)
        self.wg = [_split_pad(2 * p, self.NPG, 2) for p in pairs]
        if os.environ.get("SE_PASSREV") == "1":  # (A/B: the narrow gate / up pass first - under the x load, which binds
            # the first pass anyway - and the full-width passes after it, with only the W stream left)
            self.wg = [sorted(wg) for wg in self.wg]
        self.NGU, self.LT = self.NPG, latent  # (gate / up passes; then the latent's)
        if latent:
            lw = column_split(latent, GX, 1)
            assert min(lw) >= 1, f"se_engine3: the latent's {latent} tiles leave a column of the {GX} without one"
            self.lw, self.l0 = lw, [sum(lw[:c]) for c in range(GX)]
            self.LWMAX = max(lw)
            npl = -(-self.LWMAX // self.PGW)
            self.wg = [wg + _split(lw[c], npl, 1) for c, wg in enumerate(self.wg)]
            self.NPG += npl
        # (the latent's pages a chunk: its LAT block - with chunks its LAT + LATZ, chunk m's at m LATP: no zone reused
        # in a call)
        self.LATP = (shards if self.NCH > 1 else 1) * MT * self.LWMAX if latent else 0
        # down: the column's output tiles, NPD passes of widths <= 8 / MT, K steps of KBD
        pnd = column_split(self.NTH, GX, 1)
        self.n0d = [sum(pnd[:c]) for c in range(GX)]
        self.PWDMAX = self.DT // MT
        self.down = down
        self.NPD = max(-(-p // self.PWDMAX) for p in pnd) if down else 0
        self.wd = [_split(p, self.NPD, 1) if down else [] for p in pnd]
        self.rs = rs
        if rs is not None:  # (the column's down passes: the plan's, widths the same on every chip)
            assert down and rs.pwmax == self.PWDMAX and rs.MT == MT and rs.GX == GX and rs.GY == GY, (
                f"se_engine3: the RS plan's tiling (GX {rs.GX} GY {rs.GY} MT {rs.MT} pwmax {rs.pwmax}) is not the "
                f"matmul cores' (GX {GX} GY {GY} MT {MT} pwmax {self.PWDMAX})"
            )
            self.wd = [[pw for _, pw, _ in rs.column_passes(0, c)] for c in range(GX)]
            self.NPD = len(self.wd[0])
            assert all(len(wd) == self.NPD for wd in self.wd), f"se_engine3: uneven down passes {self.wd}"
        # (the down's K step: 8, or the largest divisor of I's tiles under it - DeepSeek-V4.1's 36 a chip: 6)
        KBD = KBD or max(d for d in range(1, 9) if self.HT % d == 0)
        KBD = int(os.environ.get("SE_KBD", KBD))  # (SE_KBD: A/B the down K step)
        self.KBD = KBD
        assert self.HT % KBD == 0, f"se_engine3: the down's K step {KBD} does not divide I's {self.HT} tiles"
        self.NSD = self.HT // KBD if down else 0
        self.NB = self.NPG * self.NSG + self.NPD * self.NSD
        # (the NCRISC's per-block tables - the stream order, a core's offsets / transaction ids / ring slots - are sized
        # by NB at compile time: at most 1024 rounds a chunk, the kernel's static_assert)
        if self.NB > 1024:
            raise ValueError(
                f"se_engine3: {self.NB} W rounds a chunk > 1024 (NPG {self.NPG} x NSG {self.NSG} + NPD {self.NPD} x NSD "
                f"{self.NSD}): a larger K step than {self.KB}"
            )
        self.WSLOT = max(self.KB * max(sum(self.wg, [])), KBD * max(sum(self.wd, []), default=0))  # (tiles)
        self.fid, self.w_dtype = fidelity, w_dtype
        wtbs = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}
        if w_dtype not in wtbs:
            raise ValueError(f"se_engine3: shared expert weights {w_dtype}: bf4 / bf8 / bf16 only")
        self.wtb = wtbs[w_dtype]

        # per-core W-own (tiles): the blocks of its column on its row
        def bsize(c, b):
            return (
                self.KB * self.wg[c][b // self.NSG]
                if b < self.NPG * self.NSG
                else KBD * self.wd[c][(b - self.NPG * self.NSG) // self.NSD]
            )

        self.bsize = bsize
        self.wown_t = max(
            sum(bsize(c, b) for b in range(self.NB) if b % GY == cy) for c in range(GX) for cy in range(GY)
        )
        # (W-own in L1: every block of mine - or, the arena too large otherwise, a ring of WRING slots (the NCRISC's
        # prefetch window + 2; with chunks every chunk reads its blocks again); the DRAM region always holds every
        # block, wown_t. h-all: HSL slots - two with chunks (the next chunk's h exchange under this one's down), else one)
        self.WRING = 0
        self.HSL = 1
        nown = max((nf // self.S0 if c in self.fam0 else 0) + (nf // self.S1 if c in self.fam1 else 0) for c in cols)
        # (x own: every chunk's steps resident - or xring (the head's DRAM fallback: x from a gathered DRAM copy) a ring of
        # XR chunk slots, each reloaded from DRAM once its chunk's rounds (and prep) are done)
        self.XR = min(xring, self.NCH) if xring else self.NCH
        self.xring = self.XR < self.NCH
        self.xfree = self.xring and prep and not latent  # (prep reads x own: the slot's reload waits for it)
        self.xown_t = self.XR * nown * MT * self.KB
        # dispatch prep: a unit = one row tile's run of x own (a K step: KB tiles) or of a latent block (LWMAX)
        self.prep = prep
        self.PFULL = self.LWMAX if latent else self.KB
        self.PBCT = max(d for d in range(1, self.DT + 1) if self.PFULL % d == 0)
        tries = [(0, 1), (5, 1), (4, 1), (3, 1)] if self.NCH == 1 else [(0, 2), (5, 2), (5, 1), (4, 1), (3, 1)]
        rsr0 = self._rs_regions() if rs is not None else []  # (the same at every W ring / h slot count)
        for self.WRING, self.HSL in tries:  # (see wown_t)
            al = lambda v: (v + 63) // 64 * 64  # noqa: E731
            self.acc_pages = MT * max(sum(c) for c in self.wg)  # (a fixed slot a pass: the busiest column's)
            off, o = {}, 0
            rsr = list(rsr0)
            # (the reduce-scatter's local buffers - HOLD, FINB: this core's compute and, after its hand-off, its port read
            # them - overlay XIN + ACC, which are dead once this core's gate / up is done: no row peer multicasts into XIN
            # after the last round it consumed; only its own compute touches ACC)
            # (one chunk only: with more, the next chunk's x rounds land in XIN while the ports still read FINB)
            ovl = dict(rsr[:2]) if rsr and self.NCH == 1 else {}
            rsr = rsr[2:] if ovl else rsr
            # (chunks: HOLD alone over ACC - both the compute's only, and its chunk m RS runs between chunk m's and
            # chunk m + 1's gate / up)
            ovh = dict(rsr[:1]) if rsr and self.NCH > 1 else {}
            rsr = rsr[1:] if ovh else rsr
            for k, v in (
                [
                    ("xown", self.xown_t * 2048),
                    ("wown", (self.WRING * self.WSLOT if self.WRING else self.wown_t) * self.wtb),
                    ("win", NWS * self.WSLOT * self.wtb),
                    ("hown", MT * self.POMAX * 2048),
                    ("hall", self.HSL * GX * MT * self.POMAX * 2048 if down else 0),
                    ("xin", 2 * MT * self.KB * 2048),
                    # (ACC: the passes' fp32 partials, a fixed slot each - the packer accumulates onto them in L1;
                    # every chunk starts at the region's start, so no block straddles its end)
                    ("acc", self.acc_pages * 4096),
                    ("out", NOUT * MT * self.PWDMAX * 2048 if down and rs is None else 0),
                    ("words", 256),
                    ("xfree", 16 * self.NCH if self.xfree else 0),
                ]
                + rsr
                + (
                    [
                        ("lat", MT * self.LWMAX * 2048),
                        ("latz", (shards - 1) * MT * self.LWMAX * 2048 + (self.NCH - 1) * self.LATP * 2048),
                    ]
                    if latent
                    else []
                )
                + ([("po", 2 * self.PFULL * 2048)] + ([("lrdy", self._lrdy_bytes())] if latent else []) if prep else [])
            ):
                off[k], o = o, al(o + v)
            self.seq_ends = sorted(set(off.values()) | {o})  # (the sequential regions' bounds: the CB plan check's)
            if ovl:
                room = off["out"] - off["xin"]
                if al(ovl["hold"]) + ovl["finb"] <= room:
                    off["hold"], off["finb"] = off["xin"], off["xin"] + al(ovl["hold"])
                else:  # (no room: their own regions)
                    for k, v in ovl.items():
                        off[k], o = o, al(o + v)
            if ovh:
                if ovh["hold"] <= self.acc_pages * 4096:
                    off["hold"] = off["acc"]
                else:
                    off["hold"], o = o, al(o + ovh["hold"])
            # the router logits on the engine (router: dict(ET, KTO)): x rows, W_r^T expert tiles, the fp32 out - in the
            # arena's tail past the words (the reduce-scatter's zones, the latent's, the prep's: all written only after
            # gate / up, the router's phase is over before it), else their own region
            self.rtr = router
            if router is not None:
                self.ETC = -(-router["ET"] // GX)
                MTR = MT * self.NCH
                # (DEST: ETC x a row group's tiles <= 8 - every chunk's rows at once, else a chunk's MT at a time)
                self.RG = MTR if self.ETC * MTR <= self.DT else (MT if self.ETC * MT <= self.DT else 1)
                assert self.ETC * self.RG <= self.DT, "the router logits on the engine: ETC x its row group in DEST"
                KTO = router["KTO"]
                # (K chunk: x / W_r^T rounds, two buffers - two rounds (KTO / 2) where the region fits past the
                # words, else more, smaller rounds behind the receivers' acks: the largest that fits. The region may
                # grow the arena by what compact RS zones gave back - its K chunk as with P-tile slots: not grown,
                # K2.7's KC went 28 -> 14 at 128 rows, +8% on its chain-bound head)
                zgrow = (
                    sum((rs.slots[d] * rs.P - rs.zone_tiles(d)) * 2048 for d in ("fwd", "bwd"))
                    if rs is not None and rs.compact
                    else 0
                )
                t0 = off[rsr[0][0]] if rsr else o
                kcs = sorted((d for d in range(1, KTO + 1) if KTO % d == 0 and KTO // d >= 2), reverse=True)
                for self.KC in kcs:
                    rsz = [
                        ("rx", 2 * self.KC * MTR * 2048),
                        ("rw", 2 * self.ETC * self.KC * 2048),
                        ("ro", self.ETC * MTR * 4096),
                    ]
                    need = sum(al(v) for _, v in rsz)
                    if (
                        t0 + need <= min(o + zgrow, max(o, self.arena_max))
                        or self.KC == kcs[-1]
                        or (self.KC == kcs[0] and KTO // self.KC == 2 and o + need <= self.arena_max)
                    ):
                        break
                if o - t0 < need and t0 + need <= min(o + zgrow, max(o, self.arena_max)):
                    o = t0 + need
                elif o - t0 < need:
                    hall = off["xin"] - off["hall"]
                    t0 = off["hall"] if down and hall >= need else o
                    if t0 == o:
                        o = o + need
                for k, v in rsz:
                    off[k], t0 = t0, al(t0 + v)
            self.off, self.arena_bytes = off, o
            if o <= self.arena_max:
                break
        # (each region's bytes, in arena order - overlays share their host's)
        ends = sorted(set(off.values()) | {o})
        self.sizes = {k: next(e for e in ends if e > v) - v if v < o else 0 for k, v in off.items()}
        # (the router's buffers over a region another chip writes - the reduce-scatter's arrival zones, the latent's:
        # those writes wait for this chip's "router done" (SE_RDONE: every matmul core's router phase over, counted on
        # the ports; they credit the zones / tell "started" only then) - the chips do not run in step: a chip that
        # does not wait for this one's x (the DRAM fallback), or starts a call earlier, reaches its reduce-scatter
        # while this one still reads x rows / W_r^T there; on the first call of the fallback the logits came out wrong)
        # (rtr_zones: over ARRF / ARRB - the reduce-scatter's chunk 0 waits; rtr_latz: over LATZ - the latent's
        # "started" waits; else as before)
        self.rtr_zones = self.rtr_latz = False
        if router is not None:
            lo, hi = off["rx"], off["ro"] + self.ETC * MT * self.NCH * 4096
            over = lambda k, v: v > 0 and off[k] < hi and lo < off[k] + v  # noqa: E731
            if rs is not None:
                self.rtr_zones = any(over(k, self.rs_sizes[k]) for k in ("arrf", "arrb"))
            if latent:
                self.rtr_latz = over("latz", (shards - 1) * MT * self.LWMAX * 2048 + (self.NCH - 1) * self.LATP * 2048)
        self.rtr_overlay = self.rtr_zones or self.rtr_latz
        if os.environ.get("SE_LOG") == "1":  # (the resolved layout: KB, W ring, router K chunk, regions)
            from loguru import logger

            logger.info(
                f"se_engine3: KB {self.KB} KBD {self.KBD} W ring {self.WRING} HSL {self.HSL} NPG {self.NPG} NPD {self.NPD}"
                f" router KC {getattr(self, 'KC', None)} arena {o // 1024} / {self.arena_max // 1024} KB: "
                + " ".join(f"{k} {v // 1024}@{off[k] // 1024}" for k, v in self.sizes.items())
            )
        if o > self.arena_max:
            sizes = {
                k: (off[k2] if i + 1 < len(off) else o) - off[k]
                for i, (k, k2) in enumerate(zip(off, list(off)[1:] + [None]))
            }
            err = ArenaTooLarge(
                f"se_engine3: arena {o // 1024} KB > {self.arena_max // 1024} KB of L1 (KB {self.KB}, W ring "
                f"{self.WRING or 'off'}, h slots {self.HSL}): "
                + ", ".join(f"{k} {v // 1024}" for k, v in sizes.items())
            )
            err.need, err.cap = o, self.arena_max
            raise err
        self.groups = {}
        for c in range(GX):
            self.groups.setdefault((tuple(self.wg[c]), tuple(self.wd[c])), []).append(c)
        if mesh_device is None:  # (host planning: sized)
            return
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx0, 0), ttnn.CoreCoord(cx0 + GX - 1, GY - 1))])
        self.crs = cores
        if router is not None:  # (the router phase's words, zero; re-armed by their owners each call)
            self.ewords = ttnn.from_torch(
                torch.zeros(GX * GY, 64, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    persist_l1.buffer_type(),
                    ttnn.ShardSpec(cores, (1, 64), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        if os.environ.get("SE_BRW") == "1":  # (the W reads on the BRISC's NOC0: its words with the NCRISC, 1 KB a core)
            nw_ = 1024 if os.environ.get("SE_XN1") == "1" else 256
            # (the NCRISC's table of a core's W blocks: 4 words each from word 16, below SE_XN1's x table at word 512)
            ng = self.NPG * self.NSG if os.environ.get("SE_BRW_DN1") == "1" else self.NB
            nmax, cap = -(-ng // GY), (min(nw_, 512) - 16) // 4
            if nmax > cap:
                raise ValueError(
                    f"se_engine3: SE_BRW's table of a core's W blocks holds {cap}, a core has up to {nmax} ({ng} blocks "
                    f"over {GY} rows): a larger KB ({self.KB}) / KBD ({self.KBD})"
                )
            self.wsync = ttnn.from_torch(
                torch.zeros(GX * GY, nw_, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, (1, nw_), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        self.alloc_arena()

    def _rs_regions(self):
        rs, G = self.rs, self.rs.G
        cores = [(cx, cy) for cx in range(self.GX) for cy in range(self.GY)]
        hold = max(rs.hold_offs(p, cx, cy)[1] for p in range(G) for cx, cy in cores)
        finb = max(rs.finb_tiles(cx, cy) for cx, cy in cores)
        self.rs_sizes = dict(
            hold=max(1, hold) * 4096,
            finb=max(1, finb) * 2048,
            arrf=rs.zone_tiles("fwd") * 2048,
            arrb=rs.zone_tiles("bwd") * 2048,
            outf=8 * 2048,
            finc=16 * 8,
            psync=16 * max(8, self.NPD),
        )
        if self.NCH > 1:  # (the ports' FINB-free credits, a page a chunk: a core no port reads (fnr 0) credits every
            # chunk up front - NCH - 1 pages, else its NCRISC blocks on the reserve and stops the W stream; a hang)
            self.rs_sizes["ffree"] = 16 * max(2, self.NCH - 1)
        return list(self.rs_sizes.items())

    def _lrdy_bytes(self):
        """The latent-arrivals credits (a 16 B page a chunk): the prep (every chunk's units in the last chunk's op) waits
        for all NCH of them at once, unpopped - 4 pages held NCH 4 at most, at 5 the NCRISC blocked on the reserve and
        stopped the W stream (K3 1280 rows: a hang)."""
        return 16 * max(4, self.NCH)

    def alloc_arena(self, extra=()):
        """The arena (call scratch: x, the weights, every CB) as an L1 tensor; base its address (the programs bake it).
        extra: more cores the same range is carved on (L1 addresses are allocated in lockstep over the grid, so the
        mover column's other buffers live in it too)."""
        o = self.arena_bytes
        crs = self.crs
        self.arena_extra = tuple(extra)  # (realloc_arena: the same cores again)
        if extra:
            crs = ttnn.CoreRangeSet(list(crs.ranges()) + [ttnn.CoreRange(c, c) for c in extra])
        n = self.GX * self.GY + len(extra)
        self.arena_crs = crs
        shard = ttnn.ShardSpec(crs, [1, o // 4], ttnn.ShardOrientation.ROW_MAJOR)
        self.arena = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, n, o // 4]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
        )
        self.base = int(self.arena.buffer_address())
        self.arena_delta = 0  # (its own arena: base is the tensor's address - use_arena: a larger one's)
        return self.arena

    def use_arena(self, arena, nbytes):
        """Run on a larger L1 tensor of the caller's (nbytes a core, lockstep over the grid) whose range holds this
        engine's [base, base + arena_bytes): every baked address stays; the CBs bind it at base - its address."""
        d = self.base - int(arena.buffer_address())
        assert 0 <= d and d + self.arena_bytes <= nbytes, (
            f"se_engine3: the arena [{self.base}, +{self.arena_bytes}) is not inside the caller's "
            f"[{int(arena.buffer_address())}, +{nbytes})"
        )
        self.arena, self.arena_delta = arena, d
        return d

    def realloc_arena(self, settle=False):
        """The arena again (a caller that shares the L1 left with another op frees it between calls): it must land where
        the programs baked it. settle (once, before the caller's other op places its L1): free L1 right above the old
        range (L1 is placed top-down) is filled first - kept in self.fillers - so it lands there."""
        base = self.base
        self.alloc_arena(self.arena_extra)
        if settle and self.base > base:
            ttnn.deallocate(self.arena)
            gap = self.base - base
            grid = self.mesh.compute_with_storage_grid_size()
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
            self.__dict__.setdefault("fillers", []).append(
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([grid.x * grid.y, gap // 4]),
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                    self.mesh,
                    ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(crs, [1, gap // 4], ttnn.ShardOrientation.ROW_MAJOR),
                    ),
                )
            )
            self.alloc_arena(self.arena_extra)
        assert self.base == base, f"se_engine3: the arena landed at {self.base}, not {base}: L1 allocated since"
        return self.arena

    def dram_bytes(self):
        w = (self.KT * 2 * self.HT + self.HT * self.NTH) * self.wtb
        return dict(w=w, x=self.rows * self.H * 2, y=self.rows * self.H * 2)

    def upload(self, w_gu, w_down, pos=None, w_lat=None):
        """Per device (row-major mesh order): w_gu [H, 2 I] (glu_weight tile pairs), w_down [I, H] -> one bf8 tensor
        a chip, per-bank: core (c, cy)'s W-own (its column's blocks b = cy, cy + GY, .. of a chunk's stream order:
        gate / up [KB x pass width] pass-major, then down [KBD x pass width]) one contiguous region, region c GY + cy in
        bank (c GY + cy) % banks (flat_expert._bank_sharded) - read in large bursts, every bank equally loaded."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import _bank_sharded

        KB, KBD, NSG, GX, GY = self.KB, self.KBD, self.NSG, self.GX, self.GY
        devs = []
        for dev, (wgt, wdt) in enumerate(zip(w_gu, w_down)):
            regions = []
            for c in range(GX):
                blocks = []
                g0 = 2 * self.h0[c]
                for pg, pw in enumerate(self.wg[c]):
                    c0 = g0 + sum(self.wg[c][:pg])
                    if pg < self.NGU and c0 + pw > 2 * (
                        self.h0[c] + self.po[c]
                    ):  # (a dummy pass: the first pair again)
                        c0 = g0
                    src = wgt
                    if pg >= self.NGU:  # (latent passes: this column's slice of the chip's latent columns)
                        src, c0 = w_lat[dev], self.l0[c] + sum(self.wg[c][self.NGU : pg])
                    for j in range(NSG):
                        blk = src[j * KB * 32 : (j + 1) * KB * 32, c0 * 32 : (c0 + pw) * 32]
                        blocks.append(blk.reshape(KB, 32, pw, 32).permute(0, 2, 1, 3).reshape(-1, 32))
                for pd, pw in enumerate(self.wd[c]):
                    c0 = self.n0d[c] + sum(self.wd[c][:pd])
                    if self.rs is not None:  # (the plan's passes, in this chip's entry order: pos[dev] its position)
                        c0 = self.rs.column_passes(pos[dev], c)[pd][0]
                    for i in range(self.NSD):
                        blk = wdt[i * KBD * 32 : (i + 1) * KBD * 32, c0 * 32 : (c0 + pw) * 32]
                        blocks.append(blk.reshape(KBD, 32, pw, 32).permute(0, 2, 1, 3).reshape(-1, 32))
                for cy in range(GY):  # (a row may hold none: fewer blocks than rows without the down's)
                    reg = torch.cat(blocks[cy::GY]) if blocks[cy::GY] else torch.zeros(0, 32, dtype=wgt.dtype)
                    pad = self.wown_t * 32 - reg.shape[0]
                    regions.append(torch.cat([reg, torch.zeros(pad, 32, dtype=reg.dtype)]) if pad else reg)
            devs.append([r.reshape(-1, 32, 32) for r in regions])
        self.nbank = self.mesh.dram_grid_size().x
        return _bank_sharded(devs, self.nbank, self.w_dtype, self.mesh)

    def owner(self, j):
        """(owner column, family, index in the family) of x step j (the same every pass and chunk)."""
        f = j & 1
        fam, S = (self.fam1, self.S1) if f else (self.fam0, self.S0)
        return fam[(j // 2) % S], f, (j // 2) % S

    def pieces(
        self,
        x,
        w,
        y,
        *,
        x_k0=0,
        x_ct=None,
        shard_order=None,
        extra_sem_cores=(),
        ports=(),
        xloaded=0,
        rs_pos=None,
        rs_port=None,
        rs_mbox=0,
        rs_arr=(0, 0),
        rs_chunk=None,
        la=None,
        prep=None,
        rtr=None,
    ):
        """(CBs, kernels, semaphores) of the engine on one chip (the kernel_lib families attached by the caller via
        attach_mcast). x: DRAM [rows, x_ct tiles] (columns x_k0 .. of the full x; None: all; 0: none). shard_order: this
        chip's K shards in processing order (own first, then by arrival; XREADY = m (G - 1) + rank once chunk m's shard
        of that rank is in); None: one shard (all of K, already there). With an RsPlan: rs_pos this chip's position on
        the TP axis, rs_port(d, l) the NoC (x, y) of its port (direction, link), rs_mbox the ports' mailbox address,
        rs_arr the arrival counters (fwd, bwd)."""
        x_ct, x_k0 = int(self.KT if x_ct is None else x_ct), int(x_k0)
        so = [int(q) for q in shard_order] if shard_order is not None else [0]
        assert self.NSG % len(so) == 0
        sor = [self.NSG // len(so), len(so)] + so  # (steps a shard, shards, the order)
        # then: the gather's ports to tell once my own-shard x is loaded (NoC x, y; their counter) and my XLD id
        # and the ports' "router done" word (SE_RDONE: my router phase is over - see rtr_overlay)
        rdone = int((rs_chunk or {}).get("rdone", 0)) if self.rtr_overlay else 0
        assert rdone or not self.rtr_overlay, "the router's buffers overlay remote-written zones: a router-done word"
        tail = [len(ports), int(xloaded)] + [v for p_ in ports for v in p_] + [XLD, rdone]
        mesh, GX, GY, cx0, MT = self.mesh, self.GX, self.GY, self.cx0, self.MT
        b, off = self.base, self.off
        virt = lambda cx, cy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0 + cx, cy))  # noqa: E731
        allc = self.crs
        cbs = []

        def cb(i, key, size, dt, page, blocks=None):
            if (
                blocks is not None
            ):  # (a FIFO whose blocks may straddle its end: its region carries the room - plan check)
                room = min(e for e in self.seq_ends if e > off[key]) - off[key]
                need = _ring_pages(size // page, blocks) * page
                assert (
                    room >= need
                ), f"se_engine3: CB {key} needs {need} B (blocks {sorted(set(blocks))}), region {room}"
            d = ttnn.cb_descriptor_from_sharded_tensor(
                i, self.arena, address_offset=off[key] + self.arena_delta, total_size=size, core_ranges=allc
            )
            d.format_descriptors = [ttnn.CBFormatDescriptor(i, dt, page)]
            cbs.append(d)

        cb(XIN, "xin", 2 * MT * self.KB * 2048, ttnn.bfloat16, 2048)
        cb(WIN, "win", NWS * self.WSLOT * self.wtb, self.w_dtype, self.wtb)
        cb(HOWN, "hown", MT * self.POMAX * 2048, ttnn.bfloat16, 2048)
        if self.down:
            cb(HALL, "hall", self.HSL * GX * MT * self.POMAX * 2048, ttnn.bfloat16, 2048)
        if self.down and self.rs is None:
            cb(OUT, "out", NOUT * MT * self.PWDMAX * 2048, ttnn.bfloat16, 2048)
        if self.rs is not None:
            z = self.rs_sizes
            cb(RS_CB["hold"], "hold", z["hold"], ttnn.float32, 4096)
            for k in ("finb", "arrf", "arrb", "outf"):
                cb(RS_CB[k], k, z[k], ttnn.bfloat16, 2048)
            for k in ("finc", "psync", "ffree")[: 3 if self.NCH > 1 else 2]:
                cb(RS_CB[k], k, z[k], ttnn.bfloat16, 16)
        if self.LT:
            cb(LAT_CB, "lat", self.NCH * self.LATP * 2048, ttnn.bfloat16, 2048)
        if prep is not None:
            assert self.prep and self.rs is not None, "dispatch prep: built with prep=True, after the reduce-scatter"
            pk = "lat" if self.LT else "xown"
            pend = (
                off["latz"] + (self.shards - 1) * MT * self.LWMAX * 2048 + (self.NCH - 1) * self.LATP * 2048
                if self.LT
                else off["xown"] + self.xown_t * 2048
            )
            cb(PREP_CB["pu"], pk, pend - off[pk], ttnn.bfloat16, 2048)
            cb(PREP_CB["po"], "po", 2 * self.PFULL * 2048, ttnn.bfloat16, 2048)
            if self.LT:
                cb(PREP_CB["lrdy"], "lrdy", self._lrdy_bytes(), ttnn.bfloat16, 16)
        if self.xfree:
            cb(XFREE_CB, "xfree", 16 * self.NCH, ttnn.bfloat16, 16)
        if rtr is not None:
            KTO = self.rtr["KTO"]
            cb(RTR_CB["rx"], "rx", 2 * self.KC * MT * self.NCH * 2048, ttnn.bfloat16, 2048)  # (two rounds' buffers)
            cb(RTR_CB["rw"], "rw", 2 * self.ETC * self.KC * 2048, ttnn.bfloat16, 2048)
            cb(RTR_CB["ro"], "ro", self.ETC * MT * self.NCH * 4096, ttnn.float32, 4096)
        cb(ACC, "acc", self.acc_pages * 4096, ttnn.float32, 4096)
        xa_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        bl, nl = {}, {}  # (core (x, y) -> its full arg list)
        nf_ = self.NSG // 2  # (the x families, for the router's x from x own: x_gather3.tables' columns and slots)
        ftab = list(self.fam0) + list(self.fam1)
        for c in range(GX):
            n0_ = nf_ // self.S0 if c in self.fam0 else 0
            ftab += [n0_ + (nf_ // self.S1 if c in self.fam1 else 0), n0_]
        for c in range(GX):
            i0 = self.fam0.index(c) if c in self.fam0 else GX
            i1 = self.fam1.index(c) if c in self.fam1 else GX
            for cy in range(GY):
                # (the kernels' order: the core's own args first, then the span every core of the chip shares - from
                # x's address to y's (BRISC), from ty to the tail (NCRISC) - then the core's again; _common_span sends
                # the span once a chip as common args, se_ra in the kernels reads the full list's indices)
                bl[(cx0 + c, cy)] = (
                    [c, cy, self.po[c], self.h0[c], i0, i1, virt(c, cy).y]
                    + [
                        int(x.buffer_address()),
                        x_ct,
                        x_k0,
                        b + off["xown"],
                        b + off["xin"],
                        b + off["hown"],
                        b + off["hall"],
                        virt(0, cy).x,
                        virt(GX - 1, cy).x,
                    ]
                    + ([int(rtr["x"].buffer_address()), b + off["rx"]] if rtr is not None else [0, 0])
                    + ftab
                    + [virt(cc, cy).x for cc in range(GX)]
                    + sor
                    + tail
                    + [int(y.buffer_address())]
                    + self._rs_brisc(rs_pos, rs_port, rs_mbox, c, cy)
                    + self._la_brisc(rs_pos, rs_port, la, c, cy)
                )
                rtr_a = self._rtr_args(rtr, c, cy) if rtr is not None else []
                nl[(cx0 + c, cy)] = (
                    [
                        cy,
                        virt(c, cy).x,
                        ((c * GY + cy) // self.nbank) * self.wown_t * self.wtb,
                        self.n0d[c],
                        (c * GY + cy) % self.nbank,
                    ]
                    + rtr_a
                    + self.wg[c]
                    + self.wd[c]
                    + [
                        virt(c, 0).y,
                        virt(c, GY - 1).y,
                        int(w.buffer_address()),
                        b + off["wown"],
                        b + off["win"],
                        b + off["words"],
                        int(y.buffer_address()),
                    ]
                    + [virt(c, yy).y for yy in range(GY)]
                    + sor
                    + tail
                    + self._rs_ncrisc(rs_pos, rs_arr, c, cy, rs_port, rs_chunk)
                    + self._la_ncrisc(rs_pos, la, c, cy)
                    + (self._prep_ncrisc(rs_pos, prep, c, cy) if prep is not None else [])
                )
                b_span = (7, 9 + 2 + GX + len(sor) + len(tail) + 1)
                n_span = (5 + len(rtr_a) + len(self.wg[c]) + len(self.wd[c]), 7 + GY + len(sor) + len(tail))
        b_rt, b_crt = _common_span(bl, *b_span)
        n_rt, n_crt = _common_span(nl, *n_span)
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )  # noqa: E731
        bct = [
            GX,
            MT,
            self.KB,
            self.NSG,
            self.NPG,
            self.NCH,
            self.HT,
            self.POMAX,
            self.CR,
            self.S0,
            self.S1,
            self.KT,
            XREADY,
            HCNT,
            HFREE,
        ]
        nct = [
            GY,
            MT,
            self.KB,
            self.NSG,
            self.NPG,
            self.NCH,
            self.KBD,
            self.NSD,
            self.NPD,
            NWS,
            self.WSLOT,
            self.wtb,
            self.NTH,
            self.CR,
            self.PWDMAX,
            WV,
            WFREE,
            NOUT,
        ]
        nd = "" if self.down else "#define SE_NODOWN\n"
        nd += f"#define SE_HSL {self.HSL}\n#define SE_ACCP {self.acc_pages}u\n#define SE_DT {self.DT}\n"
        if self.NCH > 1 and self.shards > 1:  # (chunk m = block m of every owner: the owners' rows stay contiguous)
            nd += f"#define GROW(m, lr) (((lr) / {self.CR // self.shards}) * {self.rows // 32 // self.shards} + (m) * {self.CR // self.shards} + (lr) % {self.CR // self.shards})\n"
        else:
            nd += "#define GROW(m, lr) ((m) * CR + (lr))\n"
        if self.WRING:
            nd += f"#define SE_WRING {self.WRING}\n"
        if os.environ.get("SE_XPIPE") == "1":  # (A/B: the x owners send a step as soon as its own load landed)
            nd += f"#define SE_XPIPE\n#define SE_XL {int(os.environ.get('SE_XL', '3'))}u\n"
        if os.environ.get("SE_HXINC") == "1":  # (A/B, no RS: the down starts on h columns as they land)
            nd += "#define SE_HXINC\n"
        if os.environ.get("SE_BRW") == "1":  # (A/B: the W DRAM reads on the BRISC's NOC0, paced)
            assert not self.WRING, "SE_BRW: resident weights only"
            nd += f"#define SE_BRW\n#define SE_WS {int(self.wsync.buffer_address())}u\n"
            nd += f"#define SE_WPD {int(os.environ.get('SE_WPD', '4'))}u\n"
            if os.environ.get("SE_XN1") == "1":  # (A/B, the engine alone - x from DRAM, no ports: family-1 x on NOC1)
                assert self.rs is None, "SE_XN1: the engine alone only"
                nd += "#define SE_XN1\n"
            if os.environ.get("SE_BRW_DN1") == "1":  # (the down's W blocks on the NCRISC's NOC1, early)
                nd += f"#define SE_BRW_DN1\n#define SE_DN1_AT {int(os.environ.get('SE_DN1_AT', '14'))}u\n"
            nd += f"#define SE_WDD {int(os.environ.get('SE_WDD', '24'))}u\n"
            if os.environ.get("SE_BRW_BAR", "1") == "1":
                b0 = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0, 0))
                nd += f"#define SE_BRW_BAR\n#define SE_BAR_X {b0.x}\n#define SE_BAR_Y {b0.y}\n#define SE_BAR_N {GX * GY}u\n"

        if self.NCH > 1 and not self.xfree:  # (dispatch prep: every chunk's units in the last chunk's prep op - see
            # the worklog; x own a ring: every chunk's own, before its slot is reloaded)
            nd += "#define SE_PREP_LAST\n"
        if self.xring:
            nd += f"#define SE_XRING {self.XR}u\n"
        if self.xfree:
            nd += f"#define SE_XFREE {XFREE_CB}\n"
        if self.rs is not None:
            nd += f"#define SE_NENT {self.rs.G}\n"  # (the RS's entries - TP - not the gate / up's x shards, G)
            nd += "#define SE_RS\n" + "".join(f"#define SE_{k.upper()} {v}\n" for k, v in RS_CB.items())
            if self._fin_rows():
                nd += "#define SE_FIN_ROWS\n"
            nd += f"#define SE_P {self.rs.P}\n#define SE_ZF {self.rs.zone_tiles('fwd')}\n#define SE_ZB {self.rs.zone_tiles('bwd')}\n"
        if self.LT:
            assert self.rs is not None, "the latent's all-to-all rides the reduce-scatter's ports and args"
            nd += f"#define SE_LAT {LAT_CB}\n#define SE_NGU {self.NGU}\n#define SE_LWMAX {self.LWMAX}\n"
            nd += f"#define SE_LATP {self.LATP}\n#define SE_LRB {self.CR // self.shards}\n"
            nd += f"#define G_LT {self.shards * self.LT}\n"  # (the latent's width in tiles: every chip's slice)
        if rtr is not None:
            nct_len = len(nct + w_ct + y_ct) + (
                len(ttnn.TensorAccessorArgs(la["out"]).get_compile_time_args()) if self.LT else 0
            )
            nct_len += len(ttnn.TensorAccessorArgs(prep["xbuf"]).get_compile_time_args()) if prep is not None else 0
            nd += "#define SE_RTR\n" + "".join(f"#define SE_{k.upper()} {v}\n" for k, v in RTR_CB.items())
            nd += f"#define SE_KTO {self.rtr['KTO']}\n#define SE_ETC {self.ETC}\n#define SE_ET {self.rtr['ET']}\n"
            nd += f"#define SE_MTR {MT * self.NCH}\n#define SE_RG {self.RG}\n"  # (the logits: my rows of every chunk, at the start)
            # (a row tile's destination words: 3, or 6 with the router's split - one a half)
            nd += f"#define SE_RD {6 if rtr.get('split') else 3}\n"
            nd += f"#define SE_RTR_SPLIT {rtr['split']}u\n" if rtr.get("split") else ""
            nd += f"#define XACK_ID {XACK}\n#define WACK_ID {WACK}\n"
            nd += f"#define SE_GX {GX}\n#define RXR_ID {RXR}\n#define RWR_ID {RWR}\n"
            if rtr.get("gate"):
                nd += f"#define SE_RTR_GATE\n#define RGATE_ID {RGATE}\n"
            nd += "#define SE_RDONE\n" if self.rtr_overlay else ""

            nd += f"#define RXS_ID {RXS}\n#define SE_KC {self.KC}\n"
            if not self.xring:  # (x own holds every chunk's own shard: the router's x from it)
                nd += f"#define SE_RTRX_L1\n#define XRL_ID {XRL}\n"
            nd += f"#define SE_RTR_BCT0 {len(bct + xa_ct + ([] if self.down else y_ct))}\n"
            nd += f"#define SE_EW {int(self.ewords.buffer_address())}u\n"
            nd += f"#define SE_RTR_CT0 {nct_len}\n"

        if prep is not None:
            nd += "#define SE_PREP\n" + "".join(
                f"#define SE_{k.upper()} {v}\n" for k, v in PREP_CB.items() if k != "lrdy"
            )
            nd += f"#define SE_PREP_FULL {self.PFULL}\n#define SE_PREP_BCT {self.PBCT}\n"
            nd += f"#define SE_PREP_ROW {int(prep['xbuf'].buffer_aligned_page_size())}\n"
            nd += f"#define SE_PREP_CT0 {len(nct + w_ct + y_ct) + (len(ttnn.TensorAccessorArgs(la['out']).get_compile_time_args()) if self.LT else 0)}\n"
            if self.LT:
                nd += f"#define SE_LRDY {PREP_CB['lrdy']}\n"
        self.brisc = ttnn.KernelDescriptor(
            kernel_source=nd + _probes() + _BRISC,
            source_type=st,
            core_ranges=allc,
            compile_time_args=bct
            + xa_ct
            + ([] if self.down else y_ct)
            + (list(ttnn.TensorAccessorArgs(rtr["x"]).get_compile_time_args()) if rtr is not None else []),
            runtime_args=b_rt,
            common_runtime_args=b_crt,
            config=dm("RISCV_0", ttnn.NOC.NOC_0),
        )
        kernels = [
            self.brisc,
            ttnn.KernelDescriptor(
                kernel_source=nd + _probes() + _NCRISC,
                source_type=st,
                core_ranges=allc,
                compile_time_args=nct
                + w_ct
                + y_ct
                + (list(ttnn.TensorAccessorArgs(la["out"]).get_compile_time_args()) if self.LT else [])
                + (list(ttnn.TensorAccessorArgs(prep["xbuf"]).get_compile_time_args()) if prep is not None else [])
                + (
                    list(ttnn.TensorAccessorArgs(rtr["x"]).get_compile_time_args())
                    + list(ttnn.TensorAccessorArgs(rtr["w"]).get_compile_time_args())
                    if rtr is not None
                    else []
                ),
                runtime_args=n_rt,
                common_runtime_args=n_crt,
                config=dm("RISCV_1", ttnn.NOC.NOC_1),
            ),
        ]
        glu_code, glu_def = glu_defines(self.glu)
        for (wg, wd), cols in self.groups.items():
            crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(cx0 + c, 0), ttnn.CoreCoord(cx0 + c, GY - 1)) for c in cols]
            )
            comp = ttnn.ComputeConfigDescriptor()
            comp.math_fidelity = self.fid
            comp.fp32_dest_acc_en = True
            comp.dst_full_sync_en = self.DT == 8
            comp.math_approx_mode = False
            modes = [ttnn.UnpackToDestMode.Default] * 64  # (NUM_CIRCULAR_BUFFERS)
            modes[ACC] = ttnn.UnpackToDestMode.UnpackToDestFp32
            comp.unpack_to_dest_mode = modes
            c_rt = ttnn.RuntimeArgs()
            # (the per-group block - gate / down tables, the h column offsets, HT - and the prefix of the RS / prep
            # tables every core of the group shares go as the kernel's common args, sent once a chip instead of once a
            # core: ~86 of ~91 words; each core's runtime args are the rest, the kernel indexing them from SE_NC)
            pre = list(wg) + list(wd) + list(self.h0) + [self.HT]
            ras = {}
            for c in cols:
                for cy in range(GY):
                    ra = self.rs.compute_args(rs_pos, c, cy) if self.rs is not None else []
                    if prep is not None:
                        ra = self._prep_compute(ra, len(pre), rs_pos, prep, c, cy)
                    ras[(c, cy)] = ra
            shared = 0
            first = next(iter(ras.values()))
            while shared < len(first) and all(len(r) > shared and r[shared] == first[shared] for r in ras.values()):
                shared += 1
            common_c = pre + first[:shared]
            for (c, cy), ra in ras.items():
                c_rt[cx0 + c][cy] = ra[shared:] if len(ra) > shared else [0]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"#define SE_GLU {glu_code}\n" + glu_def + nd + _probes() + _COMPUTE,  # noqa
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=[
                        MT,
                        self.KB,
                        self.NSG,
                        self.NPG,
                        self.NCH,
                        self.HT,
                        self.POMAX,
                        self.KBD,
                        self.NSD,
                        self.NPD,
                        self.WSLOT,
                        self.PWDMAX,
                        len(so),
                        self.NSG // len(so),
                        GX,
                        len(common_c),
                    ],
                    runtime_args=c_rt,
                    common_runtime_args=common_c,
                    config=comp,
                )
            )
        semc = (
            allc
            if not extra_sem_cores
            else ttnn.CoreRangeSet(list(allc.ranges()) + [ttnn.CoreRange(c, c) for c in extra_sem_cores])
        )
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=semc, initial_value=0) for i in range(NSEM_OWN)]
        return cbs, kernels, sems

    def _rs_brisc(self, p, port, mbox, cx, cy):
        """The BRISC's hand-offs: [count, my mailbox word] + per hand-off its port's NoC (x, y)."""
        if self.rs is None:
            return []
        rs = self.rs
        k = rs.k(cx, cy)
        hs = rs.handoffs(p, cx, cy)
        return [len(hs), mbox + persist_l1.handoff_bytes() * k] + [v for _, d in hs for v in port(d, rs.link(k))]

    def la_rows(self, cy, b):
        """(first local row, rows) of core row cy's MT row tiles in owner block b (RB = CR / G row tiles a block)."""
        RB = self.CR // self.shards
        rs = [r for r in range(self.MT) if (cy * self.MT + r) // RB == b]
        return (rs[0], len(rs)) if rs else (0, 0)

    def la_dir(self, p, b):
        """The direction chip p sends block b's latent (shortest; ring ties the fabric's way)."""
        G = self.shards
        if not self.rs.kind.startswith("ring"):
            return "fwd" if b > p else "bwd"
        return self.ring_dir(p, b)  # (the gather's: x_gather3 XGather3.ring_dir - ties as the fabric routes them)

    def _la_brisc(self, p, port, la, cx, cy):
        """The latent hand-offs: [count, my port word] + per destination block with rows of mine its port's NoC."""
        if not self.LT:
            return []
        k = cy * self.GX + cx
        dst = [b for b in range(self.shards) if b != p and self.la_rows(cy, b)[1]]
        if "NOA2A" in _probes():  # (timing / hang probe: no all-to-all)
            dst = []
        return [len(dst), la["word"] + persist_l1.handoff_bytes() * k] + [
            v for b in dst for v in port(self.la_dir(p, b), k % self.rs.nv)
        ]

    def _la_ncrisc(self, p, la, cx, cy):
        if not self.LT:
            return []
        r0, nr = self.la_rows(cy, p)
        RB = self.CR // self.shards
        return [
            nr,
            r0,
            self.shards - 1 if nr and "NOA2A" not in _probes() else 0,
            la["ctr"],
            self.base + self.off["latz"],
            self.LT,
            self.l0[cx],
            self.lw[cx],
            p,
            cy * self.MT + r0 - p * RB if nr else 0,
            int(la["out"].buffer_address()),
        ]

    def _rtr_args(self, rtr, cx, cy):
        """The router logits' NCRISC args (SE_RTR): x, W_r^T, my column's expert tiles (valid count, first), RX / RW,
        the row's NoC x (far end, first core - NOC1 multicasts from the far corner), my row's NoC y, my column has
        experts, per row tile of mine its partials' destination (NoC, base, count address: router engine_args), the
        W gate's counter (NoC, address)."""
        GX, ETC, ET = self.GX, self.ETC, self.rtr["ET"]
        e0 = cx * ETC
        nv = max(0, min(ETC, ET - e0))
        virt = lambda c, y: self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(self.cx0 + c, y))  # noqa: E731
        return (
            [int(rtr["x"].buffer_address()), int(rtr["w"].buffer_address()), nv, e0]
            + [self.base + self.off["rx"], self.base + self.off["rw"]]
            + [virt(GX - 1, cy).x, virt(0, cy).x, virt(0, cy).y, int(nv > 0)]
            + [
                w
                for rr in range(self.MT * self.NCH)
                for w in rtr["dests"][self.grow(rr // self.MT, cy * self.MT + rr % self.MT)]
            ]
            + list(rtr["gate_ctr"])
        )

    def gate_target(self):
        """Where the router cores multicast their "partials in" words (SE_RTR_GATE): the engine grid's NoC rectangle
        (top-left, bottom-right: a NOC0 multicast), its core count, the words' address (RGATE + 16 j)."""
        virt = [
            self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(self.cx0 + c, y))
            for c in range(self.GX)
            for y in range(self.GY)
        ]
        rect = (min(v.x for v in virt), min(v.y for v in virt), max(v.x for v in virt), max(v.y for v in virt))
        return dict(rect=rect, n=self.GX * self.GY, addr=int(self.ewords.buffer_address()) + RGATE)

    def prep_units(self, p, prep, cx, cy):
        """Dispatch prep's units on core (cx, cy) of chip position p: [(source offset in the PU view, tiles, latent
        arrival?, dispatch buffer row, byte offset in the row)] - one row tile's run of x own (a K step) or of a latent
        block (mine, then each source chip's slice), for the chip's owned rows (block p). Chunk-major: chunk m's units
        (the same count a chunk) run in the chunk's prep op."""
        MT, KB, G = self.MT, self.KB, self.shards
        RBo = self.rows // 32 // G  # (owned row tiles)
        row0 = prep["me"] * RBo * 32
        out = []
        if self.LT:
            r0, nr = self.la_rows(cy, p)
            lz = self.off["latz"] - self.off["lat"]
            rbc = self.CR // G
            for m in range(self.NCH):  # (chunk m: its LAT / LATZ at m LATP pages, the owner's rows from m rbc)
                cm = m * self.LATP * 2048
                lorow = cy * MT + r0 - p * rbc + m * rbc
                for i in range(nr):
                    rr, n = (r0 + i) * self.LWMAX * 2048, self.lw[cx]
                    out.append((cm + rr, n, 0, row0 + (lorow + i) * 32, (p * self.LT + self.l0[cx]) * 64))
                    for sl in range(G - 1):
                        q = sl if sl < p else sl + 1
                        out.append(
                            (
                                cm + lz + sl * MT * self.LWMAX * 2048 + rr,
                                n,
                                1,
                                row0 + (lorow + i) * 32,
                                (q * self.LT + self.l0[cx]) * 64,
                            )
                        )
            return out
        so = prep["shard_order"]
        sps = self.NSG // len(so)
        rank = {q: k for k, q in enumerate(so)}
        nf = self.NSG // 2
        n0 = nf // self.S0 if cx in self.fam0 else 0
        nown = n0 + (nf // self.S1 if cx in self.fam1 else 0)
        rbc = self.CR // G
        for u_ in range(RBo):  # (owned row tile u_: chunk u_ // rbc, chunk-local row p rbc + u_ % rbc - grow's inverse)
            m, lr = (u_ // rbc, p * rbc + u_ % rbc) if self.NCH > 1 else divmod(p * RBo + u_, self.CR)
            R = p * RBo + u_
            if lr // MT != cy:
                continue
            r = lr % MT
            for j in range(len(so) * sps):  # (global K step j: K tiles j KB ..)
                pos = rank[j // sps] * sps + j % sps
                col, f, _ = self.owner(pos)
                if col != cx:
                    continue
                h = pos // 2
                slot = n0 + h // self.S1 if f else h // self.S0
                out.append(
                    (
                        (((m % self.XR) * nown + slot) * MT + r) * KB * 2048,
                        KB,
                        0,
                        row0 + (R - p * RBo) * 32,
                        j * KB * 64,
                    )
                )
        return out

    def _prep_compute(self, ra, n_pre, p, prep, cx, cy):
        """The RS compute tables with the prep op (kind 3) right after the last pass (under the reduce-scatter's waits
        for arrivals; spread over the passes it delayed them: 282-286 us vs 272-273, at the end it delayed the tail's
        fins: 283) and its units ([n] + (offset, tiles | latent << 16)) at the end."""
        nops = ra[0]
        ops, rest = ra[1 : 1 + nops], ra[1 + nops :]
        last = max(i for i, o in enumerate(ops) if o & 0xFF == 0)
        base = n_pre + 1 + nops + 1 + len(rest)
        us = self.prep_units(p, prep, cx, cy)
        ops = ops[: last + 1] + [3 | (base << 8)] + ops[last + 1 :]
        assert len(us) % self.NCH == 0
        return (
            [nops + 1]
            + ops
            + rest
            + [len(us) // self.NCH]
            + [v for o, n, lat, _, _ in us for v in (o, n | (lat << 16))]
        )

    def _prep_ncrisc(self, p, prep, cx, cy):
        us = self.prep_units(p, prep, cx, cy)
        return [len(us), int(prep["xbuf"].buffer_address())] + [v for _, n, _, row, co in us for v in (row, co, n * 64)]

    def grow(self, m, lr):
        """Chunk m's local row tile lr -> the global row tile: chunk m holds block m of every owner (RBc = CR / G row
        tiles of each), so an owner's rows stay contiguous (one chunk: lr)."""
        if self.NCH == 1 or self.shards == 1:
            return m * self.CR + lr
        rbc = self.CR // self.shards
        return (lr // rbc) * (self.rows // 32 // self.shards) + m * rbc + lr % rbc

    def _rs_ncrisc(self, p, arr, cx, cy, port=None, chunk=None):
        """The NCRISC's arrival credits (per direction, slot order: thresholds) and the fin sub-block's y map; with
        chunks: the fin's y0 of chunks 1 .., the ports facing my fwd / bwd upstream (zone-free signals), their word, my
        FINB-free word, the ports reading my FINB."""
        if self.rs is None:
            return []
        rs = self.rs
        a = rs.arrivals(p, cx, cy)
        zs = {d: rs.zone_slots(p, cx, cy, d) for d in ("fwd", "bwd")}
        y0, stride, w = rs.fin_map(p, cx, cy)
        # (a slot's word: its threshold | the zone pages it takes << 16)
        out = (
            [len(a["fwd"])]
            + [t | zs["fwd"][sl][1] << 16 for _, sl, t in a["fwd"]]
            + [len(a["bwd"])]
            + [t | zs["bwd"][sl][1] << 16 for _, sl, t in a["bwd"]]
            + [rs.count(cx, cy, p), y0, stride, w, arr[0], arr[1]]
        )
        if self.NCH == 1:
            return out
        if rs.scatter == -1:  # (y [rows, NB]: my rows of chunk m - chunk 0's too)
            if rs.count(cx, cy, p):
                out[len(out) - 5] = y0 + (self.grow(0, cy * self.MT) - cy * self.MT) * rs.NB
            y0s = [y0 + (self.grow(m, cy * self.MT) - cy * self.MT) * rs.NB for m in range(1, self.NCH)]
        else:  # (the owner's y [rows / G, NTH]: chunk m's block at its row m RB)
            y0s = [y0 + m * rs.RB * rs.NTH for m in range(1, self.NCH)]
        link = rs.link(rs.k(cx, cy))

        def facing(d):  # (the port facing my d-upstream: the opposite direction's, if I take arrivals from d)
            if not a[d]:
                return 0
            x_, y_ = port("bwd" if d == "fwd" else "fwd", link)
            return x_ | y_ << 16

        out = (
            out + y0s + [facing("fwd"), facing("bwd"), chunk["zfree"], chunk["ffree"], len(rs.finb_readers(p, cx, cy))]
        )
        if self._fin_rows():  # (each (chunk, row)'s y base: y0's column offset + its global row's)
            col = y0 - cy * self.MT * rs.NB
            out += [col + self.grow(m, cy * self.MT + r) * rs.NB for m in range(self.NCH) for r in range(self.MT)]
        return out

    def _fin_rows(self):
        """y rows a core finishes not contiguous: with chunks, rows scattered (y [rows, NB]) and owner blocks of CR / G
        row tiles not a multiple of MT - a core's MT rows straddle two owners (K3 768 rows at GY 6 x MT 2: blocks of
        3), whose global rows jump (the fin wrote the next owner's first row after the last one's: a PCC loss)."""
        rs = self.rs
        return (
            rs is not None
            and rs.scatter == -1
            and self.NCH > 1
            and self.shards > 1
            and (self.CR // self.shards) % self.MT != 0
        )

    def attach_mcast(self, prog):
        """The kernel_lib families (x slots 0 / 1, h; the BRISC's NOC0) attached to prog's BRISC kernel (after prog.semaphores is set:
        attachment allocates its semaphore ids past ours)."""
        GX, GY, cx0 = self.GX, self.GY, self.cx0
        rect = self.crs
        cols = lambda cs: ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(cx0 + c, 0), ttnn.CoreCoord(cx0 + c, GY - 1)) for c in cs]
        )  # noqa: E731

        def owners(fam):  # (a family of one column - S 1: GLM-Flash at 640 rows, K step 16 - is a fixed sender a row,
            # that column's core: the unified Mcast refuses a rotation of one, which Mcast1D built as a fixed sender)
            if len(fam) > 1:
                return ttnn.McastRotatingSenderConfig(sender_cores=cols(fam))
            return ttnn.McastFixedSenderConfig(sender_index=fam[0])

        fams = [
            ("x0", owners(self.fam0)),
            ("x1", owners(self.fam1)),
            ("h", ttnn.McastRotatingSenderConfig()),
        ][: 3 if self.down else 2]
        for name, cfg in fams:  # (one family a row of the engine: the receiver groups are its rows)
            mc = ttnn.Mcast(self.mesh, ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True), rect, GX, cfg)
            mc.attach(prog, name, [self.brisc], next_semaphore_id(prog))

    def run_alone(self, x, w):
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        prog = ttnn.ProgramDescriptor()
        cbs, kernels, sems = self.pieces(x, w, y)
        prog.cbs, prog.semaphores = cbs, sems
        self.attach_mcast(prog)
        prog.kernels = kernels
        R, C = tuple(self.mesh.shape)
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
        ttnn.generic_op([x, w, y, self.arena], md)
        return y
