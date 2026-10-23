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

import torch
from ttnn.bringup.combine_rf.lin2d import column_split, glu_weight  # noqa: F401

import ttnn

NWS = 3  # (W slots)
XREADY, WV, WFREE = 0, 1, 1 + NWS  # (hand-written semaphores; the kernel_lib families' come after)
XLD = 1 + 2 * NWS  # (my own-shard x is loaded: the NCRISC's W load may start)
HCNT, HFREE = 1 + 2 * NWS, 3 + 2 * NWS  # (unused: the h exchange is the kernel_lib family again)
NSEM_OWN = 2 + 2 * NWS
# (the router logits on the engine, SE_RTR: words of a zeroed L1 tensor - a K chunk of x rows / W_r^T landed (the
# round), the leads' acks - 16 B apart; the semaphore ids are all taken by the kernel_lib families)
RXR, RWR, RXA, RWA = 0, 16, 32, 48
RGATE = 64  # (every engine core's partials written: the W streams' gate, SE_RTR_GATE)
XIN, WIN, HOWN, HALL, OUT, ACC = 0, 1, 2, 3, 4, 5  # CB ids
NOUT = 2  # (down passes OUT holds)
RS_CB = dict(hold=6, finb=7, arrf=8, arrb=9, finc=10, outf=11, psync=12)  # (down + reduce-scatter CBs)
LAT_CB = 13  # (the latent block [MT][LWMAX], bf16)
PREP_CB = dict(pu=14, po=15, lrdy=16)
RTR_CB = dict(
    rx=17, rw=18, ro=19
)  # (the router logits on the engine: x rows [KTO][MT], W_r^T [ETC][KTO], out fp32)  # (dispatch prep: the units' source view, the untilized rows, latent ready)
ARENA_MAX = 1400 * 1024  # (L1 a core the arena may take: Blackhole's 1461248-byte bank less some slack)

_BRISC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "tools/profiler/kernel_profiler.hpp"
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
constexpr uint32_t CB_XIN = 0, CB_HOWN = 2, CB_HALL = 3;
constexpr uint32_t NF = NSG / 2, NOWN0 = NF / S0, NOWN1 = NF / S1;  // (steps a family a pass; per owner)
// x position j (the j-th K step this chip processes: its own shard's first, then each shard as it lands): family j % 2,
// owner = the family's subset[(j / 2) % S] (S divides a shard's steps / 2, so the rotation over (shard, pass, step)
// rounds keeps every position on one owner), its local index (j / 2) / S
FORCE_INLINE uint32_t own_slot(uint32_t j) { return (j & 1) ? (j / 2) / S1 : (j / 2) / S0; }
void kernel_main() {
    size_t a = 0;
    const uint32_t cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);
    const uint32_t x_addr = get_arg_val<uint32_t>(a++), x_ct = get_arg_val<uint32_t>(a++), x_k0 = get_arg_val<uint32_t>(a++);
    const uint32_t xown = get_arg_val<uint32_t>(a++), xin = get_arg_val<uint32_t>(a++), hown = get_arg_val<uint32_t>(a++);
    const uint32_t hall = get_arg_val<uint32_t>(a++), po = get_arg_val<uint32_t>(a++), h0 = get_arg_val<uint32_t>(a++);
    const uint32_t i0 = get_arg_val<uint32_t>(a++), i1 = get_arg_val<uint32_t>(a++);  // my index in family 0 / 1 (or GX)
    const uint32_t lx = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t vx0 = a, sps = get_arg_val<uint32_t>(vx0 + GX), g = get_arg_val<uint32_t>(vx0 + GX + 1);
    const uint32_t ord0 = vx0 + GX + 2;  // (the row's NoC x; steps a shard, shards; the shard order: own, then arrival)
    auto step_of = [&](uint32_t j) { return get_arg_val<uint32_t>(ord0 + j / sps) * sps + j % sps; };  // (position j)
    const uint32_t nown = (i0 < GX ? NOWN0 : 0) + (i1 < GX ? NOWN1 : 0);  // (my steps a chunk)
    auto mine = [&](uint32_t j) { return (j & 1) ? ((j / 2) % S1 == i1) : ((j / 2) % S0 == i0); };
    // my steps' x-own slot: [chunk][family 0 slots, then family 1][MT][KB]
    auto own_addr = [&](uint32_t m, uint32_t j) {
        const uint32_t slot = (j & 1) ? (i0 < GX ? NOWN0 : 0) + own_slot(j) : own_slot(j);
        return xown + ((m * nown + slot) * MT) * KB * 2048;
    };
    if (x_ct) {  // (alone) my steps of every chunk from DRAM x [rows, x_ct]: columns x_k0 ..
        DeviceZoneScopedN("SE-XLOAD");
        const auto xg = TensorAccessor(x_args, x_addr, 2048);
        for (uint32_t m = 0; m < NCH; ++m)
            for (uint32_t j = 0; j < NSG; ++j) {
                const uint32_t s = step_of(j);
                if (mine(j) && s * KB >= x_k0 && s * KB < x_k0 + x_ct)
                    for (uint32_t r = 0; r < MT; ++r)
                        for (uint32_t kk = 0; kk < KB; ++kk)
                            noc_async_read(xg.get_noc_addr((m * CR + cy * MT + r) * x_ct + s * KB + kk - x_k0),
                                           own_addr(m, j) + (r * KB + kk) * 2048, 2048);
            }
        noc_async_read_barrier();
    }
    {  // (my own-shard x is loaded: count it on the gather's ports - they send it from here - and let my W load start)
        const uint32_t np0 = ord0 + g, nports = get_arg_val<uint32_t>(np0), xl_addr = get_arg_val<uint32_t>(np0 + 1);
        for (uint32_t i = 0; i < nports; ++i)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(np0 + 2 + 2 * i), get_arg_val<uint32_t>(np0 + 3 + 2 * i), xl_addr), 1);
        noc_async_atomic_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(np0 + 2 + 2 * nports))) = 1;
    }
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
        {
        DeviceZoneScopedN("SE-XROUNDS");
        // shard-major: each shard (own first, then as it lands) through every pass before the next shard
        for (uint32_t k = 0; k < g; ++k)
            for (uint32_t pg = 0; pg < NPG; ++pg)
            for (uint32_t st = 0; st < sps; ++st) {
                const uint32_t j = k * sps + st;  // (position; its family j % 2 = st % 2: a block's rounds are even)
                const uint32_t round = ((m * g + k) * NPG + pg) * (sps / 2) + st / 2;  // (the family's)
                cb_reserve_back(CB_XIN, MT * KB);
                const uint32_t dst = xin + (j & 1) * MT * KB * 2048;
                if (mine(j)) {  // (position j: the shard of rank j / sps must be in - the own shard always is)
                    if (j >= sps) noc_semaphore_wait_min(xready, m * (g - 1) + j / sps);
#ifdef SE_PROBE_NOX
                    constexpr uint32_t XBYTES = 16;  // (timing probe: no x data)
#else
                    constexpr uint32_t XBYTES = MT * KB * 2048;
#endif
                    if (j & 1) s1->send(own_addr(m, j), dst, XBYTES); else s0->send(own_addr(m, j), dst, XBYTES);
                } else {
                    if (j & 1) r1->receive(round); else r0->receive(round);
                }
                cb_push_back(CB_XIN, MT * KB);
            }
        }
#ifdef SE_LAT
        {  // (the latent's all-to-all: once it is packed, +1 on each destination's port word of mine)
            DeviceZoneScopedN("SE-LHAND");
            const uint32_t np0 = ord0 + g, nports = get_arg_val<uint32_t>(np0), r0 = np0 + 4 + 2 * nports;
            const uint32_t l0 = r0 + 2 + 2 * get_arg_val<uint32_t>(r0), nl = get_arg_val<uint32_t>(l0);
            const uint32_t lw = get_arg_val<uint32_t>(l0 + 1);
            cb_wait_front(SE_LAT, MT * SE_LWMAX);
            for (uint32_t j = 0; j < nl; ++j)
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(l0 + 2 + 2 * j), get_arg_val<uint32_t>(l0 + 3 + 2 * j), lw), 1);
        }
#endif
#ifdef SE_NODOWN
        {  // (no down here: my h block [MT][po] to DRAM h [rows, HT] - the down + reduce-scatter reads it)
        DeviceZoneScopedN("SE-HOUT");
        const uint32_t np0 = ord0 + g, nports = get_arg_val<uint32_t>(np0);
        const auto hg = TensorAccessor(h_args, get_arg_val<uint32_t>(np0 + 3 + 2 * nports), 2048);
        cb_wait_front(CB_HOWN, MT * POMAX);
        for (uint32_t r = 0; r < MT; ++r)
            for (uint32_t t = 0; t < po; ++t)
                noc_async_write(hown + (r * POMAX + t) * 2048, hg.get_noc_addr((m * CR + cy * MT + r) * HT + h0 + t), 2048);
        noc_async_write_barrier();
        cb_pop_front(CB_HOWN, MT * POMAX);
        }
#else
        DeviceZoneScopedN("SE-HX");
        // h: every core's h block [MT][POMAX] into every row core's h-all slot (m % 2) as block cx ([GX][MT][POMAX]):
        // one kernel_lib rotating round a column; the receivers' readiness ack is the slot's credit
        constexpr uint32_t HA = GX * MT * POMAX;
        cb_wait_front(CB_HOWN, MT * POMAX);
        cb_reserve_back(CB_HALL, HA);
        const uint32_t hs = hall + (NCH > 1 ? (m & 1) : 0) * HA * 2048;  // (one slot when there is one chunk)
        for (uint32_t c = 0; c < GX; ++c) {
            if (c == cx) sh->send(hown, hs + c * MT * POMAX * 2048, MT * POMAX * 2048);
            else rh->receive(m * GX + c);
#ifdef SE_RS
            cb_push_back(CB_HALL, MT * POMAX);  // (column c's block: the down's first pass starts on its K tiles)
#endif
        }
#ifndef SE_RS
        cb_push_back(CB_HALL, HA);
#endif
        cb_pop_front(CB_HOWN, MT * POMAX);
#endif
    }
#ifdef SE_RS
    {  // (down + reduce-scatter) each finished sub-block's hand-off: +1 on its port's mailbox word of mine
        DeviceZoneScopedN("SE-HAND");
        const uint32_t np0 = ord0 + g, nports = get_arg_val<uint32_t>(np0), r0 = np0 + 4 + 2 * nports;
        const uint32_t nh = get_arg_val<uint32_t>(r0), mbox = get_arg_val<uint32_t>(r0 + 1);
        for (uint32_t j = 0; j < nh; ++j) {
            cb_wait_front(SE_FINC, j + 1);
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(r0 + 2 + 2 * j), get_arg_val<uint32_t>(r0 + 3 + 2 * j), mbox), 1);
        }
    }
#endif
    noc_async_atomic_barrier();
}
"""

_NCRISC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
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
void kernel_main() {
    size_t a = 0;
    const uint32_t cy = get_arg_val<uint32_t>(a++), cxv = get_arg_val<uint32_t>(a++);
    const uint32_t ty = get_arg_val<uint32_t>(a++), by = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++), wbase = get_arg_val<uint32_t>(a++), wown = get_arg_val<uint32_t>(a++);
    const uint32_t win = get_arg_val<uint32_t>(a++), words = get_arg_val<uint32_t>(a++), y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n0d = get_arg_val<uint32_t>(a++);
#ifdef SE_RTR
    const uint32_t R0 = a;  // (the router logits' args)
    a += 12 + 3 * MT;
#endif
    const uint32_t wg0 = a, wd0 = wg0 + NPG, ys0 = wd0 + NPD;  // pass widths (gate / up, down); the column's NoC y
    const uint32_t wbank = get_arg_val<uint32_t>(ys0 + GY);  // (the DRAM bank holding my W-own region)
    const uint32_t sps = get_arg_val<uint32_t>(ys0 + GY + 1), ord0 = ys0 + GY + 3;  // (steps a shard; the shard order)
    // round w's block (natural index): a gate / up pass's steps in this chip's processing order, then down's
    auto blk = [&](uint32_t w) {
        const uint32_t c = w % NB;
        if (c >= NPG * NSG) return c;
        const uint32_t k = c / (NPG * sps), rem = c % (NPG * sps), pg = rem / sps, st = rem % sps;
        return pg * NSG + get_arg_val<uint32_t>(ord0 + k) * sps + st;  // (shard-major: shard, pass, step)
    };
    auto bsize = [&](uint32_t b) {  // (tiles)
        return b < NPG * NSG ? KB * get_arg_val<uint32_t>(wg0 + b / NSG) : KBD * get_arg_val<uint32_t>(wd0 + (b - NPG * NSG) / NSD1);
    };
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
    const uint32_t etc = get_arg_val<uint32_t>(R0 + 2), re0 = get_arg_val<uint32_t>(R0 + 3);
    const uint32_t rxa = get_arg_val<uint32_t>(R0 + 4), rwa = get_arg_val<uint32_t>(R0 + 5);
    {
        // K chunks of SE_KC tiles, one buffer each: a round's leads wait for every receiver's ack (its compute popped
        // the last round's chunk: its slot reserved again), read the chunk from DRAM, multicast it and set the round
        DeviceZoneScopedN("SE-RTRLD");
        const bool lx = cxv == get_arg_val<uint32_t>(R0 + 7), lw = etc && cy == 0;
        const bool hw = get_arg_val<uint32_t>(R0 + 9);
        const uint32_t x0 = get_arg_val<uint32_t>(R0 + 6), x1 = get_arg_val<uint32_t>(R0 + 7), yv = get_arg_val<uint32_t>(R0 + 8);
        const auto xg = TensorAccessor(rx_args, get_arg_val<uint32_t>(R0), 2048);
        const auto wg2 = TensorAccessor(rw_args, get_arg_val<uint32_t>(R0 + 1), 2048);
        for (uint32_t rd = 0; rd < SE_KTO / SE_KC; ++rd) {
            cb_reserve_back(SE_RX, SE_KC * MT);
            cb_reserve_back(SE_RW, SE_ETC * SE_KC);
            if (rd > 0) {  // (my slots are free again: tell the leads)
                noc_semaphore_inc(get_noc_addr(x1, yv, (SE_EW + RXA_ID)), 1);
                if (hw) noc_semaphore_inc(get_noc_addr(cxv, ty, (SE_EW + RWA_ID)), 1);
            }
            const uint32_t k0 = rd * SE_KC;
            if (lx) {  // (the row's first core: the chunk of its rows' x to the row)
                for (uint32_t r = 0; r < MT; ++r)
                    for (uint32_t k = 0; k < SE_KC; ++k)
                        noc_async_read(xg.get_noc_addr((cy * MT + r) * SE_KTO + k0 + k), rxa + (k * MT + r) * 2048, 2048);
                if (rd > 0) noc_semaphore_wait_min(ew(RXA_ID), rd * SE_GX);
                noc_async_read_barrier();
                noc_async_write_multicast_loopback_src(rxa, get_noc_multicast_addr(x0, yv, x1, yv, rxa), SE_KC * MT * 2048, SE_GX);
                noc_async_writes_flushed();
                *ew(RXR_ID) = rd + 1;
                noc_semaphore_set_multicast_loopback_src((SE_EW + RXR_ID), get_noc_multicast_addr(x0, yv, x1, yv, (SE_EW + RXR_ID)), SE_GX);
            }
            if (lw) {  // (the column's first core: the chunk of its expert tiles of W_r^T down the column)
                for (uint32_t e = 0; e < etc; ++e)
                    for (uint32_t k = 0; k < SE_KC; ++k)
                        noc_async_read(wg2.get_noc_addr((re0 + e) * SE_KTO + k0 + k), rwa + (e * SE_KC + k) * 2048, 2048);
                if (rd > 0) noc_semaphore_wait_min(ew(RWA_ID), rd * GY);
                noc_async_read_barrier();
                noc_async_write_multicast_loopback_src(rwa, get_noc_multicast_addr(cxv, by, cxv, ty, rwa), etc * SE_KC * 2048, GY);
                noc_async_writes_flushed();
                *ew(RWR_ID) = rd + 1;
                noc_semaphore_set_multicast_loopback_src((SE_EW + RWR_ID), get_noc_multicast_addr(cxv, by, cxv, ty, (SE_EW + RWR_ID)), GY);
            }
            noc_async_writes_flushed();
            while (*ew(RXR_ID) < rd + 1) invalidate_l1_cache();
            cb_push_back(SE_RX, SE_KC * MT);
            while (hw && *ew(RWR_ID) < rd + 1) invalidate_l1_cache();
            cb_push_back(SE_RW, SE_ETC * SE_KC);
        }
        noc_async_atomic_barrier();
        if (lx) *ew(RXA_ID) = 0;  // (re-armed: every ack landed - the last round waited for them; no more sets come)
        if (lw) *ew(RWA_ID) = 0;
        *ew(RXR_ID) = 0;
        *ew(RWR_ID) = 0;
    }
    {  // my partial logits (the compute's, a few us later) to their router cores' HOLD - before my W stream starts
        DeviceZoneScopedN("SE-RTROUT");
        cb_wait_front(SE_RO, SE_ETC * MT);
        // (row tile r of mine: my own owner's block into its router core's HOLD, another owner's straight into the
        // port sub-block it leaves from - (NoC, base, count) a row tile; the count ETC tiles, after they landed)
        const uint32_t ro = get_read_ptr(SE_RO);
        for (uint32_t r = 0; r < MT; ++r) {
            const uint32_t dw = get_arg_val<uint32_t>(R0 + 10 + 3 * r), dx = dw & 0xFFFF, dy = dw >> 16;
            const uint32_t base = get_arg_val<uint32_t>(R0 + 11 + 3 * r), ctr = get_arg_val<uint32_t>(R0 + 12 + 3 * r);
            for (uint32_t e = 0; e < etc; ++e)
                noc_async_write(ro + (e * MT + r) * 4096, get_noc_addr(dx, dy, base + (re0 + e) * 4096), 4096);
            noc_async_write_barrier();  // (landed before the count)
            if (etc) noc_semaphore_inc(get_noc_addr(dx, dy, ctr), etc);
        }
        cb_pop_front(SE_RO, SE_ETC * MT);
#ifdef SE_RTR_GATE
        {  // (my partials are written: router core 0 counts every engine core's, then opens the W gate)
            const uint32_t gw = get_arg_val<uint32_t>(R0 + 10 + 3 * MT);
            noc_semaphore_inc(get_noc_addr(gw & 0xFFFF, gw >> 16, get_arg_val<uint32_t>(R0 + 11 + 3 * MT)), 1);
        }
#endif
    }
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
            const uint32_t t0 = ord0 + get_arg_val<uint32_t>(ys0 + GY + 2);  // (past the shard order: the tail)
            const uint32_t xld = get_arg_val<uint32_t>(t0 + 2 + 2 * get_arg_val<uint32_t>(t0));
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xld)) == 0) invalidate_l1_cache();
        }
    }
    // the weights' DRAM reads go in stream order a window ahead of the sends (WPF of my blocks in flight): spread
    // over gate / up at the rate the stream uses them - a burst of all 35 MB at the start starved the gather
    constexpr uint32_t WPF = 3;
    uint8_t trid_of[64];
    uint32_t wi = 0, ni = 0, pending = 0;  // (prefetch cursor over the first chunk's rounds; issued; issued unsent)
    auto offset_of = [&](uint32_t b) {
        uint32_t o = 0;
        for (uint32_t b2 = cy; b2 < b; b2 += GY) o += bsize(b2);
        return o;
    };
    auto prefetch = [&](uint32_t upto) {  // (issue my blocks of rounds < upto, and the window)
        while (wi < NB && (wi < upto || pending < WPF)) {
            const uint32_t b = blk(wi++);
            if (b % GY != cy) continue;
            const uint32_t t = (ni++ % 14) + 1, o = offset_of(b);
            trid_of[b / GY] = t;
            noc_async_read_set_trid(t);
#ifdef SE_PROBE_NOWL
            noc_async_read(get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + o * WTB), wown + o * WTB, 16);  // (probe)
#else
            noc_async_read(get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + o * WTB), wown + o * WTB, bsize(b) * WTB);
#endif
            ++pending;
        }
    };
    const uint32_t NR = NCH * NB;  // (W rounds a call)
    uint32_t ws = 0, woff = 0, wr = 0, wc = NWS, p = 0, pm = 0;  // (send, my offset, pushed, credited; y pass / chunk)
    uint32_t nf[NWS] = {};
    while (ws < NR && blk(ws) % GY != cy) ++ws;  // (skip rounds I do not hold)
#ifdef SE_RS
    // (down + reduce-scatter) arrivals per direction in zone-slot order: slot i credited (P pages pushed) once my
    // arrival counter reached its threshold (0: nothing comes - a block I hold no tile of); the fin sub-block to y
    const uint32_t t0r = ord0 + get_arg_val<uint32_t>(ys0 + GY + 2);
    const uint32_t q0 = t0r + 3 + 2 * get_arg_val<uint32_t>(t0r);  // (past the tail)
    const uint32_t naf = get_arg_val<uint32_t>(q0), af0 = q0 + 1, nab = get_arg_val<uint32_t>(af0 + naf), ab0 = af0 + naf + 1;
    const uint32_t f0 = ab0 + nab;
    const uint32_t fcount = get_arg_val<uint32_t>(f0), fy0 = get_arg_val<uint32_t>(f0 + 1);
    const uint32_t fstride = get_arg_val<uint32_t>(f0 + 2), fw = get_arg_val<uint32_t>(f0 + 3);
    volatile tt_l1_ptr uint32_t* arrf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(f0 + 4));
    volatile tt_l1_ptr uint32_t* arrb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(f0 + 5));
    uint32_t af = 0, ab = 0, fo = 0;
    const uint32_t NPY = 0;  // (no plain y writes: the fin sub-block is the output)
#ifdef SE_LAT
    // (the latent: my rows of my own block - mine and every other chip's slice of them - to the owner's latent [RB, G LT])
    const uint32_t L0 = f0 + 6, lnr = get_arg_val<uint32_t>(L0), lr0 = get_arg_val<uint32_t>(L0 + 1);
    const uint32_t lnsrc = get_arg_val<uint32_t>(L0 + 2), lzone = get_arg_val<uint32_t>(L0 + 4);
    volatile tt_l1_ptr uint32_t* lctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(L0 + 3));
    const uint32_t LT = get_arg_val<uint32_t>(L0 + 5), lc0 = get_arg_val<uint32_t>(L0 + 6), llw = get_arg_val<uint32_t>(L0 + 7);
    const uint32_t lp = get_arg_val<uint32_t>(L0 + 8), lorow = get_arg_val<uint32_t>(L0 + 9);
    const auto lg = TensorAccessor(l_args, get_arg_val<uint32_t>(L0 + 10), 2048);
#ifdef SE_PREP
    bool lown = true, lin = lnr == 0 || lnsrc == 0;  // (dispatch prep: the compute untilizes my rows' latent)
#else
    bool lown = lnr == 0, lin = lnr == 0 || lnsrc == 0;
#endif
    auto lwrite = [&](uint32_t src, uint32_t q) {  // (rows lr0 .. of a [MT][LWMAX] block, chip q's columns)
        for (uint32_t i = 0; i < lnr; ++i)
            for (uint32_t t = 0; t < llw; ++t)
                noc_async_write(src + ((lr0 + i) * SE_LWMAX + t) * 2048, lg.get_noc_addr((lorow + i) * G_LT + q * LT + lc0 + t), 2048);
    };
#else
    bool lown = true, lin = true;
#endif
#ifdef SE_PREP  // (dispatch prep: per unit the dispatch buffer's row, byte offset in it, bytes a row)
#ifdef SE_LAT
    const uint32_t P0 = f0 + 6 + 11;
#else
    const uint32_t P0 = f0 + 6;
#endif
    const uint32_t pnu = get_arg_val<uint32_t>(P0);
    const auto pg = TensorAccessor(p_args, get_arg_val<uint32_t>(P0 + 1), SE_PREP_ROW);
#else
    constexpr uint32_t pnu = 0;
#endif
    uint32_t pu = 0;
    while (ws < NR || wr < NR || wc < NR || af < naf || ab < nab || fo < fcount || !lown || !lin || pu < pnu || !rout) {
        invalidate_l1_cache();
#ifdef SE_LAT
        if (!lown && cb_pages_available_at_front(SE_LAT, MT * SE_LWMAX)) {
            lwrite(get_read_ptr(SE_LAT), lp);
            lown = true;
        }
        if (!lin && *lctr >= lnsrc) {
#ifdef SE_PREP
            cb_reserve_back(SE_LRDY, 1);  // (the compute's prep may untilize the arrivals)
            cb_push_back(SE_LRDY, 1);
#else
            for (uint32_t sl = 0; sl < lnsrc; ++sl) lwrite(lzone + sl * MT * SE_LWMAX * 2048, sl < lp ? sl : sl + 1);
#endif
            lin = true;
        }
#endif
#ifdef SE_PREP
        if (pu < pnu && cb_pages_available_at_front(SE_PO, SE_PREP_FULL)) {  // a prep unit's 32 rows to the buffer
            const uint32_t src = get_read_ptr(SE_PO), q = P0 + 2 + 3 * pu;
            const uint32_t row = get_arg_val<uint32_t>(q), co = get_arg_val<uint32_t>(q + 1), nb = get_arg_val<uint32_t>(q + 2);
            for (uint32_t i = 0; i < 32; ++i) noc_async_write(src + i * SE_PREP_FULL * 64, pg.get_noc_addr(row + i) + co, nb);
            noc_async_writes_flushed();  // (PO may be refilled once they left L1; the kernel's last barrier acks them)
            cb_pop_front(SE_PO, SE_PREP_FULL);
            ++pu;
        }
#endif
        if (af < naf) {
            const uint32_t thr = get_arg_val<uint32_t>(af0 + af);
            if (thr == 0 || *arrf >= thr) {
                cb_reserve_back(SE_ARRF, SE_P);
                cb_push_back(SE_ARRF, SE_P);
                ++af;
            }
        }
        if (ab < nab) {
            const uint32_t thr = get_arg_val<uint32_t>(ab0 + ab);
            if (thr == 0 || *arrb >= thr) {
                cb_reserve_back(SE_ARRB, SE_P);
                cb_push_back(SE_ARRB, SE_P);
                ++ab;
            }
        }
        if (fo < fcount && cb_pages_available_at_front(SE_OUTF, 1)) {
            cb_wait_front(SE_OUTF, 1);
            noc_async_write(get_read_ptr(SE_OUTF), yg.get_noc_addr(fy0 + (fo / fw) * fstride + fo % fw), 2048);
            noc_async_write_barrier();
            cb_pop_front(SE_OUTF, 1);
            ++fo;
        }
#else
    const uint32_t NPY = NCH * NPD;
    while (ws < NR || wr < NR || wc < NR || p < NPY) {
        invalidate_l1_cache();
#endif
        if (wi < NB) prefetch(0);  // (keep the window full)
        if (ws < NR && (ws < NWS || *sem(WFREE + ws % NWS) >= GY * (nf[ws % NWS] + 1))) {  // my block: down the column
            const uint32_t b = blk(ws), n = bsize(b), kb = b / GY;  // (kb: its index among my blocks)
            woff = 0;  // (its offset in my W-own: my blocks before it)
            for (uint32_t b2 = cy; b2 < b; b2 += GY) woff += bsize(b2);
            if (ws < NB) {  // (first chunk: its reads issued and landed)
                prefetch(ws + 1);
                noc_async_read_barrier_with_trid(trid_of[kb]);
                --pending;
            }
            const uint32_t dst = win + (ws % NWS) * WSLOT * WTB, vw = words + (ws & 3) * 16;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vw) = ws + 1;
#ifdef SE_PROBE_NOW
            noc_async_write_multicast_loopback_src(wown + woff * WTB, get_noc_multicast_addr(cxv, by, cxv, ty, dst), 16, GY);  // (probe)
#else
            noc_async_write_multicast_loopback_src(wown + woff * WTB, get_noc_multicast_addr(cxv, by, cxv, ty, dst), n * WTB, GY);
#endif
            noc_async_writes_flushed();
            noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(cxv, by, cxv, ty, get_semaphore(WV + ws % NWS)), GY);
            noc_async_writes_flushed();
            if (ws >= NWS) ++nf[ws % NWS];
            ++ws;
            while (ws < NR && blk(ws) % GY != cy) ++ws;
        }
        if (wr < NR && *sem(WV + wr % NWS) >= wr + 1) {  // W round wr landed here
            cb_reserve_back(CB_WIN, WSLOT);
            cb_push_back(CB_WIN, WSLOT);
            ++wr;
        }
        if (wc < NR && wc < wr + NWS && cb_pages_reservable_at_back(CB_WIN, WSLOT * (wc - wr + 1))) {  // slot of wc free
            noc_semaphore_inc(get_noc_addr(cxv, get_arg_val<uint32_t>(ys0 + blk(wc) % GY), get_semaphore(WFREE + wc % NWS)), 1);
            ++wc;
        }
        if (p < NPY && cb_pages_available_at_front(CB_OUT, MT * PWDMAX)) {  // down pass p's y (chunk p / NPD)
            cb_wait_front(CB_OUT, MT * PWDMAX);
            const uint32_t m = p / NPD1, pd = p % NPD1, l1 = get_read_ptr(CB_OUT);
            uint32_t c0 = n0d;
            for (uint32_t i = 0; i < pd; ++i) c0 += get_arg_val<uint32_t>(wd0 + i);
            const uint32_t w = get_arg_val<uint32_t>(wd0 + pd);
            for (uint32_t r = 0; r < MT; ++r)
                for (uint32_t t = 0; t < w; ++t)
                    noc_async_write(l1 + (r * PWDMAX + t) * 2048, yg.get_noc_addr((m * CR + cy * MT + r) * NTH + c0 + t), 2048);
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, MT * PWDMAX);
            ++p;
        }
    }
    noc_async_write_barrier();
#ifdef SE_RS
    {  // re-arm my arrival counters (every increment of the call landed: each was waited to its final count)
        uint32_t nf = 0, nb = 0;
        for (uint32_t i = 0; i < naf; ++i) nf = get_arg_val<uint32_t>(af0 + i) > nf ? get_arg_val<uint32_t>(af0 + i) : nf;
        for (uint32_t i = 0; i < nab; ++i) nb = get_arg_val<uint32_t>(ab0 + i) > nb ? get_arg_val<uint32_t>(ab0 + i) : nb;
        if (nf) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrf)), 0u - nf);
        if (nb) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrb)), 0u - nb);
    }
#endif
#ifdef SE_LAT
    if (lnr && lnsrc) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(lctr)), 0u - lnsrc);  // (re-arm)
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
#include "tools/profiler/kernel_profiler.hpp"
#if defined(SE_GLU) && defined(TRISC_PACK)
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if SE_GLU == 2
#include "ckernel_sfpu_situ_glu.h"
#endif
#endif
constexpr uint32_t MT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1), NSG = get_compile_time_arg_val(2);
constexpr uint32_t NPG = get_compile_time_arg_val(3), NCH = get_compile_time_arg_val(4), HT = get_compile_time_arg_val(5);
constexpr uint32_t POMAX = get_compile_time_arg_val(6), KBD = get_compile_time_arg_val(7), NSD = get_compile_time_arg_val(8);
constexpr uint32_t NPD = get_compile_time_arg_val(9), WSLOT = get_compile_time_arg_val(10), PWDMAX = get_compile_time_arg_val(11);
constexpr uint32_t G = get_compile_time_arg_val(12), SPS = get_compile_time_arg_val(13);  // (shards, steps a shard)
constexpr uint32_t GX = get_compile_time_arg_val(14), HA = GX * MT * POMAX;  // (h-all: [column][MT][POMAX])
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
            rmm_init(SE_RW, SE_RX, MT, SE_KC);
            cb_reserve_back(SE_RO, etc * MT);
            for (uint32_t rd = 0; rd < SE_KTO / SE_KC; ++rd) {
                cb_wait_front(SE_RX, SE_KC * MT);
                cb_wait_front(SE_RW, etc * SE_KC);
                PACK((llk_pack_reconfig_l1_acc(rd > 0 ? 1 : 0)));
                tile_regs_acquire();
                for (uint32_t e = 0; e < etc; ++e)
                    for (uint32_t k = 0; k < SE_KC; ++k) rmm(SE_RW, SE_RX, e * SE_KC + k, k * MT, e * MT, MT, SE_KC);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < etc * MT; ++t) pack_tile<true>(t, SE_RO, t);
                tile_regs_release();
                cb_pop_front(SE_RX, SE_KC * MT);
                cb_pop_front(SE_RW, etc * SE_KC);
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_push_back(SE_RO, etc * MT);
            reconfig_data_format(WIN, XIN);  // (back to gate / up's formats)
        }
    }
#endif
#if defined(SE_GLU)
#if SE_GLU == 2
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
#else
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
#endif
    for (uint32_t m = 0; m < NCH; ++m) {
        // gate / up: per pass all of K in DEST, the GLU, h packed once
        {
        DeviceZoneScopedN("SE-GU");
        cb_reserve_back(HOWN, MT * POMAX);
        // shard-major: shard k (own first, then as each lands) through every pass; a pass's fp32 partial over the
        // shards so far goes to ACC (FIFO: pushed per (shard, pass), popped by the next shard's same pass) and comes
        // back into DEST; the last shard applies the GLU and packs h
        for (uint32_t k = 0; k < G; ++k) {
            uint32_t hoff = 0, loff = 0;
            for (uint32_t pg = 0; pg < NPG; ++pg) {
                const uint32_t pw = get_arg_val<uint32_t>(pg), n = MT * pw;
                tile_regs_acquire();
                if (k > 0) {
                    cb_wait_front(ACC, n);
                    reconfig_data_format_srca(WIN, ACC);
                    copy_init(ACC);
                    for (uint32_t t = 0; t < n; ++t) copy_tile(ACC, t, t);
                    cb_pop_front(ACC, n);
                    reconfig_data_format_srca(ACC, WIN);
                }
                matmul_block_init(XIN, WIN, 0, pw, MT, KB);
                for (uint32_t st = 0; st < SPS; ++st) {
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
                    cb_reserve_back(ACC, n);
                    tile_regs_wait();
                    pack_reconfig_data_format(ACC);
                    for (uint32_t t = 0; t < n; ++t) pack_tile(t, ACC);
                    tile_regs_release();
                    cb_push_back(ACC, n);
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
#if defined(SE_GLU)
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if SE_GLU == 2
                for (uint32_t t = 0; t < MT * pw; t += 2)
                    PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_situ_glu, (DST_ACCUM_MODE, 8, sfpu::SituGluConfigKimi), t, t + 1, t, VectorMode::RC)));
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
        cb_push_back(HOWN, MT * POMAX);
#ifdef SE_LAT
        cb_reserve_back(SE_LAT, MT * SE_LWMAX);  // (the latent: the BRISC hands it to the ports, the NCRISC writes mine)
        cb_push_back(SE_LAT, MT * SE_LWMAX);
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
        const uint32_t r0 = NPG + NPD + GX + 1, nops = get_arg_val<uint32_t>(r0), pt0 = r0 + 1 + nops;
        const uint32_t et0 = pt0 + NPD * (1 + MT);
        uint32_t npass = 0, hin = 0;  // (h-all columns waited for: the first pass follows the exchange's rounds)
        for (uint32_t o = 0; o < nops; ++o) {
            const uint32_t op = get_arg_val<uint32_t>(r0 + 1 + o), kind = op & 0xFF, idx = op >> 8;
            if (kind == 0) {
#ifdef SE_PROBE_RSSTAT
                DeviceZoneScopedN("SE-DPASS");
#endif
                const uint32_t pd = idx, pw = get_arg_val<uint32_t>(pt0 + pd * (1 + MT));
                reconfig_data_format(WIN, HALL);
                matmul_block_init(HALL, WIN, 0, pw, MT, POMAX);
                uint32_t c = 0;
                tile_regs_acquire();
                for (uint32_t i = 0; i < NSD; ++i) {
                    cb_wait_front(WIN, WSLOT);
                    for (uint32_t k = 0; k < KBD; ++k) {
                        const uint32_t kt = i * KBD + k;
                        while (kt >= get_arg_val<uint32_t>(NPG + NPD + c + 1)) ++c;
                        if (c >= hin) {
                            hin = c + 1;
                            cb_wait_front(HALL, hin * MT * POMAX);
                        }
                        const uint32_t a0 = c * MT * POMAX + kt - get_arg_val<uint32_t>(NPG + NPD + c);
#ifndef SE_PROBE_NOMATH
                        matmul_block(HALL, WIN, a0, k * pw, 0, false, pw, MT, POMAX);
#endif
                    }
                    cb_pop_front(WIN, WSLOT);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < MT; ++r) {
                    const uint32_t bw = get_arg_val<uint32_t>(pt0 + pd * (1 + MT) + 1 + r);
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
                const uint32_t nu = get_arg_val<uint32_t>(idx);
                uint32_t pu0 = 0;
                UNPACK((pu0 = get_local_cb_interface(SE_PU).fifo_rd_ptr));
                bool lrdy = false;
                if (nu) {
                    reconfig_data_format_srca(SE_PU);  // (the last pass left srcA at the weights' format)
                    pack_reconfig_data_format(SE_PO);
                    pack_untilize_init<SE_PREP_BCT, SE_PREP_FULL>(SE_PU, SE_PO);
                }
                for (uint32_t u = 0; u < nu; ++u) {
                    const uint32_t off = get_arg_val<uint32_t>(idx + 1 + 2 * u), fl = get_arg_val<uint32_t>(idx + 2 + 2 * u);
                    const uint32_t n = fl & 0xFFFF;
#ifdef SE_LRDY
                    if ((fl >> 16) && !lrdy) {  // (a unit of the latent's arrivals: once they all landed)
                        cb_wait_front(SE_LRDY, 1);
                        lrdy = true;
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
#endif
            } else {
                const uint32_t e6 = et0 + 6 * idx, cnt = get_arg_val<uint32_t>(e6), mode = get_arg_val<uint32_t>(e6 + 1);
                const uint32_t sf = get_arg_val<uint32_t>(e6 + 2), sb = get_arg_val<uint32_t>(e6 + 3);
                const uint32_t ho = get_arg_val<uint32_t>(e6 + 4), fbo = get_arg_val<uint32_t>(e6 + 5);
                cb_wait_front(SE_PSYNC, npass);
                if (mode & 1) cb_wait_front(SE_ARRF, (sf + 1) * SE_P);
                if (mode & 2) cb_wait_front(SE_ARRB, (sb + 1) * SE_P);
                const uint32_t a1 = (mode & 1) ? SE_ARRF : SE_ARRB, s1 = ((mode & 1) ? sf : sb) * SE_P;
                const uint32_t ocb = (mode & 4) ? SE_OUTF : SE_FINB;
                pack_reconfig_data_format(ocb);
                for (uint32_t i0 = 0; i0 < cnt; i0 += 8) {
                    const uint32_t n = cnt - i0 < 8 ? cnt - i0 : 8;
                    tile_regs_acquire();
                    reconfig_data_format(SE_HOLD, a1);
                    add_init(SE_HOLD, a1);
                    for (uint32_t j = 0; j < n; ++j) add_tiles(SE_HOLD, a1, ho + i0 + j, s1 + i0 + j, j);
                    if ((mode & 3) == 3) {  // (the second arrival: DEST -> srcA at fp32 width, + the bwd tile)
                        reconfig_data_format(SE_HOLD, SE_ARRB);
                        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(SE_ARRB);
                        for (uint32_t j = 0; j < n; ++j)
                            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(SE_ARRB, sb * SE_P + i0 + j, j);
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
        }
#elif !defined(SE_NODOWN)
        DeviceZoneScopedN("SE-DOWN");
        // down: per pass all of K in DEST over h-all and the streamed W_down blocks, y packed once
        cb_wait_front(HALL, HA);
        for (uint32_t pd = 0; pd < NPD; ++pd) {
            const uint32_t pw = get_arg_val<uint32_t>(NPG + pd);
            matmul_block_init(HALL, WIN, 0, pw, MT, POMAX);  // (h rows strided POMAX within a column's block)
            uint32_t c = 0;  // (the column block of K tile k: k in [h0[c], h0[c + 1]))
            tile_regs_acquire();
            for (uint32_t i = 0; i < NSD; ++i) {
                cb_wait_front(WIN, WSLOT);
                for (uint32_t k = 0; k < KBD; ++k) {
                    const uint32_t kt = i * KBD + k;
                    while (kt >= get_arg_val<uint32_t>(NPG + NPD + c + 1)) ++c;
                    const uint32_t a0 = c * MT * POMAX + kt - get_arg_val<uint32_t>(NPG + NPD + c);
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
    """(timing only) SE_PROBE=NOX,NOW: skip the x / W data multicasts (semaphores kept); NOMATH: no matmuls; NOWL: no W
    DRAM reads (16 B a block) - the results are wrong."""
    import os

    return "".join(f"#define SE_PROBE_{p}\n" for p in os.environ.get("SE_PROBE", "").split(",") if p)


def _split(n, parts, unit):
    """n tiles into `parts` widths, multiples of unit, as even as possible (each >= unit)."""
    assert n % unit == 0 and n // unit >= parts, (n, parts, unit)
    base, rem = divmod(n // unit, parts)
    return [(base + (1 if i < rem else 0)) * unit for i in range(parts)]


def _subsets(nf, GX):
    """Two owner-subset sizes (families 0 / 1) dividing nf, together covering as many of GX columns as possible."""
    divs = [d for d in range(1, GX + 1) if nf % d == 0]
    best = max(((a, b) for a in divs for b in divs), key=lambda ab: (min(ab[0] + ab[1], GX), -abs(ab[0] - ab[1])))
    return best


class SeEngine3:
    """The shared expert a chip, row-chunk-major, weights resident: x [rows, H] (the gathered x) -> y [rows, H] (this
    chip's partial over its I slice). glu "silu" / "situ"."""

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
        KBD=8,
        fidelity=ttnn.MathFidelity.HiFi2,
        w_dtype=ttnn.bfloat8_b,
        shards=1,
        down=True,
        rs=None,
        latent=None,
        prep=False,
        router=None,
    ):
        """prep: the dispatch prep's owned rows (pieces(prep=...)): x own's rows of the chip's owned tokens (or, with a
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
        assert rows % (32 * self.CR) == 0
        self.NCH = rows // 32 // self.CR
        self.KT, self.HT, self.NTH = H // 32, I // 32, H // 32
        self.KB = KB or 16  # (fewer, larger rounds: K3 403 -> 341 us at 640 rows; GLM-Flash neutral)
        self.NSG = self.KT // self.KB
        assert self.KT % self.KB == 0 and self.NSG % 2 == 0
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
        assert min(pairs) >= 1
        self.po, self.h0 = pairs, [sum(pairs[:c]) for c in range(GX)]
        self.POMAX = max(pairs)
        pwmax = (8 // MT) // 2 * 2
        self.NPG = max(-(-2 * p // pwmax) for p in pairs)
        self.wg = [_split(2 * p, self.NPG, 2) for p in pairs]
        self.NGU, self.LT = self.NPG, latent  # (gate / up passes; then the latent's)
        if latent:
            lw = column_split(latent, GX, 1)
            assert min(lw) >= 1
            self.lw, self.l0 = lw, [sum(lw[:c]) for c in range(GX)]
            self.LWMAX = max(lw)
            npl = -(-self.LWMAX // (8 // MT))
            self.wg = [wg + _split(lw[c], npl, 1) for c, wg in enumerate(self.wg)]
            self.NPG += npl
        # down: the column's output tiles, NPD passes of widths <= 8 / MT, K steps of KBD
        pnd = column_split(self.NTH, GX, 1)
        self.n0d = [sum(pnd[:c]) for c in range(GX)]
        self.PWDMAX = 8 // MT
        self.down = down
        self.NPD = max(-(-p // self.PWDMAX) for p in pnd) if down else 0
        self.wd = [_split(p, self.NPD, 1) if down else [] for p in pnd]
        self.rs = rs
        if rs is not None:  # (the column's down passes: the plan's, widths the same on every chip)
            assert down and rs.pwmax == self.PWDMAX and rs.MT == MT and rs.GX == GX and rs.GY == GY and self.NCH == 1
            self.wd = [[pw for _, pw, _ in rs.column_passes(0, c)] for c in range(GX)]
            self.NPD = len(self.wd[0])
            assert all(len(wd) == self.NPD for wd in self.wd)
        self.KBD = KBD
        assert self.HT % KBD == 0
        self.NSD = self.HT // KBD if down else 0
        self.NB = self.NPG * self.NSG + self.NPD * self.NSD
        self.WSLOT = max(self.KB * max(sum(self.wg, [])), KBD * max(sum(self.wd, []), default=0))  # (tiles)
        self.fid, self.w_dtype = fidelity, w_dtype
        self.wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}[w_dtype]

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
        nown = max((nf // self.S0 if c in self.fam0 else 0) + (nf // self.S1 if c in self.fam1 else 0) for c in cols)
        self.xown_t = self.NCH * nown * MT * self.KB
        # dispatch prep: a unit = one row tile's run of x own (a K step: KB tiles) or of a latent block (LWMAX)
        self.prep = prep
        self.PFULL = self.LWMAX if latent else self.KB
        self.PBCT = max(d for d in range(1, 9) if self.PFULL % d == 0)
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        off, o = {}, 0
        rsr = self._rs_regions() if rs is not None else []
        # (the reduce-scatter's local buffers - HOLD, FINB: this core's compute and, after its hand-off, its port read
        # them - overlay XIN + ACC, which are dead once this core's gate / up is done: no row peer multicasts into XIN
        # after the last round it consumed; only its own compute touches ACC)
        ovl = dict(rsr[:2]) if rsr else {}
        rsr = rsr[2:]
        for k, v in (
            [
                ("xown", self.xown_t * 2048),
                ("wown", self.wown_t * self.wtb),
                ("win", NWS * self.WSLOT * self.wtb),
                ("hown", MT * self.POMAX * 2048),
                ("hall", min(self.NCH, 2) * GX * MT * self.POMAX * 2048 if down else 0),
                ("xin", 2 * MT * self.KB * 2048),
                ("acc", MT * (max(sum(c) for c in self.wg) + max(sum(self.wg, []))) * 4096),
                ("out", NOUT * MT * self.PWDMAX * 2048 if down and rs is None else 0),
                ("words", 256),
            ]
            + rsr
            + ([("lat", MT * self.LWMAX * 2048), ("latz", (shards - 1) * MT * self.LWMAX * 2048)] if latent else [])
            + ([("po", 2 * self.PFULL * 2048)] + ([("lrdy", 64)] if latent else []) if prep else [])
        ):
            off[k], o = o, al(o + v)
        if ovl:
            room = off["out"] - off["xin"]
            if al(ovl["hold"]) + ovl["finb"] <= room:
                off["hold"], off["finb"] = off["xin"], off["xin"] + al(ovl["hold"])
            else:  # (no room: their own regions)
                for k, v in ovl.items():
                    off[k], o = o, al(o + v)
        # the router logits on the engine (router: dict(ET, KTO)): x rows, W_r^T expert tiles, the fp32 out - in the
        # arena's tail past the words (the reduce-scatter's zones, the latent's, the prep's: all written only after
        # gate / up, the router's phase is over before it), else their own region
        self.rtr = router
        if router is not None:
            self.ETC = -(-router["ET"] // GX)
            KTO = router["KTO"]
            self.KC = max(d for d in range(1, 9) if KTO % d == 0)  # (K chunk: x / W_r^T rounds)
            rsz = [("rx", self.KC * MT * 2048), ("rw", self.ETC * self.KC * 2048), ("ro", self.ETC * MT * 4096)]
            need = sum(al(v) for _, v in rsz)
            t0 = off[rsr[0][0]] if rsr else o
            if o - t0 < need:
                hall = off["xin"] - off["hall"]
                t0 = off["hall"] if down and hall >= need else o
                if t0 == o:
                    o = o + need
            for k, v in rsz:
                off[k], t0 = t0, al(t0 + v)
        self.off, self.arena_bytes = off, o
        if o > ARENA_MAX:
            sizes = {
                k: (off[k2] if i + 1 < len(off) else o) - off[k]
                for i, (k, k2) in enumerate(zip(off, list(off)[1:] + [None]))
            }
            raise ValueError(
                f"se_engine3: arena {o // 1024} KB > {ARENA_MAX // 1024} KB of L1 (KB {self.KB}): "
                + ", ".join(f"{k} {v // 1024}" for k, v in sizes.items())
            )
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
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, (1, 64), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        self.alloc_arena()
        self.groups = {}
        for c in range(GX):
            self.groups.setdefault((tuple(self.wg[c]), tuple(self.wd[c])), []).append(c)

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
        return list(self.rs_sizes.items())

    def alloc_arena(self, extra=()):
        """The arena (call scratch: x, the weights, every CB) as an L1 tensor; base its address (the programs bake it).
        extra: more cores the same range is carved on (L1 addresses are allocated in lockstep over the grid, so the
        mover column's other buffers live in it too)."""
        o = self.arena_bytes
        crs = self.crs
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
        tail = [len(ports), int(xloaded)] + [v for p_ in ports for v in p_] + [XLD]
        mesh, GX, GY, cx0, MT = self.mesh, self.GX, self.GY, self.cx0, self.MT
        b, off = self.base, self.off
        virt = lambda cx, cy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0 + cx, cy))  # noqa: E731
        allc = self.crs
        cbs = []

        def cb(i, key, size, dt, page):
            d = ttnn.cb_descriptor_from_sharded_tensor(
                i, self.arena, address_offset=off[key], total_size=size, core_ranges=allc
            )
            d.format_descriptors = [ttnn.CBFormatDescriptor(i, dt, page)]
            cbs.append(d)

        cb(XIN, "xin", 2 * MT * self.KB * 2048, ttnn.bfloat16, 2048)
        cb(WIN, "win", NWS * self.WSLOT * self.wtb, self.w_dtype, self.wtb)
        cb(HOWN, "hown", MT * self.POMAX * 2048, ttnn.bfloat16, 2048)
        if self.down:
            cb(HALL, "hall", min(self.NCH, 2) * GX * MT * self.POMAX * 2048, ttnn.bfloat16, 2048)
        if self.down and self.rs is None:
            cb(OUT, "out", NOUT * MT * self.PWDMAX * 2048, ttnn.bfloat16, 2048)
        if self.rs is not None:
            z = self.rs_sizes
            cb(RS_CB["hold"], "hold", z["hold"], ttnn.float32, 4096)
            for k in ("finb", "arrf", "arrb", "outf"):
                cb(RS_CB[k], k, z[k], ttnn.bfloat16, 2048)
            for k in ("finc", "psync"):
                cb(RS_CB[k], k, z[k], ttnn.bfloat16, 16)
        if self.LT:
            cb(LAT_CB, "lat", MT * self.LWMAX * 2048, ttnn.bfloat16, 2048)
        if prep is not None:
            assert self.prep and self.rs is not None, "dispatch prep: built with prep=True, after the reduce-scatter"
            pk = "lat" if self.LT else "xown"
            pend = (
                off["latz"] + (self.shards - 1) * MT * self.LWMAX * 2048
                if self.LT
                else off["xown"] + self.xown_t * 2048
            )
            cb(PREP_CB["pu"], pk, pend - off[pk], ttnn.bfloat16, 2048)
            cb(PREP_CB["po"], "po", 2 * self.PFULL * 2048, ttnn.bfloat16, 2048)
            if self.LT:
                cb(PREP_CB["lrdy"], "lrdy", 64, ttnn.bfloat16, 16)
        if rtr is not None:
            KTO = self.rtr["KTO"]
            cb(RTR_CB["rx"], "rx", self.KC * MT * 2048, ttnn.bfloat16, 2048)
            cb(RTR_CB["rw"], "rw", self.ETC * self.KC * 2048, ttnn.bfloat16, 2048)
            cb(RTR_CB["ro"], "ro", self.ETC * MT * 4096, ttnn.float32, 4096)
        cb(ACC, "acc", MT * (max(sum(c) for c in self.wg) + max(sum(self.wg, []))) * 4096, ttnn.float32, 4096)
        xa_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        b_rt, n_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in range(GX):
            i0 = self.fam0.index(c) if c in self.fam0 else GX
            i1 = self.fam1.index(c) if c in self.fam1 else GX
            for cy in range(GY):
                b_rt[cx0 + c][cy] = (
                    [
                        c,
                        cy,
                        int(x.buffer_address()),
                        x_ct,
                        x_k0,
                        b + off["xown"],
                        b + off["xin"],
                        b + off["hown"],
                        b + off["hall"],
                        self.po[c],
                        self.h0[c],
                        i0,
                        i1,
                        virt(0, cy).x,
                        virt(GX - 1, cy).x,
                        virt(c, cy).y,
                    ]
                    + [virt(cc, cy).x for cc in range(GX)]
                    + sor
                    + tail
                    + [int(y.buffer_address())]
                    + self._rs_brisc(rs_pos, rs_port, rs_mbox, c, cy)
                    + self._la_brisc(rs_pos, rs_port, la, c, cy)
                )
                n_rt[cx0 + c][cy] = (
                    [
                        cy,
                        virt(c, cy).x,
                        virt(c, 0).y,
                        virt(c, GY - 1).y,
                        int(w.buffer_address()),
                        ((c * GY + cy) // self.nbank) * self.wown_t * self.wtb,
                        b + off["wown"],
                        b + off["win"],
                        b + off["words"],
                        int(y.buffer_address()),
                        self.n0d[c],
                    ]
                    + (self._rtr_args(rtr, c, cy) if rtr is not None else [])
                    + self.wg[c]
                    + self.wd[c]
                    + [virt(c, yy).y for yy in range(GY)]
                    + [(c * GY + cy) % self.nbank]
                    + sor
                    + tail
                    + self._rs_ncrisc(rs_pos, rs_arr, c, cy)
                    + self._la_ncrisc(rs_pos, la, c, cy)
                    + (self._prep_ncrisc(rs_pos, prep, c, cy) if prep is not None else [])
                )
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
        if self.rs is not None:
            nd += "#define SE_RS\n" + "".join(f"#define SE_{k.upper()} {v}\n" for k, v in RS_CB.items())
            nd += f"#define SE_P {self.rs.P}\n"
        if self.LT:
            assert self.rs is not None, "the latent's all-to-all rides the reduce-scatter's ports and args"
            nd += f"#define SE_LAT {LAT_CB}\n#define SE_NGU {self.NGU}\n#define SE_LWMAX {self.LWMAX}\n"
            nd += f"#define G_LT {len(so) * self.LT}\n"
        if rtr is not None:
            nct_len = len(nct + w_ct + y_ct) + (
                len(ttnn.TensorAccessorArgs(la["out"]).get_compile_time_args()) if self.LT else 0
            )
            nct_len += len(ttnn.TensorAccessorArgs(prep["xbuf"]).get_compile_time_args()) if prep is not None else 0
            nd += "#define SE_RTR\n" + "".join(f"#define SE_{k.upper()} {v}\n" for k, v in RTR_CB.items())
            nd += f"#define SE_KTO {self.rtr['KTO']}\n#define SE_ETC {self.ETC}\n#define SE_ET {self.rtr['ET']}\n"
            nd += f"#define SE_GX {GX}\n#define RXR_ID {RXR}\n#define RWR_ID {RWR}\n"
            if rtr.get("gate"):
                nd += f"#define SE_RTR_GATE\n#define RGATE_ID {RGATE}\n"

            nd += f"#define RXA_ID {RXA}\n#define RWA_ID {RWA}\n#define SE_KC {self.KC}\n"
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
            compile_time_args=bct + xa_ct + ([] if self.down else y_ct),
            runtime_args=b_rt,
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
                config=dm("RISCV_1", ttnn.NOC.NOC_1),
            ),
        ]
        glu_code = {"silu": 1, "situ": 2}[self.glu]
        for (wg, wd), cols in self.groups.items():
            crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(cx0 + c, 0), ttnn.CoreCoord(cx0 + c, GY - 1)) for c in cols]
            )
            comp = ttnn.ComputeConfigDescriptor()
            comp.math_fidelity = self.fid
            comp.fp32_dest_acc_en = True
            comp.dst_full_sync_en = True
            comp.math_approx_mode = False
            modes = [ttnn.UnpackToDestMode.Default] * 64  # (NUM_CIRCULAR_BUFFERS)
            modes[ACC] = ttnn.UnpackToDestMode.UnpackToDestFp32
            comp.unpack_to_dest_mode = modes
            c_rt = ttnn.RuntimeArgs()
            for c in cols:
                for cy in range(GY):
                    pre = list(wg) + list(wd) + list(self.h0) + [self.HT]  # (+ the h column offsets)
                    ra = self.rs.compute_args(rs_pos, c, cy) if self.rs is not None else []
                    if prep is not None:
                        ra = self._prep_compute(ra, len(pre), rs_pos, prep, c, cy)
                    c_rt[cx0 + c][cy] = pre + ra
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"#define SE_GLU {glu_code}\n" + nd + _probes() + _COMPUTE,  # noqa
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
                    ],
                    runtime_args=c_rt,
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
        return [len(hs), mbox + 16 * k] + [v for _, d in hs for v in port(d, rs.link(k))]

    def la_rows(self, cy, b):
        """(first local row, rows) of core row cy's MT row tiles in owner block b (RB = CR / G row tiles a block)."""
        RB = self.CR // self.shards
        rs = [r for r in range(self.MT) if (cy * self.MT + r) // RB == b]
        return (rs[0], len(rs)) if rs else (0, 0)

    def la_dir(self, p, b):
        """The direction chip p sends block b's latent (shortest; ring ties by p's parity)."""
        G = self.shards
        if not self.rs.kind.startswith("ring"):
            return "fwd" if b > p else "bwd"
        df = (b - p) % G
        return "fwd" if df < G - df or (df == G - df and p % 2 == 0) else "bwd"

    def _la_brisc(self, p, port, la, cx, cy):
        """The latent hand-offs: [count, my port word] + per destination block with rows of mine its port's NoC."""
        if not self.LT:
            return []
        k = cy * self.GX + cx
        dst = [b for b in range(self.shards) if b != p and self.la_rows(cy, b)[1]]
        if "NOA2A" in _probes():  # (timing / hang probe: no all-to-all)
            dst = []
        return [len(dst), la["word"] + 16 * k] + [v for b in dst for v in port(self.la_dir(p, b), k % self.rs.nv)]

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
            + [w for r in range(self.MT) for w in rtr["dests"][cy * self.MT + r]]
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
        block (mine, then each source chip's slice), for the chip's owned rows (block p)."""
        MT, KB, G = self.MT, self.KB, self.shards
        RBo = self.rows // 32 // G  # (owned row tiles)
        row0 = prep["me"] * RBo * 32
        out = []
        if self.LT:
            r0, nr = self.la_rows(cy, p)
            lz = self.off["latz"] - self.off["lat"]
            lorow = cy * MT + r0 - p * (self.CR // G)
            for i in range(nr):
                rr, n = (r0 + i) * self.LWMAX * 2048, self.lw[cx]
                out.append((rr, n, 0, row0 + (lorow + i) * 32, (p * self.LT + self.l0[cx]) * 64))
                for sl in range(G - 1):
                    q = sl if sl < p else sl + 1
                    out.append(
                        (
                            lz + sl * MT * self.LWMAX * 2048 + rr,
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
        for R in range(p * RBo, (p + 1) * RBo):
            m, lr = divmod(R, self.CR)
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
                out.append((((m * nown + slot) * MT + r) * KB * 2048, KB, 0, row0 + (R - p * RBo) * 32, j * KB * 64))
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
        return [nops + 1] + ops + rest + [len(us)] + [v for o, n, lat, _, _ in us for v in (o, n | (lat << 16))]

    def _prep_ncrisc(self, p, prep, cx, cy):
        us = self.prep_units(p, prep, cx, cy)
        return [len(us), int(prep["xbuf"].buffer_address())] + [v for _, n, _, row, co in us for v in (row, co, n * 64)]

    def _rs_ncrisc(self, p, arr, cx, cy):
        """The NCRISC's arrival credits (per direction, slot order: thresholds) and the fin sub-block's y map."""
        if self.rs is None:
            return []
        rs = self.rs
        a = rs.arrivals(p, cx, cy)
        y0, stride, w = rs.fin_map(p, cx, cy)
        return (
            [len(a["fwd"])]
            + [t for _, _, t in a["fwd"]]
            + [len(a["bwd"])]
            + [t for _, _, t in a["bwd"]]
            + [rs.count(cx, cy, p), y0, stride, w, arr[0], arr[1]]
        )

    def attach_mcast(self, prog):
        """The kernel_lib families (x slots 0 / 1, h; the BRISC's NOC0) attached to prog's BRISC kernel (after prog.semaphores is set:
        attachment allocates its semaphore ids past ours)."""
        GX, GY, cx0 = self.GX, self.GY, self.cx0
        rect = self.crs
        cols = lambda cs: ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(cx0 + c, 0), ttnn.CoreCoord(cx0 + c, GY - 1)) for c in cs]
        )  # noqa: E731
        fams = [
            ("x0", ttnn.Mcast1DRotatingSenderConfig(sender_grid=cols(self.fam0))),
            ("x1", ttnn.Mcast1DRotatingSenderConfig(sender_grid=cols(self.fam1))),
            ("h", ttnn.Mcast1DRotatingSenderConfig()),
        ][: 3 if self.down else 2]
        for name, cfg in fams:
            mc = ttnn.Mcast1D(
                self.mesh, rect, ttnn.Mcast1DShape.PerRow, cfg, ttnn.McastConfig(handshake=True, noc=ttnn.NOC.NOC_0)
            )
            mc.attach(prog, name, [self.brisc])

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
