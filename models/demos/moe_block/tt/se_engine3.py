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
HCNT, HFREE = 1 + 2 * NWS, 3 + 2 * NWS  # (unused: the h exchange is the kernel_lib family again)
NSEM_OWN = 1 + 2 * NWS
XIN, WIN, HOWN, HALL, OUT = 0, 1, 2, 3, 4  # CB ids
NOUT = 2  # (down passes OUT holds)

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
constexpr uint32_t CB_XIN = 0, CB_HOWN = 2, CB_HALL = 3;
constexpr uint32_t NF = NSG / 2, NOWN0 = NF / S0, NOWN1 = NF / S1;  // (steps a family a pass; per owner)
// x position j (the j-th K step this chip processes: its own shard's first, then each shard as it lands): family j % 2,
// owner = the family's subset[(j / 2) % S], its local index (j / 2) / S
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
    constexpr auto mx0 = McastArgs<get_named_compile_time_arg_val("x0_ct_offset"), get_named_compile_time_arg_val("x0_rt_offset")>();
    constexpr auto mx1 = McastArgs<get_named_compile_time_arg_val("x1_ct_offset"), get_named_compile_time_arg_val("x1_rt_offset")>();
    constexpr auto mh = McastArgs<get_named_compile_time_arg_val("h_ct_offset"), get_named_compile_time_arg_val("h_rt_offset")>();
    Noc noc;
    auto sh = mh.optional_sender(noc);
    auto rh = mh.optional_receiver(noc);
    auto s0 = mx0.optional_sender(noc);
    auto r0 = mx0.optional_receiver(noc);
    auto s1 = mx1.optional_sender(noc);
    auto r1 = mx1.optional_receiver(noc);
    volatile tt_l1_ptr uint32_t* xready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(XREADY));
    for (uint32_t m = 0; m < NCH; ++m) {
        {
        DeviceZoneScopedN("SE-XROUNDS");
        for (uint32_t pg = 0; pg < NPG; ++pg)
            for (uint32_t j = 0; j < NSG; ++j) {
                const uint32_t round = (m * NPG + pg) * NF + j / 2;  // (the family's)
                cb_reserve_back(CB_XIN, MT * KB);
                const uint32_t dst = xin + (j & 1) * MT * KB * 2048;
                if (mine(j)) {  // (position j: the shard of rank j / sps must be in - the own shard always is)
                    if (j >= sps) noc_semaphore_wait_min(xready, m * (g - 1) + j / sps);
                    if (j & 1) s1->send(own_addr(m, j), dst, MT * KB * 2048); else s0->send(own_addr(m, j), dst, MT * KB * 2048);
                } else {
                    if (j & 1) r1->receive(round); else r0->receive(round);
                }
                cb_push_back(CB_XIN, MT * KB);
            }
        }
        DeviceZoneScopedN("SE-HX");
        // h: every core's rows into every row core's h-all slot (m % 2) at its h0 - kernel_lib rotating rounds
        // (row tile, column); the receivers' readiness ack is the slot's credit
        cb_wait_front(CB_HOWN, MT * POMAX);
        cb_reserve_back(CB_HALL, MT * HT);
        const uint32_t hs = hall + (m & 1) * MT * HT * 2048;
        for (uint32_t r = 0; r < MT; ++r)
            for (uint32_t c = 0; c < GX; ++c) {
                const uint32_t round = (m * MT + r) * GX + c;
                if (c == cx) sh->send(hown + r * POMAX * 2048, hs + (r * HT + h0) * 2048, po * 2048);
                else rh->receive(round);
            }
        cb_push_back(CB_HALL, MT * HT);
        cb_pop_front(CB_HOWN, MT * POMAX);
    }
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
constexpr uint32_t CB_WIN = 1, CB_OUT = 4, NB = NPG * NSG + NPD * NSD;  // (W blocks a chunk)
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
    const uint32_t wg0 = a, wd0 = wg0 + NPG, ys0 = wd0 + NPD;  // pass widths (gate / up, down); the column's NoC y
    const uint32_t wbank = get_arg_val<uint32_t>(ys0 + GY);  // (the DRAM bank holding my W-own region)
    const uint32_t sps = get_arg_val<uint32_t>(ys0 + GY + 1), ord0 = ys0 + GY + 3;  // (steps a shard; the shard order)
    // round w's block (natural index): a gate / up pass's steps in this chip's processing order, then down's
    auto blk = [&](uint32_t w) {
        const uint32_t c = w % NB;
        if (c >= NPG * NSG) return c;
        const uint32_t j = c % NSG;
        return (c / NSG) * NSG + get_arg_val<uint32_t>(ord0 + j / sps) * sps + j % sps;
    };
    auto bsize = [&](uint32_t b) {  // (tiles)
        return b < NPG * NSG ? KB * get_arg_val<uint32_t>(wg0 + b / NSG) : KBD * get_arg_val<uint32_t>(wd0 + (b - NPG * NSG) / NSD);
    };
    const auto yg = TensorAccessor(y_args, y_addr, 2048);
    // my blocks from DRAM, once: my W-own is one contiguous region in one bank (per-bank layout, wbase its offset),
    // read block by block in large bursts, every read issued now - the k-th block's under transaction id
    // min(k, 14) + 1; a block is sent once its reads landed (the column streams while the weights are still arriving)
    {
        DeviceZoneScopedN("SE-WISSUE");
        uint32_t off = 0, k = 0;
        for (uint32_t b = cy; b < NB; b += GY, ++k) {
            const uint32_t n = bsize(b);
            noc_async_read_set_trid(k < 14 ? k + 1 : 15);
            noc_async_read(get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + off * WTB), wown + off * WTB, n * WTB);
            off += n;
        }
    }
    const uint32_t NR = NCH * NB;  // (W rounds a call)
    uint32_t ws = 0, woff = 0, wr = 0, wc = NWS, p = 0, pm = 0;  // (send, my offset, pushed, credited; y pass / chunk)
    uint32_t nf[NWS] = {};
    while (ws < NR && blk(ws) % GY != cy) ++ws;  // (skip rounds I do not hold)
    const uint32_t NPY = NCH * NPD;
    while (ws < NR || wr < NR || wc < NR || p < NPY) {
        invalidate_l1_cache();
        if (ws < NR && (ws < NWS || *sem(WFREE + ws % NWS) >= GY * (nf[ws % NWS] + 1))) {  // my block: down the column
            const uint32_t b = blk(ws), n = bsize(b), kb = b / GY;  // (kb: its index among my blocks)
            woff = 0;  // (its offset in my W-own: my blocks before it)
            for (uint32_t b2 = cy; b2 < b; b2 += GY) woff += bsize(b2);
            if (ws < NB) noc_async_read_barrier_with_trid(kb < 14 ? kb + 1 : 15);  // (first chunk: its reads landed)
            const uint32_t dst = win + (ws % NWS) * WSLOT * WTB, vw = words + (ws & 3) * 16;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vw) = ws + 1;
            noc_async_write_multicast_loopback_src(wown + woff * WTB, get_noc_multicast_addr(cxv, by, cxv, ty, dst), n * WTB, GY);
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
            const uint32_t m = p / NPD, pd = p % NPD, l1 = get_read_ptr(CB_OUT);
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
    noc_async_atomic_barrier();
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
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
constexpr uint32_t XIN = 0, WIN = 1, HOWN = 2, HALL = 3, OUT = 4;
// (runtime args: the column's gate / up pass widths [NPG], then down pass widths [NPD])
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(XIN, WIN, HOWN);
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
        uint32_t hoff = 0;
        for (uint32_t pg = 0; pg < NPG; ++pg) {
            const uint32_t pw = get_arg_val<uint32_t>(pg);
            matmul_block_init(XIN, WIN, 0, pw, MT, KB);
            tile_regs_acquire();
            for (uint32_t j = 0; j < NSG; ++j) {
                cb_wait_front(XIN, MT * KB);
                cb_wait_front(WIN, WSLOT);
                for (uint32_t k = 0; k < KB; ++k) matmul_block(XIN, WIN, k, k * pw, 0, false, pw, MT, KB);
                cb_pop_front(XIN, MT * KB);
                cb_pop_front(WIN, WSLOT);
            }
            tile_regs_commit();
            tile_regs_wait();
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
        cb_push_back(HOWN, MT * POMAX);
        }
        DeviceZoneScopedN("SE-DOWN");
        // down: per pass all of K in DEST over h-all and the streamed W_down blocks, y packed once
        cb_wait_front(HALL, MT * HT);
        for (uint32_t pd = 0; pd < NPD; ++pd) {
            const uint32_t pw = get_arg_val<uint32_t>(NPG + pd);
            matmul_block_init(HALL, WIN, 0, pw, MT, HT);
            tile_regs_acquire();
            for (uint32_t i = 0; i < NSD; ++i) {
                cb_wait_front(WIN, WSLOT);
                for (uint32_t k = 0; k < KBD; ++k) matmul_block(HALL, WIN, i * KBD + k, k * pw, 0, false, pw, MT, HT);
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
        cb_pop_front(HALL, MT * HT);
    }
}
"""


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
    ):
        self.mesh = mesh_device
        self.rows, self.H, self.I, self.glu = rows, H, I, glu
        self.cx0, self.GX, self.GY, self.MT = cx0, GX, GY, MT
        self.CR = GY * MT
        assert rows % (32 * self.CR) == 0
        self.NCH = rows // 32 // self.CR
        self.KT, self.HT, self.NTH = H // 32, I // 32, H // 32
        self.KB = KB or 16  # (fewer, larger rounds: K3 403 -> 341 us at 640 rows; GLM-Flash neutral)
        self.NSG = self.KT // self.KB
        assert self.KT % self.KB == 0 and self.NSG % 2 == 0
        nf = self.NSG // 2
        self.S0, self.S1 = _subsets(nf, GX)
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
        # down: the column's output tiles, NPD passes of widths <= 8 / MT, K steps of KBD
        pnd = column_split(self.NTH, GX, 1)
        self.n0d = [sum(pnd[:c]) for c in range(GX)]
        self.PWDMAX = 8 // MT
        self.NPD = max(-(-p // self.PWDMAX) for p in pnd)
        self.wd = [_split(p, self.NPD, 1) for p in pnd]
        self.KBD = KBD
        assert self.HT % KBD == 0
        self.NSD = self.HT // KBD
        self.NB = self.NPG * self.NSG + self.NPD * self.NSD
        self.WSLOT = max(self.KB * max(sum(self.wg, [])), KBD * max(sum(self.wd, [])))  # (tiles)
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
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        off, o = {}, 0
        for k, v in [
            ("xown", self.xown_t * 2048),
            ("wown", self.wown_t * self.wtb),
            ("xin", 2 * MT * self.KB * 2048),
            ("win", NWS * self.WSLOT * self.wtb),
            ("hown", MT * self.POMAX * 2048),
            ("hall", 2 * MT * self.HT * 2048),
            ("out", NOUT * MT * self.PWDMAX * 2048),
            ("words", 256),
        ]:
            off[k], o = o, al(o + v)
        self.off, self.arena_bytes = off, o
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx0, 0), ttnn.CoreCoord(cx0 + GX - 1, GY - 1))])
        self.crs = cores
        shard = ttnn.ShardSpec(cores, [1, o // 4], ttnn.ShardOrientation.ROW_MAJOR)
        self.arena = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, GX * GY, o // 4]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
        )
        self.base = int(self.arena.buffer_address())
        self.groups = {}
        for c in range(GX):
            self.groups.setdefault((tuple(self.wg[c]), tuple(self.wd[c])), []).append(c)

    def dram_bytes(self):
        w = (self.KT * 2 * self.HT + self.HT * self.NTH) * self.wtb
        return dict(w=w, x=self.rows * self.H * 2, y=self.rows * self.H * 2)

    def upload(self, w_gu, w_down):
        """Per device (row-major mesh order): w_gu [H, 2 I] (glu_weight tile pairs), w_down [I, H] -> one bf8 tensor
        a chip, per-bank: core (c, cy)'s W-own (its column's blocks b = cy, cy + GY, .. of a chunk's stream order:
        gate / up [KB x pass width] pass-major, then down [KBD x pass width]) one contiguous region, region c GY + cy in
        bank (c GY + cy) % banks (flat_expert._bank_sharded) - read in large bursts, every bank equally loaded."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import _bank_sharded

        KB, KBD, NSG, GX, GY = self.KB, self.KBD, self.NSG, self.GX, self.GY
        devs = []
        for wgt, wdt in zip(w_gu, w_down):
            regions = []
            for c in range(GX):
                blocks = []
                g0 = 2 * self.h0[c]
                for pg, pw in enumerate(self.wg[c]):
                    c0 = g0 + sum(self.wg[c][:pg])
                    for j in range(NSG):
                        blk = wgt[j * KB * 32 : (j + 1) * KB * 32, c0 * 32 : (c0 + pw) * 32]
                        blocks.append(blk.reshape(KB, 32, pw, 32).permute(0, 2, 1, 3).reshape(-1, 32))
                for pd, pw in enumerate(self.wd[c]):
                    c0 = self.n0d[c] + sum(self.wd[c][:pd])
                    for i in range(self.NSD):
                        blk = wdt[i * KBD * 32 : (i + 1) * KBD * 32, c0 * 32 : (c0 + pw) * 32]
                        blocks.append(blk.reshape(KBD, 32, pw, 32).permute(0, 2, 1, 3).reshape(-1, 32))
                for cy in range(GY):
                    reg = torch.cat(blocks[cy::GY])
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

    def pieces(self, x, w, y, *, x_k0=0, x_ct=None, shard_order=None, extra_sem_cores=()):
        """(CBs, kernels, semaphores) of the engine on one chip (the kernel_lib families attached by the caller via
        attach_mcast). x: DRAM [rows, x_ct tiles] (columns x_k0 .. of the full x; None: all; 0: none). shard_order: this
        chip's K shards in processing order (own first, then by arrival; XREADY = m (G - 1) + rank once chunk m's shard
        of that rank is in); None: one shard (all of K, already there)."""
        x_ct, x_k0 = int(self.KT if x_ct is None else x_ct), int(x_k0)
        so = [int(q) for q in shard_order] if shard_order is not None else [0]
        assert self.NSG % len(so) == 0
        sor = [self.NSG // len(so), len(so)] + so  # (steps a shard, shards, the order)
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
        cb(HALL, "hall", 2 * MT * self.HT * 2048, ttnn.bfloat16, 2048)
        cb(OUT, "out", NOUT * MT * self.PWDMAX * 2048, ttnn.bfloat16, 2048)
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
                    + self.wg[c]
                    + self.wd[c]
                    + [virt(c, yy).y for yy in range(GY)]
                    + [(c * GY + cy) % self.nbank]
                    + sor
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
        self.brisc = ttnn.KernelDescriptor(
            kernel_source=_BRISC,
            source_type=st,
            core_ranges=allc,
            compile_time_args=bct + xa_ct,
            runtime_args=b_rt,
            config=dm("RISCV_0", ttnn.NOC.NOC_0),
        )
        kernels = [
            self.brisc,
            ttnn.KernelDescriptor(
                kernel_source=_NCRISC,
                source_type=st,
                core_ranges=allc,
                compile_time_args=nct + w_ct + y_ct,
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
            c_rt = ttnn.RuntimeArgs()
            for c in cols:
                for cy in range(GY):
                    c_rt[cx0 + c][cy] = list(wg) + list(wd)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"#define SE_GLU {glu_code}\n" + _COMPUTE,
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
        ]
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
