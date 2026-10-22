# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""se_engine - the shared expert as one engine of the fused MoE head, every weight read from DRAM ONCE a call.

y [rows, H] = (glu(x [rows, H] @ W_gu [H, 2 I])) @ W_down [I, H] on a GX x GY core rectangle (columns cx0 ..), all
of it in L1 (one arena a core: every buffer at the same address on every core):
  - rows: core row cy owns row tiles cy MT .. cy MT + MT - 1 (MT = rows / 32 / GY);
  - x: K steps of KB tiles; step s lives on core (s % GX, cy) of each row ("x own", [step][row][KB]) - the gather lands
    it there (fused head) or the BRISC loads it (alone). Step s is multicast along the row from its owner (L1 -> L1);
  - gate / up: the W blocks [KB x PNG] of the column's tile pairs stream ONCE (step order: `order`, e.g. the chip's
    arrival order) down each column from DRAM (the column's row-0 core reads, multicasts); every core accumulates
    x_rows x W_block into fp32 partials in L1 (packer L1 accumulation), the last step reloads them into DEST, adds its
    product, applies the GLU and packs h [MT x PO] in place ("h own": core (cx, cy) owns h[rows of cy, column cx's
    pairs]) - exactly down's K step cx;
  - down: NPD passes over N (the column's PNPD tiles a pass); per pass the GX K steps (h own of column c, multicast
    along the row) meet W_down blocks [PO_c x PNPD] streamed once; y a pass at a time (NCRISC writes DRAM).
Flow control: per-slot semaphores (XV / WV: set to step + 1 by the sender's multicast, after the data), credits back to
the step's owner (XFREE) / the column's reader (WFREE) once a slot is popped, counted per slot; every core of a row consumes the same x
step sequence (gate / up steps, then down passes x GX)."""

import torch
from ttnn.bringup.combine_rf.lin2d import column_split, glu_weight  # noqa: F401

import ttnn

NXS, NWS = 2, 3  # (x / W slots a core)
# semaphore ids, one a slot each: XV (x landed), XFREE (x credits, on the step's owner), WV, WFREE (on the reader).
# Credits are per slot: one aggregate count let a fast row credit two successive W slots while a slow row credited
# none, so the reader overwrote the slow row's occupied slot (its BRISC then blocked in cb_reserve_back: the hang)
XV, XFREE, WV, WFREE = 0, NXS, 2 * NXS, 2 * NXS + NWS
NSEM = 2 * (NXS + NWS)
XIN, WIN, ACCG, ACCD, HOWN, OUT = 0, 1, 2, 3, 4, 5  # CB ids
SH, SW = 2, 2  # (DEST sub-block: 4 fp32 tiles, half sync; SW = 2 = one gate / up pair)

_COMMON = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t GX = get_compile_time_arg_val(0), GY = get_compile_time_arg_val(1), MT = get_compile_time_arg_val(2);
constexpr uint32_t KB = get_compile_time_arg_val(3), NSG = get_compile_time_arg_val(4), POMAX = get_compile_time_arg_val(5);
constexpr uint32_t NPD = get_compile_time_arg_val(6), NXS = get_compile_time_arg_val(7), NWS = get_compile_time_arg_val(8);
constexpr uint32_t XSLOT = get_compile_time_arg_val(9), WSLOT = get_compile_time_arg_val(10), WTB = get_compile_time_arg_val(11);
constexpr uint32_t KT = get_compile_time_arg_val(12), NTH = get_compile_time_arg_val(13), HT = get_compile_time_arg_val(14);
constexpr uint32_t NG = NSG + NPD * GX;  // x (and W) steps a call
constexpr uint32_t XV = 0, XFREE = NXS, WV = 2 * NXS, WFREE = 2 * NXS + NWS;  // (one a slot each)
constexpr uint32_t CB_XIN = 0, CB_WIN = 1, CB_HOWN = 4, CB_OUT = 5;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
"""

_BRISC = (
    _COMMON
    + r"""
constexpr auto x_args = TensorAccessorArgs<15>();
// x: owner of step g (gate / up step order[g], or down pass (g - NSG) / GX's step (g - NSG) % GX)
void kernel_main() {
    size_t a = 0;
    const uint32_t cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);
    const uint32_t lx = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t x_addr = get_arg_val<uint32_t>(a++), load_x = get_arg_val<uint32_t>(a++);
    const uint32_t xin = get_arg_val<uint32_t>(a++), xown = get_arg_val<uint32_t>(a++), hown = get_arg_val<uint32_t>(a++);
    const uint32_t words = get_arg_val<uint32_t>(a++), rdx = get_arg_val<uint32_t>(a++), rdy = get_arg_val<uint32_t>(a++);
    const uint32_t ord0 = a;  // order[NSG], then the GX columns' NoC x (this row)
    const uint32_t oxs = ord0 + NSG;
    auto order = [&](uint32_t g) { return get_arg_val<uint32_t>(ord0 + g); };
    auto owner = [&](uint32_t g) { return g < NSG ? order(g) % GX : (g - NSG) % GX; };
    if (load_x) {  // (alone: my x steps from DRAM x [rows, H])
        const auto xg = TensorAccessor(x_args, x_addr, 2048);
        for (uint32_t s = cx, j = 0; s < NSG; s += GX, ++j)
            for (uint32_t r = 0; r < MT; ++r)
                for (uint32_t kk = 0; kk < KB; ++kk)
                    noc_async_read(xg.get_noc_addr((cy * MT + r) * KT + s * KB + kk), xown + ((j * MT + r) * KB + kk) * 2048, 2048);
        noc_async_read_barrier();
    }
    uint32_t gs = 0;  // (my next x step to send)
    while (gs < NG && owner(gs) != cx) ++gs;
    uint32_t nf[NXS] = {}, gr = 0, gc = NXS, wr = 0, wc = NWS;  // (my steps a slot past the first round; x / W pushed, credited)
    bool h_ok = false;
    while (gs < NG || gr < NG || gc < NG || wr < NG || wc < NG) {
        invalidate_l1_cache();
        if (gs < NG) {  // send my step gs once its slot is free on the whole row (and h is packed, for down)
            if (gs >= NSG && !h_ok) h_ok = cb_pages_available_at_front(CB_HOWN, MT * POMAX);
            const bool ready = gs < NSG || h_ok;
            if (ready && (gs < NXS || *sem(XFREE + gs % NXS) >= GX * (nf[gs % NXS] + 1))) {
                const uint32_t src = gs < NSG ? xown + (order(gs) / GX) * MT * KB * 2048 : hown;
                const uint32_t bytes = (gs < NSG ? KB : POMAX) * MT * 2048;
                const uint32_t dst = xin + (gs % NXS) * XSLOT * 2048;
                const uint32_t vw = words + (gs & 3) * 16;
                *word(vw) = gs + 1;
                // the east half [me .. right end] on NOC0, starting at me (the NCRISC sends the west half on NOC1: a
                // multicast whose source sits inside its rectangle away from the sweep's start corrupted / hung)
                const uint32_t mx = get_arg_val<uint32_t>(oxs + cx);
                if (cx + 1 < GX) {
                    noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(mx, ry, rx, ry, dst), bytes, GX - cx);
                    noc_async_writes_flushed();  // (the data's command issued before the semaphore's)
                    noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(mx, ry, rx, ry, get_semaphore(XV + gs % NXS)), GX - cx);
                } else {  // (just me)
                    noc_async_write(src, get_noc_addr(mx, ry, dst), bytes);
                    noc_async_writes_flushed();
                    noc_async_write(vw, get_noc_addr(mx, ry, get_semaphore(XV + gs % NXS)), 4);
                }
                noc_async_writes_flushed();
                if (gs >= NXS) ++nf[gs % NXS];
                ++gs;
                while (gs < NG && owner(gs) != cx) ++gs;
            }
        }
        if (gr < NG && *sem(XV + gr % NXS) >= gr + 1) {  // x step gr landed here
            cb_reserve_back(CB_XIN, XSLOT);
            cb_push_back(CB_XIN, XSLOT);
            ++gr;
        }
        if (gc < NG && gc < gr + NXS && cb_pages_reservable_at_back(CB_XIN, XSLOT * (gc - gr + 1))) {  // slot of gc free
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(oxs + owner(gc)), ry, get_semaphore(XFREE + gc % NXS)), 1);
            ++gc;
        }
        if (wr < NG && *sem(WV + wr % NWS) >= wr + 1) {  // W step wr landed here
            cb_reserve_back(CB_WIN, WSLOT);
            cb_push_back(CB_WIN, WSLOT);
            ++wr;
        }
        if (wc < NG && wc < wr + NWS && cb_pages_reservable_at_back(CB_WIN, WSLOT * (wc - wr + 1))) {
            noc_semaphore_inc(get_noc_addr(rdx, rdy, get_semaphore(WFREE + wc % NWS)), 1);
            ++wc;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
"""
)

_NCRISC = (
    _COMMON
    + r"""
constexpr auto w_args = TensorAccessorArgs<15>();
constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
// the column's W reader (row 0): step g's block from DRAM (once), multicast down the column; every core: y a pass at a time
void kernel_main() {
    size_t a = 0;
    const uint32_t cy = get_arg_val<uint32_t>(a++), cxv = get_arg_val<uint32_t>(a++);
    const uint32_t ty = get_arg_val<uint32_t>(a++), by = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++), wbase = get_arg_val<uint32_t>(a++);
    const uint32_t png = get_arg_val<uint32_t>(a++), pnpd = get_arg_val<uint32_t>(a++), n0d = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++), win = get_arg_val<uint32_t>(a++), wst = get_arg_val<uint32_t>(a++);
    const uint32_t words = get_arg_val<uint32_t>(a++);
    const uint32_t ord0 = a, h00 = a + NSG, po0 = h00 + GX;
    const auto wg = TensorAccessor(w_args, w_addr, WTB);
    const auto yg = TensorAccessor(y_args, y_addr, 2048);
    const bool reader = cy == get_arg_val<uint32_t>(ord0 + NSG + 2 * GX);
    // the west half of my x steps: [left end .. me - 1] on NOC1 (its sweep starts at my west neighbour)
    const uint32_t xa = ord0 + NSG + 2 * GX + 1;
    const uint32_t cx = get_arg_val<uint32_t>(xa), lx = get_arg_val<uint32_t>(xa + 1), ry = get_arg_val<uint32_t>(xa + 2);
    const uint32_t xin = get_arg_val<uint32_t>(xa + 3), xown = get_arg_val<uint32_t>(xa + 4), hown = get_arg_val<uint32_t>(xa + 5);
    const uint32_t vxs = xa + 6;  // (the GX columns' NoC x, this row)
    auto order = [&](uint32_t g) { return get_arg_val<uint32_t>(ord0 + g); };
    auto owner = [&](uint32_t g) { return g < NSG ? order(g) % GX : (g - NSG) % GX; };
    uint32_t gs = cx > 0 ? 0 : NG, nf[NXS] = {};
    while (gs < NG && owner(gs) != cx) ++gs;
    bool h_ok = false;
    uint32_t w = reader ? 0 : NG, p = 0;
    while (w < NG || p < NPD || gs < NG) {
        invalidate_l1_cache();
        if (gs < NG) {
            if (gs >= NSG && !h_ok) h_ok = cb_pages_available_at_front(CB_HOWN, MT * POMAX);
            if ((gs < NSG || h_ok) && (gs < NXS || *sem(XFREE + gs % NXS) >= GX * (nf[gs % NXS] + 1))) {
                const uint32_t src = gs < NSG ? xown + (order(gs) / GX) * MT * KB * 2048 : hown;
                const uint32_t bytes = (gs < NSG ? KB : POMAX) * MT * 2048;
                const uint32_t dst = xin + (gs % NXS) * XSLOT * 2048;
                const uint32_t vw = words + 128 + (gs & 3) * 16;
                *word(vw) = gs + 1;
                const uint32_t wx = get_arg_val<uint32_t>(vxs + cx - 1);
                noc_async_write_multicast(src, get_noc_multicast_addr(wx, ry, lx, ry, dst), bytes, cx);
                noc_async_writes_flushed();
                noc_semaphore_set_multicast(vw, get_noc_multicast_addr(wx, ry, lx, ry, get_semaphore(XV + gs % NXS)), cx);
                noc_async_writes_flushed();
                if (gs >= NXS) ++nf[gs % NXS];
                ++gs;
                while (gs < NG && owner(gs) != cx) ++gs;
            }
        }
        if (w < NG && (w < NWS || *sem(WFREE + w % NWS) >= GY * (w / NWS))) {
            uint32_t page, n;
            if (w < NSG) {
                n = KB * png;
                page = wbase + get_arg_val<uint32_t>(ord0 + w) * n;
            } else {
                const uint32_t pp = (w - NSG) / GX, c = (w - NSG) % GX;
                n = get_arg_val<uint32_t>(po0 + c) * pnpd;
                page = wbase + NSG * KB * png + pp * HT * pnpd + get_arg_val<uint32_t>(h00 + c) * pnpd;
            }
            const uint32_t stage = wst + (w & 1) * WSLOT * WTB;
            for (uint32_t t = 0; t < n; ++t) noc_async_read(wg.get_noc_addr(page + t), stage + t * WTB, WTB);
            noc_async_read_barrier();
            const uint32_t vw = words + 64 + (w & 3) * 16;
            *word(vw) = w + 1;
            // to every core of the column by unicast (each NoC carries only row multicasts: a column multicast on the
            // NoC of the rows' west halves stalled the packers - an L1 write port held by a stuck multicast)
            for (uint32_t r = 0; r < GY; ++r)
                noc_async_write(stage, get_noc_addr(cxv, get_arg_val<uint32_t>(vxs + GX + r), win + (w % NWS) * WSLOT * WTB), n * WTB);
            noc_async_writes_flushed();
            for (uint32_t r = 0; r < GY; ++r)
                noc_async_write(vw, get_noc_addr(cxv, get_arg_val<uint32_t>(vxs + GX + r), get_semaphore(WV + w % NWS)), 4);
            noc_async_writes_flushed();
            ++w;
        }
        if (p < NPD && cb_pages_available_at_front(CB_OUT, MT * pnpd)) {  // down pass p's y
            cb_wait_front(CB_OUT, MT * pnpd);
            const uint32_t l1 = get_read_ptr(CB_OUT);
            for (uint32_t r = 0; r < MT; ++r)
                for (uint32_t j = 0; j < pnpd; ++j)
                    noc_async_write(l1 + (r * pnpd + j) * 2048, yg.get_noc_addr((cy * MT + r) * NTH + n0d + p * pnpd + j), 2048);
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, MT * pnpd);
            ++p;
        }
    }
    noc_async_write_barrier();
}
"""
)

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#if defined(SE_GLU) && defined(TRISC_PACK)
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if SE_GLU == 2
#include "ckernel_sfpu_situ_glu.h"
#endif
#endif
constexpr uint32_t MT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1), NSG = get_compile_time_arg_val(2);
constexpr uint32_t POMAX = get_compile_time_arg_val(3), NPD = get_compile_time_arg_val(4), GX = get_compile_time_arg_val(5);
constexpr uint32_t PNG = get_compile_time_arg_val(6), PNPD = get_compile_time_arg_val(7);
constexpr uint32_t XSLOT = get_compile_time_arg_val(8), WSLOT = get_compile_time_arg_val(9);
#ifndef SE_DBG_NO_RELOAD
#define SE_DBG_NO_RELOAD 0
#endif
#ifndef SE_DBG_NO_GLU
#define SE_DBG_NO_GLU 0
#endif
constexpr uint32_t SH = 2, SW = 2;
constexpr uint32_t XIN = 0, WIN = 1, ACCG = 2, ACCD = 3, HOWN = 4, OUT = 5;

// one pass: NSTEP x / W steps (kt tiles of K each, x rows strided XK), partials in ACC (MT x PN), the last step into
// DEST over the reloaded partials, then out(): pack the sub-block (GLU -> h own, or y)
template <uint32_t XK, uint32_t PN, uint32_t ACC, bool GU>
FORCE_INLINE void pass(uint32_t nstep) {
    constexpr uint32_t NACC = MT * PN;
    cb_reserve_back(ACC, NACC);
    pack_reconfig_data_format(ACC);
    matmul_block_init(XIN, WIN, 0, SW, SH, XK);
    for (uint32_t i = 0; i < nstep; ++i) {
        const uint32_t kt = GU ? KB : get_arg_val<uint32_t>(i);
        const bool last = i + 1 == nstep;
        cb_wait_front(XIN, XSLOT);
        cb_wait_front(WIN, WSLOT);
        if (last) {  // (the partials of steps 0 .. nstep - 2 are packed: hand them to the unpacker)
            cb_push_back(ACC, NACC);
            cb_wait_front(ACC, NACC);
            if (GU) cb_reserve_back(HOWN, MT * POMAX); else cb_reserve_back(OUT, MT * PN);
            pack_reconfig_l1_acc(0);
            pack_reconfig_data_format(GU ? HOWN : OUT);
        } else {
#ifndef SE_DBG_NO_L1ACC
            pack_reconfig_l1_acc(i > 0);
#endif
        }
        for (uint32_t rb = 0; rb < MT / SH; ++rb)
            for (uint32_t cb = 0; cb < PN / SW; ++cb) {
                tile_regs_acquire();
                if (last && !SE_DBG_NO_RELOAD) {
                    reconfig_data_format_srca(WIN, ACC);
                    copy_init(ACC);
                    for (uint32_t t = 0; t < SH * SW; ++t) copy_tile(ACC, (rb * SH + t / SW) * PN + cb * SW + t % SW, t);
                    reconfig_data_format_srca(ACC, WIN);
                    matmul_block_init(XIN, WIN, 0, SW, SH, XK);
                }
                for (uint32_t k = 0; k < kt; ++k) matmul_block(XIN, WIN, rb * SH * XK + k, k * PN + cb * SW, 0, false, SW, SH, XK);
                tile_regs_commit();
                tile_regs_wait();
                if (!last) {
                    for (uint32_t t = 0; t < SH * SW; ++t) pack_tile<true>(t, ACC, (rb * SH + t / SW) * PN + cb * SW + t % SW);
                } else if (GU) {
#if defined(SE_GLU) && !SE_DBG_NO_GLU
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if SE_GLU == 2
                    for (uint32_t t = 0; t < SH * SW; t += 2)
                        PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_situ_glu, (DST_ACCUM_MODE, 8, sfpu::SituGluConfigKimi), t, t + 1, t, VectorMode::RC)));
#else
                    for (uint32_t t = 0; t < SH * SW; t += 2) silu_tile_pack(t);
                    for (uint32_t t = 0; t < SH * SW; t += 2)
                        PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul, (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE), t, t + 1, t, VectorMode::RC)));
#endif
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
                    for (uint32_t t = 0; t < SH * SW; t += 2) pack_tile<true>(t, HOWN, (rb * SH + t / SW) * POMAX + cb);
                } else {
                    for (uint32_t t = 0; t < SH * SW; ++t) pack_tile<true>(t, OUT, (rb * SH + t / SW) * PN + cb * SW + t % SW);
                }
                tile_regs_release();
            }
        cb_pop_front(XIN, XSLOT);
        cb_pop_front(WIN, WSLOT);
    }
    cb_pop_front(ACC, NACC);
    if (GU) cb_push_back(HOWN, MT * POMAX); else cb_push_back(OUT, MT * PN);
}

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(XIN, WIN, ACCG);
#if defined(SE_GLU) && !SE_DBG_NO_GLU
#if SE_GLU == 2
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
#else
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
#endif
    if constexpr (NSG > 0) {
        pass<KB, PNG, ACCG, true>(NSG);
    } else {  // (debug: down only - h as it is)
        cb_reserve_back(HOWN, MT * POMAX);
        cb_push_back(HOWN, MT * POMAX);
    }
    for (uint32_t p = 0; p < NPD; ++p) pass<POMAX, PNPD, ACCD, false>(GX);
}
"""


def _dbg():
    """(debug) compute defines from SE_DBG=NO_L1ACC,NO_RELOAD,NO_GLU (the defines' values are 1)."""
    import os

    return [f"SE_DBG_{d} 1" for d in os.environ.get("SE_DBG", "").split(",") if d]


def _down_split(Nt, GX, max_pn=6):
    """(NPD, per-column tiles): every column a multiple of 2 NPD tiles, at most max_pn a pass."""
    for npd in range(1, Nt // 2 + 1):
        if Nt % (2 * npd):
            continue
        pns = column_split(Nt, GX, 2 * npd)
        if max(pns) // npd <= max_pn:
            return npd, pns
    raise ValueError(f"no down split of {Nt} tiles over {GX} columns")


class SeEngine:
    """The shared expert a chip: x [rows, H] (all of it: the gathered x) -> y [rows, H] (this chip's partial over its
    I slice). glu "silu" / "situ". order: gate / up K-step order (None: natural; the fused head: arrival order)."""

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
        KB=None,
        order=None,
        fidelity=ttnn.MathFidelity.HiFi2,
        w_dtype=ttnn.bfloat8_b,
    ):
        self.mesh = mesh_device
        assert rows % (32 * GY) == 0
        self.rows, self.H, self.I, self.glu = rows, H, I, glu
        self.cx0, self.GX, self.GY, self.MT = cx0, GX, GY, rows // 32 // GY
        assert self.MT % SH == 0
        self.KT, self.HT, self.NTH = H // 32, I // 32, H // 32
        self.KB = KB or (8 if H <= 4096 else 4)
        assert self.KT % self.KB == 0
        self.NSG = self.KT // self.KB
        self.order = list(order) if order is not None else list(range(self.NSG))
        assert sorted(self.order) == list(range(self.NSG))
        pairs = column_split(self.HT, GX, 1)  # (gate / up pairs a column = its h tiles)
        assert min(pairs) >= 1, "a column without h tiles"
        self.po = pairs
        self.png = [2 * p for p in pairs]
        self.h0 = [sum(pairs[:c]) for c in range(GX)]
        self.POMAX = max(pairs)
        self.NPD, pnd = _down_split(self.NTH, GX)
        self.pnpd = [p // self.NPD for p in pnd]
        self.n0d = [sum(pnd[:c]) for c in range(GX)]
        self.XSLOT = self.MT * max(self.KB, self.POMAX)
        self.WSLOT = max(max(self.KB * g, self.POMAX * d) for g, d in zip(self.png, self.pnpd))
        self.fid, self.w_dtype = fidelity, w_dtype
        self.wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}[w_dtype]
        # the arena (bytes): XIN, WIN, ACC (gate / up and down partials alias), HOWN, OUT, XOWN, WST, words
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        self.nown = -(-self.NSG // GX)
        sizes = [
            ("xin", NXS * self.XSLOT * 2048),
            ("win", NWS * self.WSLOT * self.wtb),
            ("acc", self.MT * max(max(self.png), max(self.pnpd)) * 4096),
            ("hown", self.MT * self.POMAX * 2048),
            ("out", self.MT * max(self.pnpd) * 2048),
            ("xown", self.nown * self.MT * self.KB * 2048),
            ("wst", 2 * self.WSLOT * self.wtb),
            ("words", 256),
        ]
        self.off, o = {}, 0
        for k, v in sizes:
            self.off[k], o = o, al(o + v)
        self.arena_bytes = o
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
            self.groups.setdefault((self.png[c], self.pnpd[c]), []).append(c)
        import os

        # the columns' W reader: the last row - the start of its NOC1 column multicast (a reader at the far end, row 0,
        # corrupted its own row's results ~1 run in 2; 0 of 4 from the last row)
        self.rrow = int(os.environ.get("SE_READER_ROW", str(GY - 1))) % GY
        self.state = dict(key=None)

    def dram_bytes(self):
        """DRAM bytes a call: weights (once), x (alone: read once), y."""
        w = (self.KT * 2 * self.HT + self.HT * self.NTH) * self.wtb
        return dict(w=w, x=self.rows * self.H * 2, y=self.rows * self.H * 2)

    def upload(self, w_gu, w_down):
        """Per device (row-major mesh order): w_gu [H, 2 I] (lin2c.glu_weight: gate / up tile pairs), w_down [I, H] ->
        one bf8 tensor a chip of the columns' blocks in stream order (natural step order; the reader indexes it)."""
        R, C = tuple(self.mesh.shape)
        KB, NSG, GX = self.KB, self.NSG, self.GX
        devs = []
        for wg, wd in zip(w_gu, w_down):
            tiles = []
            for c in range(GX):
                g0, png = 2 * self.h0[c], self.png[c]
                for s in range(NSG):
                    blk = wg[s * KB * 32 : (s + 1) * KB * 32, g0 * 32 : (g0 + png) * 32]
                    tiles.append(blk.reshape(KB, 32, png, 32).permute(0, 2, 1, 3).reshape(-1, 32))
                pd = self.pnpd[c]
                for p in range(self.NPD):
                    c0 = (self.n0d[c] + p * pd) * 32
                    blk = wd[:, c0 : c0 + pd * 32]
                    tiles.append(blk.reshape(self.HT, 32, pd, 32).permute(0, 2, 1, 3).reshape(-1, 32))
            devs.append(torch.cat(tiles))
        self.wbase = []
        o = 0
        for c in range(GX):
            self.wbase.append(o)
            o += NSG * KB * self.png[c] + self.NPD * self.HT * self.pnpd[c]
        t = torch.stack(devs).reshape(R, C, -1, 32)
        return ttnn.from_torch(
            t,
            dtype=self.w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(0, 1), mesh_shape=(R, C)),
        )

    def pieces(self, x, w, y, *, load_x=True):
        """(CBs, kernels, semaphores) of the engine (the same on every chip). x: DRAM [rows, H] when load_x."""
        mesh, GX, GY, cx0 = self.mesh, self.GX, self.GY, self.cx0
        b, off = self.base, self.off
        virt = lambda cx, cy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0 + cx, cy))  # noqa: E731
        allc = self.crs
        cbs = []

        def cb(i, key, size, dt, page, crs=allc):
            d = ttnn.cb_descriptor_from_sharded_tensor(
                i, self.arena, address_offset=off[key], total_size=size, core_ranges=crs
            )
            d.format_descriptors = [ttnn.CBFormatDescriptor(i, dt, page)]
            cbs.append(d)

        cb(XIN, "xin", NXS * self.XSLOT * 2048, ttnn.bfloat16, 2048)
        cb(WIN, "win", NWS * self.WSLOT * self.wtb, self.w_dtype, self.wtb)
        cb(HOWN, "hown", self.MT * self.POMAX * 2048, ttnn.bfloat16, 2048)
        for (png, pd), cols in self.groups.items():
            crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(cx0 + c, 0), ttnn.CoreCoord(cx0 + c, GY - 1)) for c in cols]
            )
            cb(ACCG, "acc", self.MT * png * 4096, ttnn.float32, 4096, crs)
            cb(ACCD, "acc", self.MT * pd * 4096, ttnn.float32, 4096, crs)
            cb(OUT, "out", self.MT * pd * 2048, ttnn.bfloat16, 2048, crs)
        import os

        phase = os.environ.get("SE_PHASE", "")  # (debug: "gu" / "down" only)
        nsg, npd = (0 if phase == "down" else self.NSG), (0 if phase == "gu" else self.NPD)
        ct = [
            GX,
            GY,
            self.MT,
            self.KB,
            nsg,
            self.POMAX,
            npd,
            NXS,
            NWS,
            self.XSLOT,
            self.WSLOT,
            self.wtb,
            self.KT,
            self.NTH,
            self.HT,
        ]
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        b_rt, n_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in range(GX):
            for cy in range(GY):
                lx, rx, ry = virt(0, cy).x, virt(GX - 1, cy).x, virt(c, cy).y
                rd = virt(c, self.rrow)
                b_rt[cx0 + c][cy] = (
                    [
                        c,
                        cy,
                        lx,
                        rx,
                        ry,
                        int(x.buffer_address()),
                        int(load_x),
                        b + off["xin"],
                        b + off["xown"],
                        b + off["hown"],
                        b + off["words"],
                        rd.x,
                        rd.y,
                    ]
                    + self.order[:nsg]
                    + [virt(cc, cy).x for cc in range(GX)]
                )
                v = virt(c, cy)
                n_rt[cx0 + c][cy] = (
                    [
                        cy,
                        v.x,
                        virt(c, 0).y,
                        virt(c, GY - 1).y,
                        int(w.buffer_address()),
                        self.wbase[c],
                        self.png[c],
                        self.pnpd[c],
                        self.n0d[c],
                        int(y.buffer_address()),
                        b + off["win"],
                        b + off["wst"],
                        b + off["words"],
                    ]
                    + self.order[:nsg]
                    + self.h0
                    + self.po
                    + [self.rrow]
                    + [c, virt(0, cy).x, v.y, b + off["xin"], b + off["xown"], b + off["hown"]]
                    + [virt(cc, cy).x for cc in range(GX)]
                    + [virt(c, yy).y for yy in range(GY)]
                )
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )  # noqa: E731
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_BRISC,
                source_type=st,
                core_ranges=allc,
                compile_time_args=ct + x_ct,
                runtime_args=b_rt,
                config=dm("RISCV_0", ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_NCRISC,
                source_type=st,
                core_ranges=allc,
                compile_time_args=ct + w_ct + y_ct,
                runtime_args=n_rt,
                config=dm("RISCV_1", ttnn.NOC.NOC_1),
            ),
        ]
        glu_code = {"silu": 1, "situ": 2}[self.glu]
        for (png, pd), cols in self.groups.items():
            crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(cx0 + c, 0), ttnn.CoreCoord(cx0 + c, GY - 1)) for c in cols]
            )
            comp = ttnn.ComputeConfigDescriptor()
            comp.math_fidelity = self.fid
            comp.fp32_dest_acc_en = True
            comp.dst_full_sync_en = "SE_DBG_FULL_SYNC 1" in _dbg()
            comp.math_approx_mode = False
            modes = [ttnn.UnpackToDestMode.Default] * 64  # (NUM_CIRCULAR_BUFFERS)
            modes[ACCG] = modes[ACCD] = ttnn.UnpackToDestMode.UnpackToDestFp32
            comp.unpack_to_dest_mode = modes
            c_rt = ttnn.RuntimeArgs()
            for c in cols:
                for cy in range(GY):
                    c_rt[cx0 + c][cy] = list(self.po)  # (down step c: po[c] K tiles)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"#define SE_GLU {glu_code}\n" + "".join(f"#define {d}\n" for d in _dbg()) + _COMPUTE,
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=[self.MT, self.KB, nsg, self.POMAX, npd, GX, png, pd, self.XSLOT, self.WSLOT],
                    runtime_args=c_rt,
                    config=comp,
                )
            )
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=allc, initial_value=0) for i in range(NSEM)]
        return cbs, kernels, sems

    def run_alone(self, x, w):
        """Test: the engine alone (x [1, 1, rows, H] in DRAM, every chip its own)."""
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        prog = ttnn.ProgramDescriptor()
        prog.cbs, prog.kernels, prog.semaphores = self.pieces(x, w, y)
        R, C = tuple(self.mesh.shape)
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
        ttnn.generic_op([x, w, y, self.arena], md)
        return y
