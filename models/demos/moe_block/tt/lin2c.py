# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""lin2c - combine_rf/lin2d's 2D-multicast matmul, chunked, as an engine of a larger program.

y [MC x M, N] = x [MC x M, K] @ W [K, N], one chunk of M = 32 x GY rows at a time on a GX x GY core rectangle (columns
cx0 .., rows 0 ..): per chunk x is resident (row tile cy of the chunk on core row cy, multicast along the row by its two
ends), W streams down each column from DRAM (re-read every chunk), y is written per chunk. What lin2d does for one call
(see its docstring for the NoC / DRAM layout), plus, per chunk:
  - x ready: an x sender (a row end) issues chunk m's x reads once its XREADY semaphore reaches xr_need (m + 1) - the
    producer (the head's x movers, or another engine) increments it; xr_need 0: no wait;
  - x free: every core's BRISC tells its row's x senders (XFREE) when its compute has popped a chunk's x; a sender
    multicasts chunk m's x once all GX cores freed chunk m - 1;
  - the column barrier (W multicasts wait for every core of the column to hold the chunk's x) per chunk;
  - y notify: after a chunk's y is written, each core increments a semaphore (NOTE_SEM, the same id in the program) on
    its consumer cores (e.g. the next engine's x senders of the same row).
GLU (gate / up tile pairs -> the activation) as lin2d's: kind "situ" (Kimi) or "silu" (silu(g) * u)."""


from ttnn.bringup.combine_rf.lin2d import column_split, glu_weight, w_streams  # noqa: F401

import ttnn

XREADY, XFREE = 14, 15  # semaphore ids past lin2d's (0 .. 9 + nb: nb <= 4; 16 ids a core)

_COMMON = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4), NB = get_compile_time_arg_val(5);
constexpr uint32_t GX = get_compile_time_arg_val(6), GY = get_compile_time_arg_val(7), WTB = get_compile_time_arg_val(8);
constexpr uint32_t PNP = get_compile_time_arg_val(9), NT = get_compile_time_arg_val(10);
constexpr uint32_t MC = get_compile_time_arg_val(11), NST = get_compile_time_arg_val(12);
constexpr uint32_t XS = get_compile_time_arg_val(13), Q1S = get_compile_time_arg_val(14);
constexpr uint32_t NSTW = get_compile_time_arg_val(15), GLU = get_compile_time_arg_val(16);
constexpr uint32_t WBMAX = get_compile_time_arg_val(17), XRN = get_compile_time_arg_val(18);  // (XRN: XREADY a chunk)
constexpr uint32_t NOTE = get_compile_time_arg_val(19);  // (the consumers' semaphore id; 0xFF: none)
constexpr uint32_t PO = GLU ? PNP / 2 : PNP;
constexpr auto x_args = TensorAccessorArgs<20>();
constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr uint32_t XB = KB * 2048, WB = KB * PNPX * WTB, NS = NP * NBLK, WPT = KB * PNPX;
constexpr uint32_t XV = 0, WV = 2, XDQ = 8, WCB = 10, LANDED = 4, SENT = 6, XREADY = 14, XFREE = 15;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
// x: sender q (XS = 2: left even blocks, right odd) owns NOWN(q) blocks a chunk; its k-th piece (over all chunks) is chunk
// k / NOWN(q), block own_at(q, k % NOWN(q))
constexpr uint32_t n_own(uint32_t q) { return XS == 2 ? (NBLK + 1 - q) / 2 : NBLK; }
constexpr uint32_t own_at(uint32_t q, uint32_t k) { return XS == 2 ? 2 * k + q : k; }
constexpr uint32_t q_of(uint32_t q1, uint32_t i) { return XS == 2 ? (i & 1) : q1; }
constexpr uint32_t k_of(uint32_t i) { return XS == 2 ? (i >> 1) : i; }
// W step st (global, over chunks: NS a chunk, even) is chunk st / NS; the q stream's k-th piece is step 2 k + q
FORCE_INLINE bool w_credit_ok(uint32_t q, uint32_t k) {
    const uint32_t st = 2 * k + q;
    if (*sem(XDQ + q) < GY * (st / NS + 1)) return false;
    return st < NB || *sem(WCB + q * (NB / 2) + (st % NB) / 2) >= GY * (st / NB);
}
"""

_BRISC = (
    _COMMON
    + r"""
struct Stream {
    bool on = false, send = false, isw = false;
    uint32_t nst = NST, q = 0, n = 0, issued = 0, landed = 0, sent = 0, stage = 0, stride = 0, piece = 0, trid0 = 0;
    uint32_t tcx = 0, w_addr = 0, bank = 0, o = 0, nc = 0;  // (nc: pieces a chunk)
};
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cx = get_arg_val<uint32_t>(2), cy = get_arg_val<uint32_t>(3);
    const uint32_t lx = get_arg_val<uint32_t>(4), rx = get_arg_val<uint32_t>(5), ry = get_arg_val<uint32_t>(6);
    const uint32_t ty = get_arg_val<uint32_t>(7), by = get_arg_val<uint32_t>(8), cxv = get_arg_val<uint32_t>(9);
    const uint32_t s0x = get_arg_val<uint32_t>(12), s0y = get_arg_val<uint32_t>(13);
    const uint32_t s1x = get_arg_val<uint32_t>(14), s1y = get_arg_val<uint32_t>(15);
    const uint32_t lxy = get_arg_val<uint32_t>(30), rxy = get_arg_val<uint32_t>(31);  // (the row's x senders: NoC x)
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t xres = get_write_ptr(0), wring = get_write_ptr(1), words = get_write_ptr(10);  // (bases, at start)
    Stream S[3];
    Stream& X = S[0];
    X.on = (XS == 2 && cx == 0) || cx == GX - 1, X.send = XS == 2 && cx == 0, X.q = X.send ? 0 : 1;
    X.nc = n_own(X.q), X.n = MC * X.nc;
    X.stage = get_write_ptr(8), X.stride = XB, X.piece = XB, X.trid0 = 1;
    for (uint32_t q = 0; q < 2; ++q) {
        Stream& T = S[1 + q];
        const uint32_t a = 16 + 7 * q;
        T.on = get_arg_val<uint32_t>(a), T.send = q == 0, T.isw = true, T.q = q, T.nst = NSTW;
        T.tcx = get_arg_val<uint32_t>(a + 1), T.w_addr = get_arg_val<uint32_t>(a + 2), T.piece = get_arg_val<uint32_t>(a + 3);
        T.bank = get_arg_val<uint32_t>(a + 4), T.o = get_arg_val<uint32_t>(a + 5), T.nc = get_arg_val<uint32_t>(a + 6);
        T.n = MC * T.nc;
        T.stage = get_write_ptr(9) + q * Q1S * WBMAX, T.stride = WBMAX, T.trid0 = 1 + NST + q * NSTW;
    }
    uint32_t xr = 0, r = 0, c = NB, fr = 0;  // (x blocks / W steps pushed; next W step to credit; chunks freed)
    auto streams_busy = [&]() {
        for (auto& T : S)
            if (T.on && (T.send ? T.sent < T.n : T.landed < T.n)) return true;
        return false;
    };
    while (xr < MC * NBLK || r < MC * NS || c < MC * NS || fr < MC || streams_busy()) {
        for (auto& T : S) {
            if (!T.on) continue;
            if (!T.send) {
                invalidate_l1_cache();
                T.sent = *sem(SENT + T.isw);
            }
            while (T.issued < T.n && T.issued < T.sent + T.nst) {
                const uint32_t k = T.issued, slot = k % T.nst, dst = T.stage + slot * T.stride;
                if (!T.isw) {  // (chunk k / nc's x must have landed in the source: XREADY)
                    invalidate_l1_cache();
                    if (XRN && *sem(XREADY) < XRN * (k / T.nc + 1)) break;
                }
                if (T.send) noc_async_writes_flushed();
                noc_async_read_set_trid(T.trid0 + slot);
                if (T.isw) {
                    const uint32_t piece = T.o + ((2 * k + T.q) % NS) / 2;
                    noc_async_read(get_noc_addr_from_bank_id<true>(T.bank, T.w_addr + piece * T.piece), dst, T.piece);
                } else {
                    const uint32_t m = k / T.nc, j = own_at(T.q, k % T.nc);
                    for (uint32_t t = 0; t < KB; ++t)
                        noc_async_read(xg.get_noc_addr((m * GY + cy) * KT + j * KB + t), dst + t * 2048, 2048);
                }
                ++T.issued;
            }
            while (T.landed < T.issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, T.trid0 + T.landed % T.nst)) {
                ++T.landed;
                if (!T.send) *sem(LANDED + T.isw) = T.landed;
            }
            if (T.send && T.sent < T.landed) {
                const uint32_t k = T.sent, slot = k % T.nst;
                invalidate_l1_cache();
                // W: the column has the chunk's x and the ring slot is free; x: the row freed the previous chunk
                const bool ok = T.isw ? w_credit_ok(0, k) : (k / T.nc == 0 || *sem(XFREE) >= GX * (k / T.nc));
                if (ok) {
                    const uint32_t src = T.stage + slot * T.stride;
                    const uint32_t vw = words + (T.isw ? 64 : 0) + (k & 3) * 16;
                    *word(vw) = k + 1;
                    if (T.isw) {
                        const uint64_t d = get_noc_multicast_addr(T.tcx, ty, T.tcx, by, wring + ((2 * k) % NB) * T.piece);
                        const uint64_t v = get_noc_multicast_addr(T.tcx, ty, T.tcx, by, get_semaphore(WV));
                        if (T.tcx == cxv) {
                            noc_async_write_multicast_loopback_src(src, d, T.piece, GY);
                            noc_semaphore_set_multicast_loopback_src(vw, v, GY);
                        } else {
                            noc_async_write_multicast(src, d, T.piece, GY);
                            noc_semaphore_set_multicast(vw, v, GY);
                        }
                    } else {
                        noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(lx, ry, rx, ry, xres + own_at(0, k % T.nc) * XB), T.piece, GX);
                        noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(lx, ry, rx, ry, get_semaphore(XV)), GX);
                    }
                    ++T.sent;
                }
            }
        }
        // receive: x blocks (resident a chunk, pass 0) and W steps in order
        invalidate_l1_cache();
        if (xr < MC * NBLK && (xr % NBLK != 0 || fr * NBLK >= xr) && *sem(XV + q_of(1, xr % NBLK)) >= (xr / NBLK) * n_own(q_of(1, xr % NBLK)) + k_of(xr % NBLK) + 1) {
            cb_reserve_back(0, KB);
            cb_push_back(0, KB);
            if (++xr % NBLK == 0) {  // (a chunk's x complete here: the column barrier)
                noc_semaphore_inc(get_noc_addr(s0x, s0y, get_semaphore(XDQ)), 1);
                noc_semaphore_inc(get_noc_addr(s1x, s1y, get_semaphore(XDQ + 1)), 1);
            }
        }
        // x free: the compute popped chunk fr's x (the whole CB reservable again): tell the row's x senders
        if (fr < MC && fr * NBLK < xr && xr % NBLK == 0 && xr / NBLK == fr + 1 && cb_pages_reservable_at_back(0, KT)) {
            if (XS == 2) noc_semaphore_inc(get_noc_addr(lxy, ry, get_semaphore(XFREE)), 1);
            noc_semaphore_inc(get_noc_addr(rxy, ry, get_semaphore(XFREE)), 1);
            ++fr;
        }
        if (r < MC * NS && *sem(WV + (r & 1)) >= (r >> 1) + 1) {
            cb_reserve_back(1, WPT);
            cb_push_back(1, WPT);
            ++r;
        }
        if (c < MC * NS && c < r + NB && cb_pages_reservable_at_back(1, WPT * (c - r + 1))) {
            const uint32_t q = c & 1;
            noc_semaphore_inc(get_noc_addr(q ? s1x : s0x, q ? s1y : s0y, get_semaphore(WCB + q * (NB / 2) + (c % NB) / 2)), 1);
            ++c;
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
void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t cx = get_arg_val<uint32_t>(1), cy = get_arg_val<uint32_t>(2), n0 = get_arg_val<uint32_t>(3);
    const uint32_t rx = get_arg_val<uint32_t>(4), lx = get_arg_val<uint32_t>(5), ry = get_arg_val<uint32_t>(6);
    const uint32_t by = get_arg_val<uint32_t>(7), ty = get_arg_val<uint32_t>(8), cxv = get_arg_val<uint32_t>(9);
    const auto yg = TensorAccessor(y_args, y_addr, 2048);
    const uint32_t xres = get_write_ptr(0), wring = get_write_ptr(1), words = get_write_ptr(10);
    const uint32_t xst = get_write_ptr(8), wst = get_write_ptr(9);
    const bool xo = cx == GX - 1;
    const uint32_t tcx = get_arg_val<uint32_t>(12), wb = get_arg_val<uint32_t>(13), nwc = get_arg_val<uint32_t>(14);
    const uint32_t n_note = get_arg_val<uint32_t>(15);  // consumers (NoC x, y) from 16
    const uint32_t nxc = xo && XS == 2 ? n_own(1) : (xo ? n_own(1) : 0);
    const uint32_t nx = MC * nxc, nw = MC * nwc;
    uint32_t xs = 0, ws = 0;
    uint32_t m = 0;  // (the chunk whose y is next)
    auto y_step = [&]() {  // write chunk m's y when the compute has packed it
        if (m >= MC || !cb_pages_available_at_front(4, NP * PO)) return;
        cb_wait_front(4, NP * PO);
        const uint32_t l1 = get_read_ptr(4);
        for (uint32_t p = 0; p < NP; ++p)
            for (uint32_t t = 0; t < PO; ++t)
                noc_async_write(l1 + (p * PO + t) * 2048, yg.get_noc_addr((m * GY + cy) * NT + n0 / (GLU + 1) + p * PO + t), 2048);
        noc_async_write_barrier();
        cb_pop_front(4, NP * PO);
        if (NOTE != 0xFF)
            for (uint32_t i = 0; i < n_note; ++i)
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(16 + 2 * i), get_arg_val<uint32_t>(17 + 2 * i), get_semaphore(NOTE)), 1);
        ++m;
    };
    while (xs < nx || ws < nw || m < MC) {
        invalidate_l1_cache();
        if (xs < nx && xs < *sem(LANDED) && (xs / nxc == 0 || *sem(XFREE) >= GX * (xs / nxc))) {
            const uint32_t vw = words + 128 + (xs & 3) * 16;
            *word(vw) = xs + 1;
            noc_async_write_multicast_loopback_src(xst + (xs % NST) * XB, get_noc_multicast_addr(rx, ry, lx, ry, xres + own_at(1, xs % nxc) * XB), XB, GX);
            noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(rx, ry, lx, ry, get_semaphore(XV + 1)), GX);
            noc_async_writes_flushed();
            *sem(SENT) = ++xs;
        }
        if (ws < nw && ws < *sem(LANDED + 1) && w_credit_ok(1, ws)) {
            const uint32_t vw = words + 192 + (ws & 3) * 16;
            *word(vw) = ws + 1;
            const uint32_t src = wst + (Q1S + ws % NSTW) * WBMAX;
            const uint64_t d = get_noc_multicast_addr(tcx, by, tcx, ty, wring + ((2 * ws + 1) % NB) * wb);
            const uint64_t v = get_noc_multicast_addr(tcx, by, tcx, ty, get_semaphore(WV + 1));
            if (tcx == cxv) {
                noc_async_write_multicast_loopback_src(src, d, wb, GY);
                noc_semaphore_set_multicast_loopback_src(vw, v, GY);
            } else {
                noc_async_write_multicast(src, d, wb, GY);
                noc_semaphore_set_multicast(vw, v, GY);
            }
            noc_async_writes_flushed();
            *sem(SENT + 1) = ++ws;
        }
        y_step();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
"""
)

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#if defined(L2C_GLU) && defined(TRISC_PACK)
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if L2C_GLU == 2
#include "ckernel_sfpu_situ_glu.h"
#endif
#endif
constexpr uint32_t KB = get_compile_time_arg_val(0), PNP = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4), MC = get_compile_time_arg_val(5);
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 4);
    matmul_block_init(0, 1, 0, PNP, 1, KB);
#ifdef L2C_GLU
#if L2C_GLU == 2
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
#else
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
    constexpr uint32_t PO = PNP / 2;
#else
    constexpr uint32_t PO = PNP;
#endif
    for (uint32_t m = 0; m < MC; ++m) {
        cb_reserve_back(4, NP * PO);
        for (uint32_t p = 0; p < NP; ++p) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < NBLK; ++j) {
                if (p == 0) cb_wait_front(0, (j + 1) * KB);
                cb_wait_front(1, KB * PNPX);
                for (uint32_t k = 0; k < KB; ++k) matmul_block(0, 1, j * KB + k, k * PNPX, 0, false, PNP, 1, KB);
                cb_pop_front(1, KB * PNPX);
            }
            tile_regs_commit();
            tile_regs_wait();
#ifdef L2C_GLU
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if L2C_GLU == 2
            for (uint32_t t = 0; t < PNP; t += 2)
                PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_situ_glu, (DST_ACCUM_MODE, 8, sfpu::SituGluConfigKimi), t, t + 1, t, VectorMode::RC)));
#else
            for (uint32_t t = 0; t < PNP; t += 2) silu_tile_pack(t);
            for (uint32_t t = 0; t < PNP; t += 2)
                PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul, (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE), t, t + 1, t, VectorMode::RC)));
#endif
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            for (uint32_t t = 0; t < PNP; t += 2) pack_tile<true>(t, 4, p * PO + t / 2);
#else
            for (uint32_t t = 0; t < PNP; ++t) pack_tile<true>(t, 4, p * PO + t);
#endif
            tile_regs_release();
        }
        cb_push_back(4, NP * PO);
        cb_pop_front(0, KB * NBLK);
    }
}
"""


class Lin2c:
    """One chunked lin2d engine: columns cx0 .. cx0 + GX - 1, rows 0 .. GY - 1 of the core grid; M = 32 GY rows a chunk,
    MC chunks. glu: None, "silu" or "situ" (W's columns as gate / up tile pairs: glu_weight). xr_need: XREADY increments
    a chunk an x sender waits for (0: x is complete before the program). note: (consumer cores (logical) per core row,
    their semaphore id) or None."""

    def __init__(
        self,
        mesh_device,
        *,
        cx0,
        GX,
        GY,
        MC,
        K,
        N,
        glu=None,
        fidelity=ttnn.MathFidelity.HiFi2,
        kb=8,
        nb=4,
        nst=3,
        nstw=4,
        w_dtype=ttnn.bfloat8_b,
        xr_need=0,
    ):
        self.mesh = mesh_device
        self.cx0, self.GX, self.GY, self.MC, self.K, self.N = cx0, GX, GY, MC, K, N
        self.M = 32 * GY
        assert nb <= 4 and nb % 2 == 0, "W ring slots: semaphores 10 .. 9 + nb must stay below XREADY"
        self.glu, self.fid, self.kb, self.nb, self.nst, self.nstw, self.w_dtype = (
            glu,
            fidelity,
            kb,
            nb,
            nst,
            nstw,
            w_dtype,
        )
        self.xr_need = xr_need
        Kt, Nt = K // 32, N // 32
        self.Kt, self.Nt = Kt, Nt
        G = 2 if glu else 1
        self.G = G
        NP = 1
        while (Nt % (NP * G)) or max(column_split(Nt, GX, NP * G)) // NP > 8:
            NP += 1
        self.NP = NP
        self.pns = column_split(Nt, GX, NP * G)
        assert Kt % kb == 0 and (NP * (Kt // kb)) % 2 == 0, "W steps a chunk must be even"
        self.nblk = Kt // kb
        self.wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}[w_dtype]
        self.nbank = mesh_device.dram_grid_size().x
        self.NS = NP * self.nblk
        self.groups = {}
        for i, pn in enumerate(self.pns):
            self.groups.setdefault(pn, []).append(i)
        self.gidx = {cx: i for i, (pn, cols) in enumerate(sorted(self.groups.items())) for cx in cols}
        self.wbs = [kb * (pn // NP) * self.wtb for pn, _ in sorted(self.groups.items())]
        # (the W stream readers: lin2d's placement on a grid of GX columns, shifted by cx0)
        streams = w_streams(mesh_device, GX, GY, self.NS, 0, self.nbank, [self.wbs[self.gidx[i]] for i in range(GX)])
        used = {}
        for st_ in streams:
            key = (self.gidx[st_["cx"]], st_["bank"])
            st_["o"], used[key] = used.get(key, 0), used.get(key, 0) + st_["n"]
        self.streams = streams
        self.by_reader = {}
        for st_ in streams:
            self.by_reader.setdefault(st_["reader"], {})[st_["q"]] = st_
        self.q1s = nstw if any(len(v) == 2 for v in self.by_reader.values()) else 0
        self.reader_of = {(st_["cx"], st_["q"]): st_["reader"] for st_ in streams}
        self.wbmax = max(self.wbs)
        self.layout = dict(
            pns=self.pns,
            NP=NP,
            kb=kb,
            banks=self.nbank,
            groups=sorted(self.groups.items()),
            streams=streams,
            NS=self.NS,
        )

    def cores(self):
        return [ttnn.CoreCoord(self.cx0 + i, y) for i in range(self.GX) for y in range(self.GY)]

    def x_senders(self, cy):
        """The logical cores whose XREADY a producer increments for core row cy."""
        return [ttnn.CoreCoord(self.cx0, cy), ttnn.CoreCoord(self.cx0 + self.GX - 1, cy)]

    def upload(self, ws):
        """ws: per device (row-major mesh order) W [K, N] torch -> the device weights (lin2d.upload_stacked, per chip)."""
        import torch
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import _bank_sharded

        pns, NP, kb, banks, NS = self.pns, self.NP, self.kb, self.nbank, self.NS
        nblk = NS // NP
        n0s = [sum(pns[:i]) for i in range(len(pns))]
        out = []
        for pn, cols in sorted(self.groups.items()):
            pnp = pn // NP
            mine = [st_ for st_ in self.streams if st_["cx"] in cols]
            per_bank = {}
            for st_ in mine:
                per_bank[st_["bank"]] = max(per_bank.get(st_["bank"], 0), st_["o"] + st_["n"])
            slots = max(per_bank.values())
            dev_regs = []
            for w in ws:
                zero = torch.zeros(kb * pnp * 32, 32, dtype=w.dtype)
                regs = [zero] * (slots * banks)
                for st_ in mine:
                    for k in range(st_["n"]):
                        step = 2 * k + st_["q"]
                        p, j = divmod(step, nblk)
                        c0 = (n0s[st_["cx"]] + p * pnp) * 32
                        blk = w[j * kb * 32 : (j + 1) * kb * 32, c0 : c0 + pnp * 32]
                        regs[(st_["o"] + k) * banks + st_["bank"]] = (
                            blk.reshape(kb, 32, pnp, 32).permute(0, 2, 1, 3).reshape(-1, 32)
                        )
                dev_regs.append(regs)
            out.append(_bank_sharded(dev_regs, banks, self.w_dtype, self.mesh))
        return out

    def pieces(self, x, wl, y, note=None, note_sem=0xFF):
        """(CBs, kernels) of the engine for one program (the same on every chip); semaphores: ids 0 .. 9 + nb, XREADY,
        XFREE on self.cores() (the caller declares them). note: per core row cy, the consumer cores (logical) whose
        semaphore note_sem each core increments once a chunk's y is written."""
        mesh = self.mesh
        GX, GY, NP, kb, nb = self.GX, self.GY, self.NP, self.kb, self.nb
        cx0 = self.cx0
        vv = {
            (i, cy): mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0 + i, cy))
            for i in range(GX)
            for cy in range(GY)
        }
        vv_any = lambda c: mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0 + c[0], c[1]))  # noqa: E731
        fmt = lambda i, dt, page: [ttnn.CBFormatDescriptor(i, dt, page)]  # noqa: E731
        allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx0, 0), ttnn.CoreCoord(cx0 + GX - 1, GY - 1))])
        xb = kb * 2048
        cbs = [
            ttnn.CBDescriptor(
                total_size=self.Kt * 2048, core_ranges=allc, format_descriptors=fmt(0, ttnn.bfloat16, 2048)
            ),
            ttnn.CBDescriptor(
                total_size=self.nst * xb, core_ranges=allc, format_descriptors=fmt(8, ttnn.bfloat16, 2048)
            ),
            ttnn.CBDescriptor(total_size=512, core_ranges=allc, format_descriptors=fmt(10, ttnn.uint32, 512)),
            ttnn.CBDescriptor(
                total_size=(self.nstw + self.q1s) * self.wbmax,
                core_ranges=allc,
                format_descriptors=fmt(9, self.w_dtype, self.wtb),
            ),
        ]
        for pn, cols in sorted(self.groups.items()):
            crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(cx0 + i, 0), ttnn.CoreCoord(cx0 + i, GY - 1)) for i in cols]
            )
            wbg = kb * (pn // NP) * self.wtb
            cbs += [
                ttnn.CBDescriptor(
                    total_size=nb * wbg, core_ranges=crs, format_descriptors=fmt(1, self.w_dtype, self.wtb)
                ),
                ttnn.CBDescriptor(
                    total_size=NP * (pn // NP // self.G) * 2048,
                    core_ranges=crs,
                    format_descriptors=fmt(4, ttnn.bfloat16, 2048),
                ),
            ]
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )  # noqa: E731
        n0s = [sum(self.pns[:i]) for i in range(GX)]
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        kernels = []
        glu_code = {None: 0, "silu": 1, "situ": 2}[self.glu]
        for gi_, (pn, cols) in enumerate(sorted(self.groups.items())):
            w = wl[gi_]
            crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(cx0 + i, 0), ttnn.CoreCoord(cx0 + i, GY - 1)) for i in cols]
            )
            pnp = pn // NP
            ct = (
                [
                    self.Kt,
                    kb,
                    pnp,
                    self.nblk,
                    NP,
                    nb,
                    GX,
                    GY,
                    self.wtb,
                    pnp,
                    self.Nt // self.G,
                    self.MC,
                    self.nst,
                    2,
                    self.q1s,
                    self.nstw,
                    int(bool(self.glu)),
                    self.wbmax,
                    self.xr_need,
                    note_sem,
                ]
                + x_ct
                + y_ct
            )
            b_rt, n_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i in cols:
                for cy in range(GY):
                    lx, rx, ry = vv[(0, cy)].x, vv[(GX - 1, cy)].x, vv[(i, cy)].y
                    ty, by, cxv = vv[(i, 0)].y, vv[(i, GY - 1)].y, vv[(i, cy)].x
                    ba = [
                        int(x.buffer_address()),
                        int(w.buffer_address()),
                        i,
                        cy,
                        lx,
                        rx,
                        ry,
                        ty,
                        by,
                        cxv,
                        0,
                        cols.index(i),
                    ]
                    for q in (0, 1):
                        rd = vv_any(self.reader_of[(i, q)])
                        ba += [rd.x, rd.y]
                    mine = self.by_reader.get((i, cy), {})
                    for q in (0, 1):
                        st_ = mine.get(q)
                        if st_ is None:
                            ba += [0] * 7
                        else:
                            ba += [
                                1,
                                vv[(st_["cx"], 0)].x,
                                int(wl[self.gidx[st_["cx"]]].buffer_address()),
                                self.wbs[self.gidx[st_["cx"]]],
                                st_["bank"],
                                st_["o"],
                                st_["n"],
                            ]
                    ba += [lx, rx]  # (args 30, 31: the row's x senders' NoC x)
                    na = [int(y.buffer_address()), i, cy, n0s[i], rx, lx, ry, by, ty, cxv, 0, 0]
                    st1 = mine.get(1)
                    na += [vv[(st1["cx"], 0)].x, self.wbs[self.gidx[st1["cx"]]], st1["n"]] if st1 else [0, 0, 0]
                    cons = note[cy] if note is not None else []
                    na += [len(cons)]
                    for c_ in cons:
                        v_ = mesh.worker_core_from_logical_core(c_)
                        na += [v_.x, v_.y]
                    b_rt[cx0 + i][cy], n_rt[cx0 + i][cy] = ba, na
                    c_rt[cx0 + i][cy] = []
            comp = ttnn.ComputeConfigDescriptor()
            comp.math_fidelity = self.fid
            comp.fp32_dest_acc_en = True
            comp.dst_full_sync_en = True
            comp.math_approx_mode = False
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=_BRISC,
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=b_rt,
                    config=dm("RISCV_0", ttnn.NOC.NOC_0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_NCRISC,
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=n_rt,
                    config=dm("RISCV_1", ttnn.NOC.NOC_1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=(f"#define L2C_GLU {glu_code}\n" if self.glu else "") + _COMPUTE,
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=[kb, pnp, pnp, self.nblk, NP, self.MC],
                    runtime_args=c_rt,
                    config=comp,
                ),
            ]
        return cbs, kernels

    def semaphores(self, extra_cores=()):
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores() + list(extra_cores)])
        return [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0)
            for i in list(range(10 + self.nb)) + [XREADY, XFREE]
        ]

    def run_chain(self, down, x, wl, wl_down):
        """Test: this engine (gate / up) -> h in DRAM -> down (another Lin2c, same GY / MC, xr_need = self.GX) in one
        program; each of this engine's cores notifies the down engine's x senders of its row once a chunk's h is written.
        """
        assert self.xr_need == 0 and down.xr_need == self.GX and down.GY == self.GY and down.MC == self.MC
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.MC * self.M, self.N // self.G]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([down.MC * down.M, down.N // down.G]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        prog = ttnn.ProgramDescriptor()
        c1, k1 = self.pieces(x, wl, h, note=[down.x_senders(cy) for cy in range(self.GY)], note_sem=XREADY)
        c2, k2 = down.pieces(h, wl_down, y)
        prog.cbs, prog.kernels = c1 + c2, k1 + k2
        prog.semaphores = semaphores(self, down)
        R, C = tuple(self.mesh.shape)
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
        ttnn.generic_op([x] + list(wl) + [h] + list(wl_down) + [y], md)
        ttnn.deallocate(h)
        return y

    def run_alone(self, x, wl):
        """Test: the engine alone in a program (x complete in DRAM: xr_need must be 0)."""
        assert self.xr_need == 0
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.MC * self.M, self.N // self.G]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        prog = ttnn.ProgramDescriptor()
        prog.cbs, prog.kernels = self.pieces(x, wl, y)
        prog.semaphores = self.semaphores()
        R, C = tuple(self.mesh.shape)
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
        ttnn.generic_op([x] + list(wl) + [y], md)
        return y


def semaphores(*engines, extra_cores=()):
    """The semaphore descriptors of several engines (and extra cores) of one program: one per id over all their cores."""
    ids = sorted({i for e in engines for i in list(range(10 + e.nb)) + [XREADY, XFREE]})
    out = []
    for i in ids:
        cores = [c for e in engines if i < 10 + e.nb or i in (XREADY, XFREE) for c in e.cores()] + list(extra_cores)
        out.append(
            ttnn.SemaphoreDescriptor(
                id=i, core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]), initial_value=0
            )
        )
    return out
