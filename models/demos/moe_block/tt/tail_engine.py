# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tail_engine - the fused K3 MoE tail on one program a chip: out [rows, Ho] = rmsnorm(z) @ W_up + se, with z [rows, Hr]
the mesh row's gathered latent rows (chunk q: TP chip q's owned rows), W_up this chip's column slice [Hr, Ho] and se the
shared expert's output in the block's layout.

The norm is folded: its weight g into W' = diag(g) W_up (upload), its 1 / rms per row applied to the matmul's fp32 sums
(out = diag(s) (z @ W') + se, s = rsqrt(mean(z^2) + eps)) - the matmul never waits for whole rows, only s does.
  grid    GX x GY cores, GY = 2 RT (RT row tiles a chunk); core (cx, cy): chunk-local row tile r = cy % RT, K half
          h = cy // RT (z's K tiles kh(h)), output tile columns n0(cx) .. + pn(cx) (column_split of Ho / 32 over GX);
          its partner (cx, other half) has the same columns and the other K half
  W'      resident: core (cx, cy) holds [K half h][pn(cx)] (bf8); the RT cores of a half column read their K ranges of
          it from DRAM (one burst each: bank-sharded regions) and multicast them down the half column (NOC0)
  z       chunk by chunk, a ring of NSLOT chunk slots [K half] a core: a row's K half split into shares (by column),
          each read into its core's landing region (per chunk, never reused in a call) and multicast to the row - one
          share of a chunk at a time, in column order (the chunk's arrival count the token: concurrent row multicasts
          reserve crossing paths and stall each other) - once every core of the row freed the slot (FREE credits)
  s       each core's squares of its share accumulated elementwise in DEST (one fp32 E tile), unicast to the row; the
          row's E tiles @ ones (one HiFi4 matmul accumulated over them) = the half's sum of squares (every column the
          row's); the partners swap theirs: s = rsqrt((sum h0 + sum h1) / Hr + eps)
  out     per chunk all of the half's K in DEST (fp32): the columns the partner finishes go to it with the half's sum of
          squares (XSEND -> its XRECV, a column unicast); each core finishes about half its columns: (mine + the
          partner's) x s + se, bf16 (NCRISC)
Every multicast is row-local or column-local on its own NoC (rows NOC1, columns NOC0). Chunk order: `order` (the TP
chips' row blocks in processing order). Semaphores are the program's (zero each launch): every producer is on this
chip."""

import math
import struct

import torch
from ttnn.bringup.combine_rf.lin2d import column_split

import ttnn
from models.demos.moe_block.tt.se_engine3 import _ring_pages

CB_W, CB_A, CB_PART, CB_PMY, CB_ONES, CB_ACCM, CB_XSEND, CB_SQM, CB_XRECV, CB_SE, CB_OUT, CB_L0, CB_RM = range(13)
XT = 4  # (a chunk's exchange: up to 3 partial tiles + the half's sum of squares)
ARENA_MAX = 1400 * 1024

_CT = r"""
constexpr uint32_t KH = get_compile_time_arg_val(0), PNMAX = get_compile_time_arg_val(1), RT = get_compile_time_arg_val(2);
constexpr uint32_t NCK = get_compile_time_arg_val(3), GX = get_compile_time_arg_val(4), WTB = get_compile_time_arg_val(5);
constexpr uint32_t NSLOT = get_compile_time_arg_val(6), KT = get_compile_time_arg_val(7), XT = get_compile_time_arg_val(8);
// (SA0 + c: chunk c's shares in; SP0 + c: its E tiles; SX: the partner's exchanges (cumulative: it sends in chunk
// order); SW: W' K ranges in, a bit each (1 << r added by range r's reader). My share of chunk c landed: the global
// semaphore at sl + 16 c (chunk 0: my BRISC; the others: the fabric - the upstream port's last packet of the share))
constexpr uint32_t SA0 = 0, SP0 = NCK, SX = 2 * NCK, SFREE = 2 * NCK + 1, SW = 2 * NCK + 2;
constexpr uint32_t CB_W = 0, CB_A = 1, CB_PART = 2, CB_PMY = 3, CB_ONES = 4, CB_XSEND = 6, CB_XRECV = 8, CB_SE = 9;
constexpr uint32_t CB_OUT = 10, CB_L0 = 11, CB_RM = 12;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
"""

_BRISC = (
    '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n#include "tools/profiler/kernel_profiler.hpp"\n'
    + _CT
    + r"""
constexpr auto z_args = TensorAccessorArgs<9>();  // (the combine's output: this chip's owned rows, row-major)
constexpr auto s_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
// NOC0: every DRAM read (my W' K range, my z shares, my se tiles) and the W' multicast down my half column
void kernel_main() {
    size_t a = 0;
    const uint32_t r = get_arg_val<uint32_t>(a++), pn = get_arg_val<uint32_t>(a++), kh0 = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++), wbank = get_arg_val<uint32_t>(a++), woff = get_arg_val<uint32_t>(a++);
    const uint32_t wk0 = get_arg_val<uint32_t>(a++), wk1 = get_arg_val<uint32_t>(a++);
    const uint32_t mx = get_arg_val<uint32_t>(a++), yt = get_arg_val<uint32_t>(a++), yb = get_arg_val<uint32_t>(a++);
    const uint32_t z_addr = get_arg_val<uint32_t>(a++), ak0 = get_arg_val<uint32_t>(a++), ak1 = get_arg_val<uint32_t>(a++);
    const uint32_t se_addr = get_arg_val<uint32_t>(a++), NTO = get_arg_val<uint32_t>(a++), f0 = get_arg_val<uint32_t>(a++);
    const uint32_t fn = get_arg_val<uint32_t>(a++), land = get_arg_val<uint32_t>(a++), SHMAX = get_arg_val<uint32_t>(a++);
    const uint32_t sl = get_arg_val<uint32_t>(a++);
    const uint32_t ord = a;
    const auto zg = TensorAccessor(z_args, z_addr, KT * 64);  // (a page: one latent row)
    const auto sg = TensorAccessor(s_args, se_addr, 2048);
    cb_reserve_back(CB_ONES, 1);  // (the all-ones tile of the squares' row sums)
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONES));
        for (uint32_t i = 0; i < 512; ++i) o[i] = 0x3F803F80u;
    }
    cb_push_back(CB_ONES, 1);
    {  // my share of my own rows, row-major (the combine's output: the 32 rows of my row tile r, K tiles kh0 + ak0 ..):
       // the compute tilizes it into my landing slot 0 (the other chunks' shares land there by the fabric)
        const uint32_t n = ak1 - ak0;
        cb_reserve_back(CB_RM, n);
        const uint32_t dst = get_write_ptr(CB_RM);
        for (uint32_t i = 0; i < 32; ++i)
            noc_async_read(zg.get_noc_addr(r * 32 + i, (kh0 + ak0) * 64), dst + i * n * 64, n * 64);
        noc_async_read_barrier();
        cb_push_back(CB_RM, n);
    }
    {  // W': my K range of my half column's [K half][pn] (one contiguous bank region: one burst), then down the half column
        DeviceZoneScopedN("TL-WLOAD");
        const uint32_t src = get_write_ptr(CB_W) + wk0 * pn * WTB, bytes = (wk1 - wk0) * pn * WTB;
        noc_async_read(get_noc_addr_from_bank_id<true>(wbank, w_addr + woff), src, bytes);
        noc_async_read_barrier();
        if (RT > 1) {
            if (bytes) noc_async_write_multicast(src, get_noc_multicast_addr(mx, yt, mx, yb, src), bytes, RT - 1);
            noc_async_writes_flushed();
            noc_semaphore_inc_multicast(get_noc_multicast_addr(mx, yt, mx, yb, get_semaphore(SW)), 1u << r, RT - 1);
        }
        noc_semaphore_inc(get_noc_addr(get_semaphore(SW)), 1u << r);
    }
    for (uint32_t c = 0; c < NCK; ++c) {  // my se tiles (the columns I finish)
        const uint32_t R = get_arg_val<uint32_t>(ord + c) * RT + r;
        cb_reserve_back(CB_SE, fn);
        const uint32_t dst = get_write_ptr(CB_SE);
        for (uint32_t j = 0; j < fn; ++j) noc_async_read(sg.get_noc_addr(R * NTO + f0 + j), dst + j * 2048, 2048);
        noc_async_read_barrier();
        cb_push_back(CB_SE, fn);
    }
    noc_async_atomic_barrier();
}
"""
)

_NCRISC = (
    '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n#include "tools/profiler/kernel_profiler.hpp"\n'
    + _CT
    + r"""
constexpr auto o_args = TensorAccessorArgs<9>();
// NOC1: my z share's and my E tile's way to my row, the partner exchange, the credits, the CB pushes, the output writes
void kernel_main() {
    size_t a = 0;
    const uint32_t r = get_arg_val<uint32_t>(a++), pn = get_arg_val<uint32_t>(a++), f0 = get_arg_val<uint32_t>(a++);
    const uint32_t fn = get_arg_val<uint32_t>(a++), ak0 = get_arg_val<uint32_t>(a++), ak1 = get_arg_val<uint32_t>(a++);
    const uint32_t lx = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), cx = get_arg_val<uint32_t>(a++);
    const uint32_t o_addr = get_arg_val<uint32_t>(a++), NTO = get_arg_val<uint32_t>(a++);
    const uint32_t land = get_arg_val<uint32_t>(a++), SHMAX = get_arg_val<uint32_t>(a++), ring = get_arg_val<uint32_t>(a++);
    const uint32_t part = get_arg_val<uint32_t>(a++), xrecv = get_arg_val<uint32_t>(a++);
    const uint32_t mx = get_arg_val<uint32_t>(a++), my = get_arg_val<uint32_t>(a++), py = get_arg_val<uint32_t>(a++);
    const uint32_t sl = get_arg_val<uint32_t>(a++);
    const uint32_t ord = a, xs = ord + NCK, wkb = xs + GX;  // (chunk order; my row's NoC x by column; W' K range bounds)
    auto landed = [&](uint32_t c) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + 16 * c); };
    // (the landed words only count up - every landing of every call; this call's: >= my call count, kept after them -
    // so a share read by both me and a relaying port needs no re-arm either of us could race)
    volatile tt_l1_ptr uint32_t* epw = landed(NCK);
    const uint32_t ep = *epw + 1;
    *epw = ep;
    const auto og = TensorAccessor(o_args, o_addr, 2048);
    auto row = [&](uint32_t addr) { return get_noc_multicast_addr(rx, my, lx, my, addr); };  // (NOC1: right end first)
    uint32_t cs = 0, wr = 0;
    bool own = false;
    auto pump = [&]() {
        invalidate_l1_cache();
        if (!own && cb_pages_available_at_front(CB_L0, ak1 - ak0)) {  // (my own share tilized: landed, chunk 0)
            noc_semaphore_inc(get_noc_addr(sl), 1);
            own = true;
        }
        while (wr < RT && (*sem(SW) >> wr & 1)) {  // (W' K range wr landed: to the compute, in K order)
            const uint32_t n = (get_arg_val<uint32_t>(wkb + wr + 1) - get_arg_val<uint32_t>(wkb + wr)) * pn;
            cb_reserve_back(CB_W, n);
            cb_push_back(CB_W, n);
            ++wr;
        }
        // my share of chunk cs: loaded (BRISC), its slot free on the whole row, my turn in the row (column order)
        if (cs >= NCK || *landed(cs) < ep || (cs >= NSLOT && *sem(SFREE) < GX * (cs - NSLOT + 1))) return;
        if (*sem(SA0 + cs) < cx) return;
        DeviceZoneScopedN("TL-SHARE");
        const uint32_t src = land + cs * SHMAX * 2048, dst = ring + ((cs % NSLOT) * KH + ak0) * 2048;
        noc_async_write_multicast_loopback_src(src, row(dst), (ak1 - ak0) * 2048, GX);
        noc_async_write_barrier();
        if (GX > 1) noc_semaphore_inc_multicast(row(get_semaphore(SA0 + cs)), 1, GX - 1);
        noc_semaphore_inc(get_noc_addr(get_semaphore(SA0 + cs)), 1);
        ++cs;
    };
    auto wait = [&](volatile tt_l1_ptr uint32_t* w, uint32_t n) {
        while (*w < n) pump();
    };
    auto wait_front = [&](uint32_t cb, uint32_t n) {
        while (!cb_pages_available_at_front(cb, n)) pump();
        cb_wait_front(cb, n);
    };
    auto write_out = [&](uint32_t c) {  // my finished columns of chunk c; then my ring slot of c is free (FREE)
        const uint32_t R = get_arg_val<uint32_t>(ord + c) * RT + r;
        wait_front(CB_OUT, fn);
        const uint32_t src = get_read_ptr(CB_OUT);
        for (uint32_t j = 0; j < fn; ++j) noc_async_write(src + j * 2048, og.get_noc_addr(R * NTO + f0 + j), 2048);
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, fn);
        if (GX > 1) noc_semaphore_inc_multicast(row(get_semaphore(SFREE)), 1, GX - 1);
        noc_semaphore_inc(get_noc_addr(get_semaphore(SFREE)), 1);
    };
    for (uint32_t c = 0; c < NCK; ++c) {
        {
            DeviceZoneScopedN("TL-AWAIT");
            wait(sem(SA0 + c), GX);
        }
        cb_reserve_back(CB_A, KH);
        cb_push_back(CB_A, KH);
        {  // my E tile to every core of my row (unicasts: 4 KB - no multicast path to reserve)
            wait_front(CB_PMY, 1);
            const uint32_t src = get_read_ptr(CB_PMY), dst = part + (c * GX + cx) * 4096;
            for (uint32_t i = 0; i < GX; ++i) noc_async_write(src, get_noc_addr(get_arg_val<uint32_t>(xs + i), my, dst), 4096);
            noc_async_write_barrier();
            for (uint32_t i = 0; i < GX; ++i)
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xs + i), my, get_semaphore(SP0 + c)), 1);
            cb_pop_front(CB_PMY, 1);
        }
        {
            DeviceZoneScopedN("TL-PWAIT");
            wait(sem(SP0 + c), GX);
        }
        cb_reserve_back(CB_PART, GX);
        cb_push_back(CB_PART, GX);
        {  // the partner's columns' partials + my half's sum of squares to its XRECV slot of chunk c (a column unicast)
            wait_front(CB_XSEND, XT);
            noc_async_write(get_read_ptr(CB_XSEND), get_noc_addr(mx, py, xrecv + c * XT * 4096), XT * 4096);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(mx, py, get_semaphore(SX)), 1);
            cb_pop_front(CB_XSEND, XT);
        }
        {
            DeviceZoneScopedN("TL-XWAIT");
            wait(sem(SX), c + 1);
        }
        cb_reserve_back(CB_XRECV, XT);
        cb_push_back(CB_XRECV, XT);
        if (c > 0) write_out(c - 1);
    }
    write_out(NCK - 1);
    while (wr < RT || cs < NCK || !own) pump();
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
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/tilize.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif
constexpr uint32_t KH = get_compile_time_arg_val(0), NCK = get_compile_time_arg_val(1), GX = get_compile_time_arg_val(2);
constexpr uint32_t INV_HR = get_compile_time_arg_val(3), NORM_EPS = get_compile_time_arg_val(4), XT = get_compile_time_arg_val(5);
constexpr uint32_t CB_W = 0, CB_A = 1, CB_PART = 2, CB_PMY = 3, CB_ONES = 4, CB_ACCM = 5, CB_XSEND = 6, CB_SQM = 7;
constexpr uint32_t CB_XRECV = 8, CB_SE = 9, CB_OUT = 10, CB_L0 = 11, CB_RM = 12;
// (the squares' row sums at HiFi4 whatever the kernel's fidelity: HiFi2 truncates them ~0.8% low)
ALWI void mm4_init(uint32_t in0, uint32_t in1) {
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, 1, 1, 1)));
    MATH((llk_math_matmul_init<MathFidelity::HiFi4, MM_THROTTLE>(in0, in1, 0, 1, 1)));
}
ALWI void mm4(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, 1, 1, 1)));
    MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(idst, 1, 1)));
}
void kernel_main() {
    // args: pn, my finished columns' offset in mine (fo) and count (fn), my share [sk0, sk1), the W' K range bounds
    const uint32_t pn = get_arg_val<uint32_t>(0), fo = get_arg_val<uint32_t>(1), fn = get_arg_val<uint32_t>(2);
    const uint32_t sk0 = get_arg_val<uint32_t>(3), sk1 = get_arg_val<uint32_t>(4);
    uint32_t wr = 0, wnext = 0;  // (args 5 ..: the W' K range bounds 0, .., KH)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_A, CB_W, CB_ACCM);
    {  // my share of my own rows tilized into my landing slot 0 (no separate tilize: the combine's rows read once)
        DeviceZoneScopedN("TL-TILIZE");
        const uint32_t n = sk1 - sk0;
        cb_wait_front(CB_RM, n);
        cb_reserve_back(CB_L0, n);
        reconfig_data_format_srca(CB_RM);
        pack_reconfig_data_format(CB_L0);
        tilize_init(CB_RM, n, CB_L0);
        tilize_block(CB_RM, n, CB_L0);
        tilize_uninit(CB_RM, CB_L0);
        cb_push_back(CB_L0, n);
        cb_pop_front(CB_RM, n);
    }
    cb_wait_front(CB_ONES, 1);
    for (uint32_t c = 0; c < NCK; ++c) {
        cb_wait_front(CB_A, KH);
        {  // my share's squares accumulated elementwise in DEST (fp32, SFPU): E = sum_k z_k * z_k
            DeviceZoneScopedN("TL-SQ");
            cb_reserve_back(CB_PMY, 1);
            reconfig_data_format_srca(CB_A);
            pack_reconfig_data_format(CB_PMY);
            tile_regs_acquire();
            for (uint32_t k0 = sk0; k0 < sk1; k0 += 7) {  // (DEST 1 .. 7 a batch; 0 the sum)
                const uint32_t n = sk1 - k0 < 7 ? sk1 - k0 : 7;
                copy_init(CB_A);
                for (uint32_t i = 0; i < n; ++i) copy_tile(CB_A, k0 + i, 1 + i);
                mul_binary_tile_init();
                for (uint32_t i = 0; i < n; ++i) mul_binary_tile(1 + i, 1 + i, 1 + i);
                if (k0 == sk0) copy_dest_values<DataFormat::Float32>(1, 0);
                add_binary_tile_init();
                for (uint32_t i = k0 == sk0 ? 1 : 0; i < n; ++i) add_binary_tile(0, 1 + i, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_PMY);
            tile_regs_release();
            cb_push_back(CB_PMY, 1);
        }
        {  // all of my half's K in DEST: my finished columns to ACCM, the partner's to XSEND (fp32)
            DeviceZoneScopedN("TL-MM");
            reconfig_data_format(CB_W, CB_A);
            matmul_block_init(CB_A, CB_W, 0, pn, 1, KH);
            cb_reserve_back(CB_ACCM, fn);
            cb_reserve_back(CB_XSEND, XT);
            tile_regs_acquire();
            for (uint32_t k = 0; k < KH; ++k) {
                if (c == 0 && k == wnext) {  // (chunk 0: W' K range by range, as each lands)
                    wnext = get_arg_val<uint32_t>(5 + ++wr);
                    cb_wait_front(CB_W, wnext * pn);
                }
                matmul_block(CB_A, CB_W, k, k * pn, 0, false, pn, 1, KH);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_ACCM);
            for (uint32_t j = 0; j < fn; ++j) pack_tile<true>(fo + j, CB_ACCM, j);
            for (uint32_t j = 0, i = 0; j < pn; ++j)
                if (j < fo || j >= fo + fn) pack_tile<true>(j, CB_XSEND, i++);
            tile_regs_release();
            cb_push_back(CB_ACCM, fn);
            cb_pop_front(CB_A, KH);
        }
        {  // my half's sum of squares (the row's E tiles @ ones, HiFi4, accumulated in DEST): to the partner and mine
            DeviceZoneScopedN("TL-S");
            cb_wait_front(CB_PART, GX);
            cb_reserve_back(CB_SQM, 1);
            reconfig_data_format(CB_ONES, CB_PART);
            mm4_init(CB_PART, CB_ONES);
            tile_regs_acquire();
            for (uint32_t i = 0; i < GX; ++i) mm4(CB_PART, CB_ONES, i, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_XSEND);
            pack_tile<true>(0, CB_XSEND, XT - 1);
            pack_tile(0, CB_SQM);
            tile_regs_release();
            cb_push_back(CB_XSEND, XT);
            cb_push_back(CB_SQM, 1);
            cb_pop_front(CB_PART, GX);
        }
        {  // out = (mine + the partner's) x s + se, s = rsqrt((my half's + its half's sum of squares) / Hr + eps)
            DeviceZoneScopedN("TL-OUT");
            cb_wait_front(CB_XRECV, XT);
            cb_wait_front(CB_SQM, 1);
            cb_wait_front(CB_ACCM, fn);
            cb_wait_front(CB_SE, fn);
            cb_reserve_back(CB_OUT, fn);
            tile_regs_acquire();
            // (DEST: s 0, mine 1 .., the partner's sum of squares 4, its partials 5 .., then se 4 ..)
            reconfig_data_format_srca(CB_SQM);
            copy_init(CB_SQM);
            copy_tile(CB_SQM, 0, 0);
            reconfig_data_format_srca(CB_XRECV);
            copy_init(CB_XRECV);
            copy_tile(CB_XRECV, XT - 1, 4);
            for (uint32_t j = 0; j < fn; ++j) copy_tile(CB_XRECV, j, 5 + j);
            add_binary_tile_init();
            add_binary_tile(0, 4, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, INV_HR);
            add_unary_tile(0, NORM_EPS);
            rsqrt_tile_init();
            rsqrt_tile(0);
            reconfig_data_format_srca(CB_ACCM);
            copy_init(CB_ACCM);
            for (uint32_t j = 0; j < fn; ++j) copy_tile(CB_ACCM, j, 1 + j);
            add_binary_tile_init();
            for (uint32_t j = 0; j < fn; ++j) add_binary_tile(1 + j, 5 + j, 1 + j);
            mul_binary_tile_init();
            for (uint32_t j = 0; j < fn; ++j) mul_binary_tile(1 + j, 0, 1 + j);
            reconfig_data_format_srca(CB_SE);
            copy_init(CB_SE);
            for (uint32_t j = 0; j < fn; ++j) copy_tile(CB_SE, j, 4 + j);
            add_binary_tile_init();
            for (uint32_t j = 0; j < fn; ++j) add_binary_tile(1 + j, 4 + j, 1 + j);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            for (uint32_t j = 0; j < fn; ++j) pack_tile(1 + j, CB_OUT);
            tile_regs_release();
            cb_push_back(CB_OUT, fn);
            cb_pop_front(CB_XRECV, XT);
            cb_pop_front(CB_SQM, 1);
            cb_pop_front(CB_ACCM, fn);
            cb_pop_front(CB_SE, fn);
        }
    }
}
"""

# ---- the latent all-gather's ports (one per direction and link, on the column left of the engine) ----
# A port (d, l) sends half l of the K tiles of its blocks (own first, then the ones it relays) to the next chip in
# direction d, straight into the engine cores' landing regions there: share by share (row tile r, column cx: K tiles
# kh(l) + ak(cx)), packets of up to RP tiles; a share's last packet a fused write + increment of the owner's landed word
# of the block's chunk position there. A relayed share is read from this chip's owner once its landed word says so.
_PORT_CT = r"""
#include "api/debug/dprint.h"
constexpr uint32_t RT = get_compile_time_arg_val(0), GX = get_compile_time_arg_val(1), RP = get_compile_time_arg_val(2);
constexpr uint32_t SHMAX = get_compile_time_arg_val(3), KT = get_compile_time_arg_val(4), CBP = 0, CBM = 1;
constexpr uint32_t PG = get_compile_time_arg_val(5);  // (packets a group: the reader's barrier, the sender's flush)
// the shares of a unit, in the walk both kernels follow: part 0 every column, 1 / 2 the first / second half of them
template <typename F>
FORCE_INLINE void walk(uint32_t part, uint32_t ak, F&& f) {
    for (uint32_t r = 0; r < RT; ++r)
        for (uint32_t cx = 0; cx < GX; ++cx) {
            if ((part == 1 && cx >= GX / 2) || (part == 2 && cx < GX / 2)) continue;
            const uint32_t k0 = get_arg_val<uint32_t>(ak + 2 * cx), k1 = get_arg_val<uint32_t>(ak + 2 * cx + 1);
            for (uint32_t t0 = k0; t0 < k1; t0 += RP) f(r, cx, k0, t0, (k1 - t0 < RP ? k1 - t0 : RP), t0 + RP >= k1);
        }
}
"""

_PORT_READER = (
    '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n'
    + _PORT_CT
    + r"""
// NoC0: my own block's shares from DRAM (y_own), the relayed ones from this chip's owners (once landed), into the CB
void kernel_main() {
    size_t a = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(a++), kh0 = get_arg_val<uint32_t>(a++), land = get_arg_val<uint32_t>(a++);
    const uint32_t sl = get_arg_val<uint32_t>(a++), poll = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++);
    const uint32_t epa = get_arg_val<uint32_t>(a++);  // (my call count: a landed word of this call is >= it)
    volatile tt_l1_ptr uint32_t* epw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(epa);
    const uint32_t ep = *epw + 1;
    *epw = ep;
    const uint32_t units = a, ak = units + n, xs = ak + 2 * GX, ys = xs + GX;  // (unit: part | position here << 8)
#ifdef TL_DPRINT
    DPRINT("PR start n {}\n", n);
#endif
    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll);
    // a group: PG packets' slots reserved, read, one barrier, pushed whole (padded) with its packet count in CBM - a
    // relayed share not landed yet ends the group early (never hold read packets while waiting)
    uint32_t b = 0, wp = 0;
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_reserve_back(CBM, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CBM)) = b;
        cb_push_back(CBM, 1);
        cb_push_back(CBP, PG * RP);
        b = 0;
    };
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(units + u), part = w & 0xFF, pos = w >> 8;
        walk(part, ak, [&](uint32_t r, uint32_t cx, uint32_t k0, uint32_t t0, uint32_t m, bool) {
            const uint32_t ox = get_arg_val<uint32_t>(xs + cx), oy = get_arg_val<uint32_t>(ys + r);
            if (t0 == k0) {  // (the share landed / tilized at its owner here: its landed word, over the NoC)
                auto in = [&]() {
                    noc_async_read(get_noc_addr(ox, oy, sl + 16 * pos), poll, 4);
                    noc_async_read_barrier();
                    invalidate_l1_cache();  // (the copy just landed: not the cached word)
                    return *pw >= ep;
                };
                if (!in()) {
                    if (b > 0) flush();
                    while (!in()) {
                    }
                }
            }
            if (b == 0) {
                cb_reserve_back(CBP, PG * RP);
                wp = get_write_ptr(CBP);
            }
            const uint32_t dst = wp + b * RP * 2048;
            noc_async_read(get_noc_addr(ox, oy, land + (pos * SHMAX + t0 - k0) * 2048), dst, m * 2048);
            if (++b == PG) flush();
        });
    }
    if (b > 0) flush();
}
"""
)

_PORT_SENDER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
"""
    + _PORT_CT
    + r"""
// NoC1: the fence (my ready to the port that writes into this chip; the next chip's ready before I write there), then
// every packet of my units into the next chip's owners
void kernel_main() {
    size_t a = 0;
    const uint32_t land = get_arg_val<uint32_t>(a++), sl = get_arg_val<uint32_t>(a++), ready = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = get_arg_val<uint32_t>(a++), chip_id = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++), units = a, ak = units + n, xs = ak + 2 * GX, ys = xs + GX;
    a = ys + RT;
    if (n == 0 && !send_ready) return;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
    conn.open();
    if (send_ready) {  // (this program runs: everything before it on this chip is done - my landing regions are free)
        hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    }
#ifdef TL_DPRINT
    DPRINT("PS start n {} send_ready {} chip {}\n", n, send_ready, (uint32_t)chip_id);
#endif
    if (n > 0) {
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
        noc_semaphore_wait_min(rd, 1);
#ifdef TL_DPRINT
        DPRINT("PS ready\n");
#endif
        noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);  // (re-arm: one ready a call)
        noc_async_atomic_barrier();
    }
    // the reader's groups: PG slots each, its packet count in CBM; a header a slot, one flush a group
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[PG];
    hdrs[0] = hdr;
    for (uint32_t h = 1; h < PG; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdrs[h], chip_id, mesh_id);
    }
    uint32_t gi = 0, gn = 0;
    auto next_group = [&]() {
        if (gn > 0) {
            noc_async_writes_flushed();  // (the group's packets and headers have left L1)
            cb_pop_front(CBP, PG * RP);
        }
        cb_wait_front(CBM, 1);
        gn = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CBM));
        cb_pop_front(CBM, 1);
        cb_wait_front(CBP, PG * RP);
        gi = 0;
    };
    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t w = get_arg_val<uint32_t>(units + u), part = w & 0xFF, pos = w >> 8;  // (pos: there)
        walk(part, ak, [&](uint32_t r, uint32_t cx, uint32_t k0, uint32_t t0, uint32_t m, bool last) {
            if (gi == gn) next_group();
            const uint32_t ox = get_arg_val<uint32_t>(xs + cx), oy = get_arg_val<uint32_t>(ys + r);
            const uint64_t dst = get_noc_addr(ox, oy, land + (pos * SHMAX + t0 - k0) * 2048, 0);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hd = hdrs[gi];
            if (last) {
                hd->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(ox, oy, sl + 16 * pos, 0), 1, true}, m * 2048);
            } else {
                hd->to_noc_unicast_write(NocUnicastCommandHeader{dst}, m * 2048);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(get_read_ptr(CBP) + gi * RP * 2048, m * 2048, reinterpret_cast<uint32_t>(hd));
            ++gi;
        });
    }
    if (gn > 0) {
        noc_async_writes_flushed();
        cb_pop_front(CBP, PG * RP);
    }
    conn.close();
}
"""
)


def chunk_order(p, G, ring):
    """Chip p's chunk positions -> TP positions q (the plan's "out" index): own first, then by hop distance (ties: the
    previous chip first) - roughly the order the gather lands them."""
    dist = lambda q: min((q - p) % G, (p - q) % G) if ring else abs(q - p)  # noqa: E731
    return sorted(range(G), key=lambda q: (dist(q), (q - p) % G != G - 1, q))


def share_in_part(cx, GX, part):
    """A share (column cx) belongs to a unit's part: 0 every column, 1 / 2 the first / second half (a ring's opposite
    block split between the directions) - the ports' walk."""
    return part == 0 or (part == 1 and cx < GX // 2) or (part == 2 and cx >= GX // 2)


def _split(n, parts):
    """n into parts contiguous ranges, as even as possible: [(k0, k1)]."""
    base, rem = divmod(n, parts)
    out, k = [], 0
    for i in range(parts):
        w = base + (1 if i < rem else 0)
        out.append((k, k + w))
        k += w
    return out


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


class TailEngine:
    """K3's tail a chip (module docstring): z [rows, Hr] (the latent rows of the mesh row, chunk q = TP chip q's) ->
    rmsnorm(z) @ W_up[:, chip's columns] + se [rows, Ho]."""

    def __init__(
        self,
        mesh_device,
        *,
        rows,
        Hr,
        Ho,
        chunks,
        eps,
        GX=None,
        GY=10,
        cx0=0,
        NSLOT=None,
        fidelity=ttnn.MathFidelity.HiFi2,
        w_dtype=ttnn.bfloat8_b,
        topology=ttnn.Topology.Linear,
        num_links=2,
    ):
        self.mesh = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        GX = GX or grid.x - 1  # (the last column: the gather's ports - step B)
        # (z's ring: a slot a chunk by default - no slot reused, no FREE credits on the way)
        NSLOT = NSLOT or chunks
        self.GX, self.GY, self.cx0, self.NSLOT = GX, GY, cx0, NSLOT
        self.rows, self.Hr, self.Ho, self.NCK, self.eps = rows, Hr, Ho, chunks, eps
        self.KT, self.NTO = Hr // 32, Ho // 32
        assert rows % (32 * chunks) == 0
        self.RT = rows // 32 // chunks
        assert GY == 2 * self.RT, f"GY {GY}: two K halves of a chunk's {self.RT} row tiles"
        self.kh = _split(self.KT, 2)  # (the halves' K ranges)
        self.KH = self.kh[0][1]
        assert self.kh[1][1] - self.kh[1][0] == self.KH, "equal K halves"
        self.pn = column_split(self.NTO, GX, 1)
        self.n0 = [sum(self.pn[:cx]) for cx in range(GX)]
        self.PNMAX = max(self.pn)
        assert self.PNMAX <= 7, "a chunk's pn fp32 tiles in DEST"
        # (the columns a core finishes: half 0 the first ceil(pn / 2), half 1 the rest - at most XT - 1 either way)
        self.fin = [[(0, -(-p // 2)), (-(-p // 2), p - -(-p // 2))] for p in self.pn]
        assert all(fn <= XT - 1 and p - fn <= XT - 1 for p, f in zip(self.pn, self.fin) for _, fn in f)
        self.FNS = sorted({fn for f in self.fin for _, fn in f})
        lcm = math.lcm(*self.FNS)
        self.FNR = lcm * -(-2 * max(self.FNS) // lcm)  # (the finished columns' FIFOs: a multiple of every core's)
        self.wk = _split(self.KH, self.RT)  # (a half column's W' K ranges, by row tile)
        self.ak = _split(self.KH, GX)  # (a row's z shares, by column: each core also squares its share)
        self.SHMAX = max(k1 - k0 for k0, k1 in self.ak)
        self.fid, self.w_dtype = fidelity, w_dtype
        self.wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[w_dtype]
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        off, o = {}, 0
        for k, v in [
            ("w", self.KH * self.PNMAX * self.wtb),
            ("ring", NSLOT * self.KH * 2048),
            ("land", chunks * self.SHMAX * 2048),
            ("part", chunks * GX * 4096),
            ("pmy", 4096),
            ("ones", 2048),
            ("accm", self.FNR * 4096),
            ("xsend", 2 * XT * 4096),
            ("sqm", 2 * 4096),
            ("xrecv", chunks * XT * 4096),
            ("se", self.FNR * 2048),
            ("out", self.FNR * 2048),
            ("rm", self.SHMAX * 2048),
        ]:
            off[k], o = o, al(o + v)
        assert o <= ARENA_MAX, f"tail_engine: arena {o >> 10} KB > {ARENA_MAX >> 10} KB"
        self.off, self.arena_bytes = off, o
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx0, 0), ttnn.CoreCoord(cx0 + GX - 1, GY - 1))])
        self._gather_plan(topology, num_links)
        self.arena = None
        self.dbg = "#define TL_DPRINT\n" if __import__("os").environ.get("TL_DPRINT") else ""  # (probe)
        self.keep_arena = False
        self.state = dict(key=None)

    # ---- the gather's plan ----
    def _gather_plan(self, topology, num_links):
        """The latent all-gather over TP into the engines' landing regions: fabric_all_gather's plan (line / ring,
        balanced ring: the opposite block half each way) with its ports on the column right of the engine; link l
        carries K half l (engine rows l RT ..). Per chip its chunk order: own block first, then by hop distance."""
        from ttnn.bringup.combine_rf.fabric_all_gather_py import _plan

        assert num_links == 2, "a link a K half"
        mesh, GX, GY, cx0 = self.mesh, self.GX, self.GY, self.cx0
        col = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx0 + GX, 0), ttnn.CoreCoord(cx0 + GX, GY - 1))])
        # (the ports' column has no row with room for fabric_all_gather's placement probe: placed down it - fwd above,
        # bwd below, a link apart)
        rows = [GY // 10, 3 * GY // 10, 6 * GY // 10, 8 * GY // 10]
        place = {
            (0, d, l): ttnn.CoreCoord(cx0 + GX, rows[2 * i + l]) for i, d in enumerate(("fwd", "bwd")) for l in range(2)
        }
        self.chips, self.ring = _plan(  # (not the memoized plan: a placement dict is no key)
            mesh,
            cluster_axis=1,
            topology=topology,
            num_links=num_links,
            placement=place,
            balance=True,
            sub_core_grid=col,
        )
        G = self.NCK
        for coord, ch in self.chips.items():
            ch["order"] = chunk_order(ch["rings"][0]["p"], G, self.ring)
        self.port_xy = sorted({(c.x, c.y) for ch in self.chips.values() for c in ch["ports"].values()})
        self.port_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in self.port_xy]
        )
        # (the fence: a port's ready counter - persistent, re-armed by its waiter; the engine cores' landed words: a
        # word a chunk position, 16 B apart, re-armed by their NCRISC)
        self.ready = ttnn.create_global_semaphore(mesh, self.port_crs, 0)
        # (+ the engine NCRISC's call count after them; the ports' readers keep theirs in port_epoch)
        self.landed = ttnn.from_torch(
            torch.zeros(GX * GY, 4 * (G + 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.crs, [1, 4 * (G + 1)], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.port_epoch = ttnn.from_torch(
            torch.zeros(len(self.port_xy), 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.port_crs, [1, 4], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.RP = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // 2048)

    # ---- weights ----
    def upload(self, w_up_cols, g):
        """w_up_cols: per device (row-major mesh order) its [Hr, Ho] column slice of W_up^T; g [Hr] the norm weight
        (folded: W' = diag(g) W). Bank-sharded: region (h, cx, r) = the K range r of half h of column cx's [KH][pn]
        tiles - the core that reads it, one burst."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import _bank_sharded

        self.nbank = self.mesh.dram_grid_size().x
        self.wreg = max(k1 - k0 for k0, k1 in self.wk) * self.PNMAX  # (tiles a region, padded)
        devs = []
        for w in w_up_cols:
            wf = w.float() * g.float().view(-1, 1)
            regions = []
            for h0, h1 in self.kh:
                for cx in range(self.GX):
                    pn = self.pn[cx]
                    blk = wf[h0 * 32 : h1 * 32, self.n0[cx] * 32 : (self.n0[cx] + pn) * 32]
                    tiles = blk.reshape(self.KH, 32, pn, 32).permute(0, 2, 1, 3).reshape(self.KH, pn, 32, 32)
                    for k0, k1 in self.wk:
                        reg = tiles[k0:k1].reshape(-1, 32, 32)
                        pad = self.wreg - reg.shape[0]
                        regions.append(torch.cat([reg, torch.zeros(pad, 32, 32)]) if pad else reg)
            devs.append(regions)
        return _bank_sharded(devs, self.nbank, self.w_dtype, self.mesh)

    def _region(self, h, cx, r):
        return (h * self.GX + cx) * self.RT + r

    # ---- arena ----
    def alloc_arena(self):
        # (the ports' staging is carved from the same range: L1 is allocated in lockstep over the grid)
        crs = ttnn.CoreRangeSet(list(self.crs.ranges()) + list(self.port_crs.ranges()))
        n = self.GX * self.GY + len(self.port_xy)
        shard = ttnn.ShardSpec(crs, [1, self.arena_bytes // 4], ttnn.ShardOrientation.ROW_MAJOR)
        self.arena = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, n, self.arena_bytes // 4]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard),
        )
        return int(self.arena.buffer_address())

    def free_arena(self):
        if self.arena is not None and self.arena.is_allocated():
            ttnn.deallocate(self.arena)

    # ---- the program ----
    def _program(self, coord, y, w, se, out):
        """Chip coord's program: the engine (every core of the rectangle) and its gather ports."""
        mesh, GX, GY, cx0, RT = self.mesh, self.GX, self.GY, self.cx0, self.RT
        ch = self.chips[coord]
        order = ch["order"]
        b, off = int(self.arena.buffer_address()), self.off
        sl = int(self.landed.buffer_address())
        virt = lambda cx, cy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0 + cx, cy))  # noqa: E731
        cbs = []

        def cb(i, key, size, dt, page, blocks, crs=None):
            # (the plan check of se_engine3: a block addressed from the CB's pointer must not straddle its end - every
            # core pushes one size here and the dataflow ends read page by page, so each size must divide the CB)
            for blk in set(blocks):
                assert (
                    _ring_pages(size // page, [blk]) == size // page
                ), f"tail_engine: CB {key} ({size // page} pages) takes blocks of {blk}: one would straddle its end"
            d = ttnn.cb_descriptor_from_sharded_tensor(
                i, self.arena, address_offset=off[key], total_size=size, core_ranges=crs or self.crs
            )
            d.format_descriptors = [ttnn.CBFormatDescriptor(i, dt, page)]
            cbs.append(d)

        # (W' is resident: its K ranges pushed in order, never popped - no wrap)
        cb(CB_W, "w", self.KH * self.PNMAX * self.wtb, self.w_dtype, self.wtb, [1])
        cb(CB_A, "ring", self.NSLOT * self.KH * 2048, ttnn.bfloat16, 2048, [self.KH])
        cb(CB_PART, "part", self.NCK * GX * 4096, ttnn.float32, 4096, [GX])
        cb(CB_PMY, "pmy", 4096, ttnn.float32, 4096, [1])
        cb(CB_ONES, "ones", 2048, ttnn.bfloat16, 2048, [1])
        cb(CB_ACCM, "accm", self.FNR * 4096, ttnn.float32, 4096, self.FNS)
        cb(CB_XSEND, "xsend", 2 * XT * 4096, ttnn.float32, 4096, [XT])
        cb(CB_SQM, "sqm", 2 * 4096, ttnn.float32, 4096, [1])
        cb(CB_XRECV, "xrecv", self.NCK * XT * 4096, ttnn.float32, 4096, [XT])
        cb(CB_SE, "se", self.FNR * 2048, ttnn.bfloat16, 2048, self.FNS)
        cb(CB_OUT, "out", self.FNR * 2048, ttnn.bfloat16, 2048, self.FNS)
        cb(CB_L0, "land", self.SHMAX * 2048, ttnn.bfloat16, 2048, [1])  # (landing slot 0: my tilized own share)
        cb(CB_RM, "rm", self.SHMAX * 2048, ttnn.bfloat16, 2048, [1])  # (its 32 rows, row-major: one push)
        # the ports' staging (carved at the arena's start on the port cores) and their poll word after it
        PG = 8  # (packets a port group: 4 158 us, 8 148, 12 151)
        PSTAGE = 2 * PG * self.RP
        assert PSTAGE * 2048 + 64 + 64 <= self.arena_bytes
        d = ttnn.cb_descriptor_from_sharded_tensor(
            0, self.arena, address_offset=0, total_size=PSTAGE * 2048, core_ranges=self.port_crs
        )
        d.format_descriptors = [ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048)]
        cbs.append(d)
        d = ttnn.cb_descriptor_from_sharded_tensor(
            1, self.arena, address_offset=PSTAGE * 2048 + 64, total_size=4 * 16, core_ranges=self.port_crs
        )
        d.format_descriptors = [ttnn.CBFormatDescriptor(1, ttnn.uint32, 16)]
        cbs.append(d)
        ct = [self.KH, self.PNMAX, RT, self.NCK, GX, self.wtb, self.NSLOT, self.KT, XT]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        wkb = [k0 for k0, _ in self.wk] + [self.KH]
        b_rt, n_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for cx in range(GX):
            pn = self.pn[cx]
            for cy in range(GY):
                r, h = cy % RT, cy // RT
                fo, fn = self.fin[cx][h]
                v = virt(cx, cy)
                yt, yb = virt(cx, h * RT).y, virt(cx, h * RT + RT - 1).y
                ak0, ak1 = self.ak[cx]
                wk0, wk1 = self.wk[r]
                reg = self._region(h, cx, r)
                b_rt[cx0 + cx][cy] = [
                    r,
                    pn,
                    self.kh[h][0],
                    int(w.buffer_address()),
                    reg % self.nbank,
                    (reg // self.nbank) * self.wreg * self.wtb,
                    wk0,
                    wk1,
                    v.x,
                    yt,
                    yb,
                    int(y.buffer_address()),
                    ak0,
                    ak1,
                    int(se.buffer_address()),
                    self.NTO,
                    self.n0[cx] + fo,
                    fn,
                    b + off["land"],
                    self.SHMAX,
                    sl,
                ] + list(order)
                n_rt[cx0 + cx][cy] = (
                    [
                        r,
                        pn,
                        self.n0[cx] + fo,
                        fn,
                        ak0,
                        ak1,
                        virt(0, cy).x,
                        virt(GX - 1, cy).x,
                        cx,
                        int(out.buffer_address()),
                        self.NTO,
                        b + off["land"],
                        self.SHMAX,
                        b + off["ring"],
                        b + off["part"],
                        b + off["xrecv"],
                        v.x,
                        v.y,
                        virt(cx, (cy + RT) % GY).y,
                        sl,
                    ]
                    + list(order)
                    + [virt(c2, cy).x for c2 in range(GX)]
                    + wkb
                )
                c_rt[cx0 + cx][cy] = [pn, fo, fn, ak0, ak1, 0] + wkb[1:]
        # ---- ports ----
        from ttnn.bringup.combine_rf.fabric_all_gather_py import _OPP, _needs_ready

        node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        pr_rt, ps_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        prog = ttnn.ProgramDescriptor()
        akf = [k for k0k1 in self.ak for k in k0k1]
        pep = int(self.port_epoch.buffer_address())
        xs = [virt(c2, 0).x for c2 in range(GX)]
        for (j, dd, l), core in ch["ports"].items():
            rg = ch["rings"][j]
            peer = rg["next"] if dd == "fwd" else rg["prev"]
            sends = rg[dd] if peer is not None else []
            ys = [virt(0, r + RT * l).y for r in range(RT)]
            here = [part | order.index(q) << 8 for q, part in sends]
            pr_rt[core.x][core.y] = (
                [int(y.buffer_address()), self.kh[l][0], b + off["land"], sl, b + PSTAGE * 2048, len(sends), pep]
                + here
                + akf
                + xs
                + ys
            )
            send_ready = _needs_ready(self.chips, coord, j, dd)
            rc = (
                virt(*[0, 0])
                if not send_ready
                else mesh.worker_core_from_logical_core(self.chips[peer]["ports"][(j, _OPP[dd], l)])
            )
            args = [
                b + off["land"],
                sl,
                int(ttnn.get_global_semaphore_address(self.ready)),
                int(send_ready),
                rc.x,
                rc.y,
            ]
            if rg.get(f"{dd}_links"):
                pnode = node(peer)
                there = [part | self.chips[peer]["order"].index(q) << 8 for q, part in sends]
                args += [int(pnode.mesh_id), int(pnode.chip_id), len(sends)] + there + akf + xs + ys
                args += list(ttnn.setup_fabric_connection(node(coord), pnode, rg[f"{dd}_links"][l], prog, core))
            else:
                args += [0, 0, 0] + akf + xs + ys
            ps_rt[core.x][core.y] = args
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )
        comp = ttnn.ComputeConfigDescriptor()
        comp.math_fidelity = self.fid
        comp.fp32_dest_acc_en = True
        comp.dst_full_sync_en = True
        comp.math_approx_mode = False
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (CB_ACCM, CB_SQM, CB_XRECV):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        comp.unpack_to_dest_mode = modes
        pct = [RT, GX, self.RP, self.SHMAX, self.KT, PG]
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_BRISC,
                source_type=st,
                core_ranges=self.crs,
                compile_time_args=ct + acc(y) + acc(se),
                runtime_args=b_rt,
                config=dm("RISCV_0", ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_NCRISC,
                source_type=st,
                core_ranges=self.crs,
                compile_time_args=ct + acc(out),
                runtime_args=n_rt,
                config=dm("RISCV_1", ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=st,
                core_ranges=self.crs,
                compile_time_args=[self.KH, self.NCK, GX, _f32_bits(1.0 / self.Hr), _f32_bits(self.eps), XT],
                runtime_args=c_rt,
                config=comp,
            ),
            ttnn.KernelDescriptor(
                kernel_source=self.dbg + _PORT_READER,
                source_type=st,
                core_ranges=self.port_crs,
                compile_time_args=pct,
                runtime_args=pr_rt,
                config=dm("RISCV_1", ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=self.dbg + _PORT_SENDER,
                source_type=st,
                core_ranges=self.port_crs,
                compile_time_args=pct,
                runtime_args=ps_rt,
                config=dm("RISCV_0", ttnn.NOC.NOC_1),
            ),
        ]
        nsem = 2 * self.NCK + 3
        prog.cbs = cbs
        prog.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in range(nsem)]
        prog.custom_program_hash = ttnn.compute_program_descriptor_hash(prog)
        return prog

    def _mesh(self):
        md = ttnn.MeshProgramDescriptor()
        for (r, c), prog in self.state["progs"].items():
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        return md

    def _rebind(self):
        """The cached programs' CBs (all arena-backed) onto the arena taken again at the same address."""
        tmpl = ttnn.cb_descriptor_from_sharded_tensor(0, self.arena)
        for prog in self.state["progs"].values():
            cbs = list(prog.cbs)
            for c in cbs:
                c.set_buffer_from_cb(tmpl)
            prog.cbs = cbs
        self.state["md"] = self._mesh()

    def run(self, y, w, se, out=None):
        """y [1, 1, rows / G, Hr] ROW_MAJOR (DRAM): this chip's owned rows - the combine's output (the gather brings the
        rest of the mesh row's);
        se [1, 1, rows, Ho] TILE -> out [1, 1, rows, Ho] bf16 TILE. The arena is taken for the call and given back."""
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.rows, self.Ho]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        base = self.alloc_arena()
        key = (
            base,
            int(y.buffer_address()),
            int(w.buffer_address()),
            int(se.buffer_address()),
            int(out.buffer_address()),
        )
        if self.state["key"] != key:
            self.state["progs"] = {coord: self._program(coord, y, w, se, out) for coord in self.chips}
            self.state.update(key=key, md=self._mesh())
        else:
            self._rebind()
        ttnn.generic_op([y, w, se, out, self.arena], self.state["md"])
        if not self.keep_arena:  # (keep_arena: a test reads it back)
            self.free_arena()
        return out
