# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tail_engine - the fused K3 MoE tail on one program a chip: out [rows, Ho] = rmsnorm(z) @ W_up + se, with z [rows, Hr]
the mesh row's gathered latent rows (chunk q: TP chip q's owned rows), W_up this chip's column slice [Hr, Ho] and se the
shared expert's output in the block's layout.

The norm is folded: its weight g into W' = diag(g) W_up (upload), its 1 / rms per row applied to the matmul's fp32 sums
(out = diag(s) (z @ W') + se, s = rsqrt(mean(z^2) + eps)) - the matmul never waits for whole rows, only s does.
  grid    GX x GY cores; core (cx, cy): chunk-local row tile r = cy % RT, half h = cy // RT; column slice
          sl = h GX + cx: output tile columns n0(sl) .. + pn(sl) (column_split of Ho / 32 over HALVES GX slices)
  W'      resident: core (cx, cy) holds its slice [KT][pn] (bf8); the slice is read from DRAM once by its RT cores (the
          half column), each its K range, multicast down the half column (NOC0)
  z       chunk by chunk (RT row tiles), a ring of NSLOT chunk slots [KT] a core (row tile r); a chunk's row tile r is
          read by the 2D group of row tile r (rows r + RT hh, every column: NS cores), each its K share (into its
          landing region - per chunk, never reused in a call), then multicast into every group core's ring slot (NOC1)
          once every group core freed it (FREE credits, NS a chunk: a core frees a chunk when its output is packed)
  s       the group's helpers (cores whose slice is narrower than the widest: their spare time) each sum the squares of
          their K range of the row tile ((z * z) @ ones: every column the row's sum) and multicast the fp32 partial to
          the group's PART slot (chunk, helper); every core adds the NH partials, s = rsqrt(sum / Hr + eps)
  out     per chunk all of K in DEST (fp32), packed to ACC, then acc x s + se in DEST (SFPU), bf16 out (NCRISC)
Chunk order: `order` (the TP chips' row blocks in processing order). Semaphores are the program's (zero each launch):
every producer is on this chip."""

import math
import struct

import torch
from ttnn.bringup.combine_rf.lin2d import column_split

import ttnn
from models.demos.moe_block.tt.se_engine3 import _ring_pages

CB_W, CB_A, CB_SQ, CB_PART, CB_PMY, CB_ONES, CB_S, CB_SE, CB_OUT, CB_ACC = range(10)
SQ_T = 4  # (squares a group)
ARENA_MAX = 1400 * 1024

_CT = r"""
constexpr uint32_t KT = get_compile_time_arg_val(0), PNMAX = get_compile_time_arg_val(1), RT = get_compile_time_arg_val(2);
constexpr uint32_t NCK = get_compile_time_arg_val(3), GX = get_compile_time_arg_val(4), HALVES = get_compile_time_arg_val(5);
constexpr uint32_t WTB = get_compile_time_arg_val(6), NSLOT = get_compile_time_arg_val(7), NH = get_compile_time_arg_val(8);
constexpr uint32_t NS = GX;  // (a row: its cores hold the same row tile)
constexpr uint32_t SW = 0, SA0 = 1, SP0 = 1 + NCK, SFREE = 1 + 2 * NCK, SL = 2 + 2 * NCK;
constexpr uint32_t CB_W = 0, CB_A = 1, CB_PART = 3, CB_PMY = 4, CB_ONES = 5, CB_SE = 7, CB_OUT = 8;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
"""

_BRISC = (
    '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n#include "tools/profiler/kernel_profiler.hpp"\n'
    + _CT
    + r"""
constexpr auto w_args = TensorAccessorArgs<9>();
constexpr auto z_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto s_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
// NOC0: every DRAM read (W' K range of my slice, my z shares, my se tiles) and the W' multicast down my half column
void kernel_main() {
    size_t a = 0;
    const uint32_t r = get_arg_val<uint32_t>(a++), pn = get_arg_val<uint32_t>(a++), n0 = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++), wp0 = get_arg_val<uint32_t>(a++);
    const uint32_t wk0 = get_arg_val<uint32_t>(a++), wk1 = get_arg_val<uint32_t>(a++);
    const uint32_t mx = get_arg_val<uint32_t>(a++), yt = get_arg_val<uint32_t>(a++), yb = get_arg_val<uint32_t>(a++);
    const uint32_t z_addr = get_arg_val<uint32_t>(a++), ak0 = get_arg_val<uint32_t>(a++), ak1 = get_arg_val<uint32_t>(a++);
    const uint32_t se_addr = get_arg_val<uint32_t>(a++), helper = get_arg_val<uint32_t>(a++), NTO = get_arg_val<uint32_t>(a++);
    const uint32_t land = get_arg_val<uint32_t>(a++), SHMAX = get_arg_val<uint32_t>(a++);
    const uint32_t ord = a;
    const auto wg = TensorAccessor(w_args, w_addr, WTB);
    const auto zg = TensorAccessor(z_args, z_addr, 2048);
    const auto sg = TensorAccessor(s_args, se_addr, 2048);
    if (helper) {  // (the all-ones tile of the squares' row sums)
        cb_reserve_back(CB_ONES, 1);
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONES));
        for (uint32_t i = 0; i < 512; ++i) o[i] = 0x3F803F80u;
        cb_push_back(CB_ONES, 1);
    }
    auto share = [&](uint32_t c) {  // my z share of chunk c into my landing region (per chunk: never reused in a call)
        const uint32_t R = get_arg_val<uint32_t>(ord + c) * RT + r, dst = land + c * SHMAX * 2048;
        for (uint32_t k = ak0; k < ak1; ++k) noc_async_read(zg.get_noc_addr(R * KT + k), dst + (k - ak0) * 2048, 2048);
        noc_async_read_barrier();
        *sem(SL) = c + 1;
    };
    share(0);  // (the NCRISC multicasts it while W' loads)
    {  // W': my K range of my slice, then down the half column (its other RT - 1 cores)
        DeviceZoneScopedN("TL-WLOAD");
        const uint32_t wl1 = get_write_ptr(CB_W);
        for (uint32_t k = wk0; k < wk1; ++k)
            for (uint32_t j = 0; j < pn; ++j) noc_async_read(wg.get_noc_addr(wp0 + k * pn + j), wl1 + (k * pn + j) * WTB, WTB);
        noc_async_read_barrier();
        const uint32_t src = wl1 + wk0 * pn * WTB, bytes = (wk1 - wk0) * pn * WTB;
        if (RT > 1) {
            if (bytes) noc_async_write_multicast(src, get_noc_multicast_addr(mx, yt, mx, yb, src), bytes, RT - 1);
            noc_async_writes_flushed();
            noc_semaphore_inc_multicast(get_noc_multicast_addr(mx, yt, mx, yb, get_semaphore(SW)), 1, RT - 1);
        }
        noc_semaphore_inc(get_noc_addr(get_semaphore(SW)), 1);
    }
    for (uint32_t c = 1; c < NCK; ++c) share(c);
    for (uint32_t c = 0; c < NCK; ++c) {  // my se tiles
        const uint32_t R = get_arg_val<uint32_t>(ord + c) * RT + r;
        cb_reserve_back(CB_SE, pn);
        const uint32_t dst = get_write_ptr(CB_SE);
        for (uint32_t j = 0; j < pn; ++j) noc_async_read(sg.get_noc_addr(R * NTO + n0 + j), dst + j * 2048, 2048);
        noc_async_read_barrier();
        cb_push_back(CB_SE, pn);
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
// NOC1: my z share and my squares' partial multicast to my group's rows, the credits, the CB pushes for the compute,
// the output writes
void kernel_main() {
    size_t a = 0;
    const uint32_t r = get_arg_val<uint32_t>(a++), pn = get_arg_val<uint32_t>(a++), n0 = get_arg_val<uint32_t>(a++);
    const uint32_t ak0 = get_arg_val<uint32_t>(a++), ak1 = get_arg_val<uint32_t>(a++);
    const uint32_t lx = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), hidx = get_arg_val<uint32_t>(a++);
    const uint32_t o_addr = get_arg_val<uint32_t>(a++), NTO = get_arg_val<uint32_t>(a++);
    const uint32_t land = get_arg_val<uint32_t>(a++), SHMAX = get_arg_val<uint32_t>(a++), ring = get_arg_val<uint32_t>(a++);
    const uint32_t part = get_arg_val<uint32_t>(a++), my = get_arg_val<uint32_t>(a++);
    const uint32_t ord = a, xs = ord + NCK;  // (my row's NoC x, by column)
    const auto og = TensorAccessor(o_args, o_addr, 2048);
    // (NOC1 multicast corners: the right one first) my row - row multicasts only: a column hop on the same NoC stalls
    // the row multicasts crossing it
    auto row = [&](uint32_t addr) { return get_noc_multicast_addr(rx, my, lx, my, addr); };
    // to every core of my row, me as well: data then the increment of a semaphore
    auto to_group = [&](uint32_t src, uint32_t dst, uint32_t bytes, uint32_t s) {
        if (bytes) noc_async_write_multicast_loopback_src(src, row(dst), bytes, GX);
        noc_async_write_barrier();
        if (GX > 1) noc_semaphore_inc_multicast(row(get_semaphore(s)), 1, GX - 1);
        noc_semaphore_inc(get_noc_addr(get_semaphore(s)), 1);
    };
    // my shares go out as soon as each is loaded (BRISC) and its ring slot is free on every group core (FREE): from
    // every wait below, so a chunk's distribution never waits for the previous chunk's
    uint32_t cs = 0;
    bool wpushed = false;
    auto pump = [&]() {
        invalidate_l1_cache();
        if (!wpushed && *sem(SW) >= RT) {  // (my slice of W' landed: the half column's RT K ranges)
            cb_reserve_back(CB_W, KT * pn);
            cb_push_back(CB_W, KT * pn);
            wpushed = true;
        }
        if (cs >= NCK || *sem(SL) < cs + 1 || (cs >= NSLOT && *sem(SFREE) < NS * (cs - NSLOT + 1))) return;
        // (one row multicast of a chunk at a time, in column order - the chunk's arrival count is the token:
        // concurrent row multicasts reserve crossing paths and stall each other for tens of us)
        if (*sem(SA0 + cs) < hidx) return;
        DeviceZoneScopedN("TL-SHARE");
        to_group(land + cs * SHMAX * 2048, ring + ((cs % NSLOT) * KT + ak0) * 2048, (ak1 - ak0) * 2048, SA0 + cs);
        ++cs;
    };
    auto wait = [&](volatile tt_l1_ptr uint32_t* w, uint32_t n) {
        while (*w < n) pump();
    };
    // (small: unicasts to every core of my row, me as well - no multicast path to reserve)
    auto to_row_unicast = [&](uint32_t src, uint32_t dst, uint32_t bytes, uint32_t s) {
        for (uint32_t i = 0; i < GX; ++i) noc_async_write(src, get_noc_addr(get_arg_val<uint32_t>(xs + i), my, dst), bytes);
        noc_async_write_barrier();
        for (uint32_t i = 0; i < GX; ++i) noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xs + i), my, get_semaphore(s)), 1);
    };
    auto write_out = [&](uint32_t c) {
        const uint32_t R = get_arg_val<uint32_t>(ord + c) * RT + r;
        while (!cb_pages_available_at_front(CB_OUT, pn)) pump();
        cb_wait_front(CB_OUT, pn);
        const uint32_t src = get_read_ptr(CB_OUT);
        for (uint32_t j = 0; j < pn; ++j) noc_async_write(src + j * 2048, og.get_noc_addr(R * NTO + n0 + j), 2048);
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, pn);
        // (my ring slot of chunk c is free: the compute packs a chunk's output after its last use of the slot)
        if (GX > 1) noc_semaphore_inc_multicast(row(get_semaphore(SFREE)), 1, GX - 1);
        noc_semaphore_inc(get_noc_addr(get_semaphore(SFREE)), 1);
    };
    for (uint32_t c = 0; c < NCK; ++c) {
        {
            DeviceZoneScopedN("TL-AWAIT");
            wait(sem(SA0 + c), NS);
        }
        cb_reserve_back(CB_A, KT);
        cb_push_back(CB_A, KT);
        if (hidx != 0xFF) {  // my squares' partial (the compute's) to the group's PART slot (c, hidx)
            while (!cb_pages_available_at_front(CB_PMY, 1)) pump();
            cb_wait_front(CB_PMY, 1);
            to_row_unicast(get_read_ptr(CB_PMY), part + (c * NH + hidx) * 4096, 4096, SP0 + c);
            cb_pop_front(CB_PMY, 1);
        }
        {
            DeviceZoneScopedN("TL-PWAIT");
            wait(sem(SP0 + c), NH);
        }
        cb_reserve_back(CB_PART, NH);
        cb_push_back(CB_PART, NH);
        if (c > 0) write_out(c - 1);
    }
    write_out(NCK - 1);
    while (!wpushed || cs < NCK) pump();
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
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/copy_dest_values.h"
#include "tools/profiler/kernel_profiler.hpp"
constexpr uint32_t KT = get_compile_time_arg_val(0), NCK = get_compile_time_arg_val(1), NH = get_compile_time_arg_val(2);
constexpr uint32_t INV_HR = get_compile_time_arg_val(3), NORM_EPS = get_compile_time_arg_val(4), SQ_T = get_compile_time_arg_val(5);
constexpr uint32_t CB_W = 0, CB_A = 1, CB_SQ = 2, CB_PART = 3, CB_PMY = 4, CB_ONES = 5, CB_S = 6, CB_SE = 7, CB_OUT = 8;
constexpr uint32_t CB_ACC = 9;
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif
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
    const uint32_t pn = get_arg_val<uint32_t>(0), helper = get_arg_val<uint32_t>(1);
    const uint32_t hk0 = get_arg_val<uint32_t>(2), hk1 = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_A, CB_W, CB_ACC);
    cb_wait_front(CB_W, KT * pn);
    if (helper) cb_wait_front(CB_ONES, 1);
    for (uint32_t c = 0; c < NCK; ++c) {
        cb_wait_front(CB_A, KT);
        if (helper) {  // my K range's squares, summed along the row ((z * z) @ ones), fp32 into PMY (L1 accumulation)
            DeviceZoneScopedN("TL-SQ");
            cb_reserve_back(CB_PMY, 1);
            for (uint32_t k0 = hk0; k0 < hk1; k0 += SQ_T) {
                const uint32_t n = hk1 - k0 < SQ_T ? hk1 - k0 : SQ_T;
                reconfig_data_format_srca(CB_A);
                pack_reconfig_data_format(CB_SQ);
                PACK((llk_pack_reconfig_l1_acc(0)));  // (the squares overwrite; only PMY accumulates)
                copy_init(CB_A);
                cb_reserve_back(CB_SQ, SQ_T);  // (whole groups: the FIFO's blocks never straddle)
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; ++i) copy_tile(CB_A, k0 + i, i);  // (z^2 in fp32 on the SFPU)
                mul_binary_tile_init();
                for (uint32_t i = 0; i < n; ++i) mul_binary_tile(i, i, i);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < n; ++i) pack_tile(i, CB_SQ);
                tile_regs_release();
                cb_push_back(CB_SQ, SQ_T);
                cb_wait_front(CB_SQ, SQ_T);
                reconfig_data_format(CB_ONES, CB_SQ);
                pack_reconfig_data_format(CB_PMY);
                mm4_init(CB_SQ, CB_ONES);
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; ++i) mm4(CB_SQ, CB_ONES, i, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_reconfig_l1_acc(k0 > hk0 ? 1 : 0)));
                pack_tile<true>(0, CB_PMY, 0);
                tile_regs_release();
                cb_pop_front(CB_SQ, SQ_T);
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_push_back(CB_PMY, 1);
        }
        {  // my output tiles of the chunk: all of K in DEST, fp32 to ACC
            DeviceZoneScopedN("TL-MM");
            reconfig_data_format(CB_W, CB_A);
            pack_reconfig_data_format(CB_ACC);
            matmul_block_init(CB_A, CB_W, 0, pn, 1, KT);
            cb_reserve_back(CB_ACC, pn);
            tile_regs_acquire();
            for (uint32_t k = 0; k < KT; ++k) matmul_block(CB_A, CB_W, k, k * pn, 0, false, pn, 1, KT);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < pn; ++j) pack_tile(j, CB_ACC);
            tile_regs_release();
            cb_push_back(CB_ACC, pn);
            cb_pop_front(CB_A, KT);
        }
        {  // s = rsqrt(sum of the NH partials / Hr + eps): every column the row's
            DeviceZoneScopedN("TL-S");
            cb_wait_front(CB_PART, NH);
            cb_reserve_back(CB_S, 1);
            reconfig_data_format_srca(CB_PART);
            pack_reconfig_data_format(CB_S);
            copy_init(CB_PART);
            tile_regs_acquire();
            copy_tile(CB_PART, 0, 0);
            for (uint32_t i = 1; i < NH; ++i) {
                copy_init(CB_PART);
                copy_tile(CB_PART, i, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
#if TL_PROBE != 4
            binop_with_scalar_tile_init();
            mul_unary_tile(0, INV_HR);
            add_unary_tile(0, NORM_EPS);
            rsqrt_tile_init();
            rsqrt_tile(0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_S);
            tile_regs_release();
            cb_push_back(CB_S, 1);
            cb_pop_front(CB_PART, NH);
        }
        {  // out = acc x s + se (DEST: acc 0 .. pn - 1, s 4, se 5 ..), bf16
            DeviceZoneScopedN("TL-OUT");
            cb_wait_front(CB_ACC, pn);
            cb_wait_front(CB_S, 1);
            cb_wait_front(CB_SE, pn);
            cb_reserve_back(CB_OUT, pn);
            tile_regs_acquire();
            reconfig_data_format_srca(CB_ACC);
            copy_init(CB_ACC);
            for (uint32_t j = 0; j < pn; ++j) copy_tile(CB_ACC, j, j);
            reconfig_data_format_srca(CB_S);
            copy_init(CB_S);
            copy_tile(CB_S, 0, 4);
#if TL_PROBE == 3 || TL_PROBE == 4
            for (uint32_t j = 0; j < pn; ++j) copy_dest_values<DataFormat::Float32>(4, j);
#endif
#if TL_PROBE == 0 || TL_PROBE == 1
            mul_binary_tile_init();
            for (uint32_t j = 0; j < pn; ++j) mul_binary_tile(j, 4, j);
#endif
            reconfig_data_format_srca(CB_SE);
            copy_init(CB_SE);
            for (uint32_t j = 0; j < pn; ++j) copy_tile(CB_SE, j, 5 + j);
#if TL_PROBE == 0
            add_binary_tile_init();
            for (uint32_t j = 0; j < pn; ++j) add_binary_tile(j, 5 + j, j);
#elif TL_PROBE == 1
            for (uint32_t j = 0; j < pn; ++j) copy_dest_values<DataFormat::Float32>(5 + j, j);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            for (uint32_t j = 0; j < pn; ++j) pack_tile(j, CB_OUT);
            tile_regs_release();
            cb_push_back(CB_OUT, pn);
            cb_pop_front(CB_ACC, pn);
            cb_pop_front(CB_S, 1);
            cb_pop_front(CB_SE, pn);
        }
    }
}
"""


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
        NSLOT=3,
        fidelity=ttnn.MathFidelity.HiFi2,
        w_dtype=ttnn.bfloat8_b,
    ):
        self.mesh = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        GX = GX or grid.x - 1  # (the last column: the gather's ports - step B)
        self.GX, self.GY, self.cx0, self.NSLOT = GX, GY, cx0, NSLOT
        self.rows, self.Hr, self.Ho, self.NCK, self.eps = rows, Hr, Ho, chunks, eps
        self.KT, self.NTO = Hr // 32, Ho // 32
        assert rows % (32 * chunks) == 0
        self.RT = rows // 32 // chunks
        assert GY % self.RT == 0, f"GY {GY}: whole row tiles of a chunk a core row ({self.RT} a chunk)"
        self.HALVES = GY // self.RT
        assert self.NTO % self.HALVES == 0
        self.NS = self.HALVES * GX
        # (half h: output columns h NTO / HALVES .. - every row owns a row tile and half the columns; nothing crosses rows)
        half = column_split(self.NTO // self.HALVES, GX, 1)
        self.pn = half * self.HALVES
        self.n0 = [h * (self.NTO // self.HALVES) + sum(half[:cx]) for h in range(self.HALVES) for cx in range(GX)]
        self.PNMAX = max(self.pn)
        # (a FIFO of per-chunk blocks pn wide: its size a multiple of every core's pn - a block never straddles its end)
        lcm = math.lcm(*set(self.pn))
        self.PNR = lcm * -(-2 * self.PNMAX // lcm)
        assert self.PNMAX <= 3, "acc 0 .. 2, s 4, se 5 ..: 8 fp32 DEST tiles"
        self.wk = _split(self.KT, self.RT)  # (a half column's W' K ranges, by row tile)
        self.ak = _split(self.KT, GX)  # (a row's z shares, by column: each core also squares its share)
        self.NH = GX
        self.SHMAX = max(k1 - k0 for k0, k1 in self.ak)
        self.fid, self.w_dtype = fidelity, w_dtype
        self.wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[w_dtype]
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        off, o = {}, 0
        for k, v in [
            ("w", self.KT * self.PNMAX * self.wtb),
            ("ring", NSLOT * self.KT * 2048),
            ("land", chunks * self.SHMAX * 2048),
            ("sq", SQ_T * 4096),
            ("part", chunks * self.NH * 4096),
            ("pmy", 4096),
            ("ones", 2048),
            ("s", 4096),
            ("se", self.PNR * 2048),
            ("out", self.PNR * 2048),
            ("acc", self.PNR * 4096),
        ]:
            off[k], o = o, al(o + v)
        assert o <= ARENA_MAX, f"tail_engine: arena {o >> 10} KB > {ARENA_MAX >> 10} KB"
        self.off, self.arena_bytes = off, o
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx0, 0), ttnn.CoreCoord(cx0 + GX - 1, GY - 1))])
        self.arena = None
        self.keep_arena = False
        self.state = dict(key=None)
        # (TL_PROBE, test-only: 1 out = se, 2 out = the raw sums z @ W', 3 out = s)
        self.probe = int(__import__("os").environ.get("TL_PROBE", "0"))

    def slice_of(self, cx, cy):
        return (cy // self.RT) * self.GX + cx

    # ---- weights ----
    def upload(self, w_up_cols, g):
        """w_up_cols: per device (row-major mesh order) its [Hr, Ho] column slice of W_up^T; g [Hr] the norm weight
        (folded: W' = diag(g) W). One tensor a chip: per slice its [KT][pn] tiles, slice after slice (pages)."""
        devs = []
        for w in w_up_cols:
            wf = w.float() * g.float().view(-1, 1)
            pages = []
            for s in range(self.NS):
                blk = wf[:, self.n0[s] * 32 : (self.n0[s] + self.pn[s]) * 32]
                pages.append(blk.reshape(self.KT, 32, self.pn[s], 32).permute(0, 2, 1, 3).reshape(-1, 32))
            devs.append(torch.cat(pages))
        t = torch.stack(devs)  # [N, KT Ho, 32]
        R, C = tuple(self.mesh.shape)
        self.wp0 = [sum(self.KT * p for p in self.pn[:s]) for s in range(self.NS)]
        return ttnn.from_torch(
            t.view(R, C, -1, 32),
            dtype=self.w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(R, C), dims=(0, 1)),
        )

    # ---- arena ----
    def alloc_arena(self):
        n = self.GX * self.GY
        shard = ttnn.ShardSpec(self.crs, [1, self.arena_bytes // 4], ttnn.ShardOrientation.ROW_MAJOR)
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
    def _program(self, z, w, se, out, order):
        mesh, GX, GY, cx0, RT = self.mesh, self.GX, self.GY, self.cx0, self.RT
        b, off = int(self.arena.buffer_address()), self.off
        virt = lambda cx, cy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx0 + cx, cy))  # noqa: E731
        cbs = []

        def cb(i, key, size, dt, page, blocks):
            # (the plan check of se_engine3: a block addressed from the CB's pointer must not straddle its end - every
            # core pushes one size here and the dataflow ends read page by page, so each size must divide the CB)
            for blk in set(blocks):
                assert (
                    _ring_pages(size // page, [blk]) == size // page
                ), f"tail_engine: CB {key} ({size // page} pages) takes blocks of {blk}: one would straddle its end"
            d = ttnn.cb_descriptor_from_sharded_tensor(
                i, self.arena, address_offset=off[key], total_size=size, core_ranges=self.crs
            )
            d.format_descriptors = [ttnn.CBFormatDescriptor(i, dt, page)]
            cbs.append(d)

        pns = list(self.pn)
        cb(CB_W, "w", self.KT * self.PNMAX * self.wtb, self.w_dtype, self.wtb, [self.KT * self.PNMAX])  # (one push)
        cb(CB_A, "ring", self.NSLOT * self.KT * 2048, ttnn.bfloat16, 2048, [self.KT])
        cb(CB_SQ, "sq", SQ_T * 4096, ttnn.float32, 4096, [SQ_T])
        cb(CB_PART, "part", self.NCK * self.NH * 4096, ttnn.float32, 4096, [self.NH])
        cb(CB_PMY, "pmy", 4096, ttnn.float32, 4096, [1])
        cb(CB_ONES, "ones", 2048, ttnn.bfloat16, 2048, [1])
        cb(CB_S, "s", 4096, ttnn.float32, 4096, [1])
        cb(CB_SE, "se", self.PNR * 2048, ttnn.bfloat16, 2048, pns)
        cb(CB_OUT, "out", self.PNR * 2048, ttnn.bfloat16, 2048, pns)
        cb(CB_ACC, "acc", self.PNR * 4096, ttnn.float32, 4096, pns)
        ct = [self.KT, self.PNMAX, RT, self.NCK, GX, self.HALVES, self.wtb, self.NSLOT, self.NH]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        b_rt, n_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for cx in range(GX):
            for cy in range(GY):
                r, hh = cy % RT, cy // RT
                s = self.slice_of(cx, cy)
                pn = self.pn[s]
                v = virt(cx, cy)
                yt, yb = virt(cx, hh * RT).y, virt(cx, hh * RT + RT - 1).y
                ak0, ak1 = self.ak[cx]
                wk0, wk1 = self.wk[r]
                b_rt[cx0 + cx][cy] = [
                    r,
                    pn,
                    self.n0[s],
                    int(w.buffer_address()),
                    self.wp0[s],
                    wk0,
                    wk1,
                    v.x,
                    yt,
                    yb,
                    int(z.buffer_address()),
                    ak0,
                    ak1,
                    int(se.buffer_address()),
                    1,
                    self.NTO,
                    b + off["land"],
                    self.SHMAX,
                ] + list(order)
                n_rt[cx0 + cx][cy] = (
                    [
                        r,
                        pn,
                        self.n0[s],
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
                        v.y,
                    ]
                    + list(order)
                    + [virt(c2, cy).x for c2 in range(GX)]
                )
                c_rt[cx0 + cx][cy] = [pn, 1, ak0, ak1]
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
        for i in (CB_PART, CB_S, CB_ACC):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        comp.unpack_to_dest_mode = modes
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_BRISC,
                source_type=st,
                core_ranges=self.crs,
                compile_time_args=ct + acc(w) + acc(z) + acc(se),
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
                kernel_source=f"#define TL_PROBE {self.probe}\n" + _COMPUTE,
                source_type=st,
                core_ranges=self.crs,
                compile_time_args=[
                    self.KT,
                    self.NCK,
                    self.NH,
                    _f32_bits(1.0 / self.Hr),
                    _f32_bits(self.eps),
                    SQ_T,
                ],
                runtime_args=c_rt,
                config=comp,
            ),
        ]
        nsem = 3 + 2 * self.NCK
        prog = ttnn.ProgramDescriptor()
        prog.cbs = cbs
        prog.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in range(nsem)]
        prog.kernels = kernels
        prog.custom_program_hash = ttnn.compute_program_descriptor_hash(prog)
        self.state["prog"] = prog
        return self._mesh(prog)

    def _mesh(self, prog):
        R, C = tuple(self.mesh.shape)
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
        return md

    def _rebind(self):
        """The cached program's CBs (all arena-backed) onto the arena taken again at the same address."""
        tmpl = ttnn.cb_descriptor_from_sharded_tensor(0, self.arena)
        prog = self.state["prog"]
        cbs = list(prog.cbs)
        for c in cbs:
            c.set_buffer_from_cb(tmpl)
        prog.cbs = cbs
        self.state["md"] = self._mesh(prog)

    def run(self, z, w, se, out=None, order=None):
        """z [1, 1, rows, Hr] TILE (DRAM), se [1, 1, rows, Ho] TILE -> out [1, 1, rows, Ho] bf16 TILE. The arena is taken
        for the call and given back."""
        order = list(order if order is not None else range(self.NCK))
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.rows, self.Ho]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        base = self.alloc_arena()
        key = (base, int(z.buffer_address()), int(w.buffer_address()), int(se.buffer_address()))
        key += (int(out.buffer_address()), tuple(order))
        if self.state["key"] != key:
            self.state.update(key=key, md=self._program(z, w, se, out, order))
        else:
            self._rebind()
        ttnn.generic_op([z, w, se, out, self.arena], self.state["md"])
        if not self.keep_arena:  # (keep_arena: a test reads it back)
            self.free_arena()
        return out
