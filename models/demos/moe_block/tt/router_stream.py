# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""router_stream - the fused head's router on its own cores, under the x chain: logits = x_own [rows, H / TP] @
W_r[H / TP, E] (the router's H-split slice, bf16, HiFi4, fp32 out) -> the logits reduce-scatter by tokens over TP
(se_rs.LogitsPlan: fp32 on the wire, measured - bf16 changes 3-6% of the rows' picks) -> on the owners: scores =
sigmoid(logits), biased b = scores + bias -> top-K by b, weights = the picked scores / their sum x route_scale -> the
joint routing rows (router_topk's format: per token K expert ids then K bf16 weights, [owned / 32, 32 x 2 K] uint16).

Cores: RB = CR / TP router cores (one chunk of CR row tiles; core j owns row tile j of every block, i.e. of every
chip's tokens), one column of the mover column. Core j: its x rows and W_r^T streamed in K blocks of KBR (W_r^T read
from DRAM once: a rotating column multicast), logits computed TRANSPOSED (W_r^T x x^T: a token is a tile column), the
G x ET fp32 partials accumulated in L1 (packer L1 accumulation), then the plan's hand-offs / finishes (a relay adds the
arrival in fp32 DEST, packed back in place), the fin entry's sum -> sigmoid -> + bias -> scores, the top-K on the
SFPU (ttnn.topk's bitonic network over each token column, fp32), the joint rows (BRISC) for its 32 owner tokens. Hand-offs are pushed into the port's L1
(the sub-block, then +1 on the port's word of this core); the port's sender polls them between its other packets (the
per-model priority) and sends them into the downstream router core's zone with a fused increment."""

import struct

import torch

import ttnn
from models.demos.moe_block.tt.se_rs import DIRS, LogitsPlan

XR, WR, HOLD, ARRF, ARRB, BIAS, SC, FINC, PSYNC, BEST, BESTI, IDX = range(12)
_NCRISC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
using namespace dataflow_kernel_lib;
constexpr uint32_t KT = get_compile_time_arg_val(0), ET = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr uint32_t KBR = get_compile_time_arg_val(3);
constexpr auto x_args = TensorAccessorArgs<5>();
constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto i_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr uint32_t XR = 0, WR = 1, ARRF = 3, ARRB = 4, BIAS = 5, IDX = 11;
// args: x, W_r^T, bias^T, expert index tiles; my rows [G]; arrivals fwd: n, thresholds; bwd: n, thresholds; the
// counters (fwd, bwd)
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1), b_addr = get_arg_val<uint32_t>(2);
    const uint32_t i_addr = get_arg_val<uint32_t>(3);
    const uint32_t r0 = 4, af0 = r0 + G + 1, naf = get_arg_val<uint32_t>(r0 + G);
    const uint32_t nab = get_arg_val<uint32_t>(af0 + naf), ab0 = af0 + naf + 1;
    volatile tt_l1_ptr uint32_t* arrf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ab0 + nab));
    volatile tt_l1_ptr uint32_t* arrb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ab0 + nab + 1));
    {  // (the top-K's done word: zero before the compute can write it - it is written after the scores)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ab0 + nab + 2)) = 0;
    }
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const auto wg = TensorAccessor(w_args, w_addr, 2048);
    const auto bg = TensorAccessor(b_args, b_addr, 4096);
    const auto ig = TensorAccessor(i_args, i_addr, 2048);
    cb_reserve_back(BIAS, ET);
    cb_reserve_back(IDX, ET);
    for (uint32_t t = 0; t < ET; ++t) {
        noc_async_read(bg.get_noc_addr(t), get_write_ptr(BIAS) + t * 4096, 4096);
        noc_async_read(ig.get_noc_addr(t), get_write_ptr(IDX) + t * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(BIAS, ET);
    cb_push_back(IDX, ET);
    uint32_t af = 0, ab = 0;
    // W_r^T is read from DRAM once: K block kb by the column's rotating sender kb % RB, multicast down the router
    // column into every core's WR slot (the kernel_lib family's handshake: a core is ready once its slot is reserved)
    constexpr auto mw = McastArgs<get_named_compile_time_arg_val("rw_ct_offset"), get_named_compile_time_arg_val("rw_rt_offset")>();
    Noc noc;
    auto sw = mw.optional_sender(noc);
    auto rw = mw.optional_receiver(noc);
    for (uint32_t kb = 0; kb < KT / KBR; ++kb) {  // K block kb: x [KBR][G] (k-major: the matmul's in1) and W_r^T [ET][KBR]
        cb_reserve_back(XR, G * KBR);
        cb_reserve_back(WR, KBR * ET);
        const uint32_t xl = get_write_ptr(XR), l1 = get_write_ptr(WR);
        for (uint32_t e = 0; e < G; ++e) {
            const uint32_t R = get_arg_val<uint32_t>(r0 + e);
            for (uint32_t k = 0; k < KBR; ++k) noc_async_read(xg.get_noc_addr(R * KT + kb * KBR + k), xl + (k * G + e) * 2048, 2048);
        }
        if (mw.should_send(kb)) {
            for (uint32_t e = 0; e < ET; ++e)
                for (uint32_t k = 0; k < KBR; ++k) noc_async_read(wg.get_noc_addr(e * KT + kb * KBR + k), l1 + (e * KBR + k) * 2048, 2048);
            noc_async_read_barrier();
            sw->send(l1, l1, KBR * ET * 2048);
        } else {
            rw->receive(kb);
        }
        noc_async_read_barrier();
        cb_push_back(XR, G * KBR);
        cb_push_back(WR, KBR * ET);
    }
    while (af < naf || ab < nab) {  // arrivals in zone-slot order, each once its threshold landed
        invalidate_l1_cache();
        if (af < naf && *arrf >= get_arg_val<uint32_t>(af0 + af)) {
            cb_reserve_back(ARRF, ET);
            cb_push_back(ARRF, ET);
            ++af;
        }
        if (ab < nab && *arrb >= get_arg_val<uint32_t>(ab0 + ab)) {
            cb_reserve_back(ARRB, ET);
            cb_push_back(ARRB, ET);
            ++ab;
        }
    }
    if (naf) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrf)), 0u - naf);  // (re-arm)
    if (nab) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrb)), 0u - nab);
    noc_async_atomic_barrier();
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/topk.h"
#include "tools/profiler/kernel_profiler.hpp"
constexpr uint32_t KT = get_compile_time_arg_val(0), ET = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr uint32_t KBR = get_compile_time_arg_val(3), K = get_compile_time_arg_val(4);
constexpr uint32_t XR = 0, WR = 1, HOLD = 2, ARRF = 3, ARRB = 4, BIAS = 5, SC = 6, FINC = 7, PSYNC = 8;
constexpr uint32_t BEST = 9, BESTI = 10, IDX = 11;
// args: nops, ops (kind | entry << 8: 1 hand, 2 fin), per entry: mode (1 fwd, 2 bwd, 4 fin), fwd slot, bwd slot;
// then the top-K's done word. Every tile here is TRANSPOSED: logits^T [ET][32 tokens] - a token is a column.
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(WR, XR, HOLD);
    pack_reconfig_data_format(HOLD);
    {
    DeviceZoneScopedN("SE-RMM");
    for (uint32_t kb = 0; kb < KT / KBR; ++kb) {  // K-outer: every token tile's partial accumulated in L1 (fp32)
        cb_wait_front(XR, G * KBR);
        cb_wait_front(WR, KBR * ET);
        PACK((llk_pack_reconfig_l1_acc(kb > 0 ? 1 : 0)));
        for (uint32_t e0 = 0; e0 < ET; e0 += 2) {  // (logits^T [e0 .. e0 + re) x [G token tiles]: W_r^T x x^T)
            const uint32_t re = ET - e0 < 2 ? ET - e0 : 2;
            matmul_block_init(WR, XR, 1, G, re, KBR);
            tile_regs_acquire();
            for (uint32_t kk = 0; kk < KBR; ++kk) matmul_block(WR, XR, e0 * KBR + kk, kk * G, 0, true, G, re, KBR);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < re; ++r)
                for (uint32_t g = 0; g < G; ++g) pack_tile<true>(r * G + g, HOLD, g * ET + e0 + r);
            tile_regs_release();
        }
        cb_pop_front(XR, G * KBR);
        cb_pop_front(WR, KBR * ET);
    }
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    DeviceZoneScopedN("SE-RFIN");
    cb_reserve_back(PSYNC, 1);
    cb_push_back(PSYNC, 1);
    cb_wait_front(PSYNC, 1);  // (UNPACK: the partials are in L1)
    cb_wait_front(BIAS, ET);
    const uint32_t nops = get_arg_val<uint32_t>(0), et0 = 1 + nops;
    for (uint32_t o = 0; o < nops; ++o) {
        const uint32_t op = get_arg_val<uint32_t>(1 + o), kind = op & 0xFF, e = op >> 8;
        if (kind == 1) {
            cb_reserve_back(FINC, 1);
            cb_push_back(FINC, 1);
            continue;
        }
        const uint32_t mode = get_arg_val<uint32_t>(et0 + 3 * e);
        const uint32_t sf = get_arg_val<uint32_t>(et0 + 3 * e + 1), sb = get_arg_val<uint32_t>(et0 + 3 * e + 2);
        if (mode & 1) cb_wait_front(ARRF, (sf + 1) * ET);
        if (mode & 2) cb_wait_front(ARRB, (sb + 1) * ET);
        const uint32_t a1 = (mode & 1) ? ARRF : ARRB, s1 = ((mode & 1) ? sf : sb) * ET;
        if (mode & 4) cb_reserve_back(SC, ET);
        pack_reconfig_data_format(HOLD);
        for (uint32_t i0 = 0; i0 < ET; i0 += 8) {
            const uint32_t n = ET - i0 < 8 ? ET - i0 : 8;
            tile_regs_acquire();
            reconfig_data_format(HOLD, a1);
            add_init(HOLD, a1);
            for (uint32_t j = 0; j < n; ++j) add_tiles(HOLD, a1, e * ET + i0 + j, s1 + i0 + j, j);
            if ((mode & 3) == 3) {
                reconfig_data_format(HOLD, ARRB);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(ARRB);
                for (uint32_t j = 0; j < n; ++j)
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(ARRB, sb * ET + i0 + j, j);
            }
            if (mode & 4) {  // the owner: scores = sigmoid(logits), biased = scores + bias (bias^T: per row)
                sigmoid_tile_init();
                for (uint32_t j = 0; j < n; ++j) sigmoid_tile(j);
                reconfig_data_format(HOLD, BIAS);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(BIAS);
                for (uint32_t j = 0; j < n; ++j) add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(BIAS, i0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                if (mode & 4) pack_tile<true>(j, SC, i0 + j);
                else pack_tile<true>(j, HOLD, e * ET + i0 + j);  // (relay: the sum in place, then handed off)
            }
            tile_regs_release();
        }
        if (!(mode & 4)) {
            cb_reserve_back(FINC, 1);
            cb_push_back(FINC, 1);
            continue;
        }
        cb_push_back(SC, ET);
        // the owner's 32 tokens' top-K on the SFPU (ttnn.topk's bitonic network): each token a column; the running
        // best 32 of every column (fp32 values, uint16 expert ids) against the next 32 experts, 64-wide sort,
        // descending; fp32 straight to DEST (no srcA rounding of the scores)
        DeviceZoneScopedN("SE-RTOPK");
        cb_wait_front(SC, ET);
        cb_wait_front(IDX, ET);
        ckernel::topk_tile_init<false, false>();
        for (uint32_t t = 1; t < ET; ++t) {
            tile_regs_acquire();
            const uint32_t vb = t == 1 ? SC : BEST, ib = t == 1 ? IDX : BESTI;
            const uint32_t vi = t == 1 ? 0 : t - 2, ii = t == 1 ? 0 : t - 2;
            if (t > 1) {
                cb_wait_front(BEST, t - 1);
                cb_wait_front(BESTI, t - 1);
            }
            reconfig_data_format_srca(vb);
            copy_init(vb);
            copy_tile(vb, vi, 0);
            reconfig_data_format_srca(vb, ib);
            copy_init(ib);
            copy_tile(ib, ii, 2);
            reconfig_data_format_srca(ib, SC);
            copy_init(SC);
            copy_tile(SC, t, 1);
            reconfig_data_format_srca(SC, IDX);
            copy_init(IDX);
            copy_tile(IDX, t, 3);
            ckernel::topk_tile_init<false, false>();
            ckernel::topk_local_sort<false, DST_ACCUM_MODE, false, false, ckernel::TopkTieOrder::Unset>(0, 0, 5);
            tile_regs_commit();
            cb_reserve_back(BEST, 1);
            cb_reserve_back(BESTI, 1);
            tile_regs_wait();
            pack_reconfig_data_format(BEST);
            pack_tile(0, BEST);
            pack_reconfig_data_format(BESTI);
            pack_tile(2, BESTI);
            tile_regs_release();
            cb_push_back(BEST, 1);
            cb_push_back(BESTI, 1);
        }
        PACK(({  // (the last best pair packed: the BRISC's rows can go)
            *reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(et0 + 3 * G)) = 1;
        }));
    }
}
"""

_BRISC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
constexpr uint32_t ET = get_compile_time_arg_val(1), K = get_compile_time_arg_val(4);
constexpr auto o_args = TensorAccessorArgs<5>();
constexpr uint32_t HOLD = 2, BIAS = 5, FINC = 7, BEST = 9, BESTI = 10;
FORCE_INLINE float f32(uint32_t u) { union { uint32_t u; float f; } v{u}; return v.f; }
FORCE_INLINE uint32_t u32(float f) { union { float f; uint32_t u; } v{f}; return v.u; }
FORCE_INLINE uint32_t bf16_rne(float f) { const uint32_t u = u32(f); return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
// (r, c) of a tile: face (r / 16) * 2 + c / 16, then row-major within the face
FORCE_INLINE uint32_t fo(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
// args: nh, per hand-off (entry, port x, port y, port L1 offset of the sub-block, port word address); the out address,
// my owner page, route scale bits, the joint rows' L1 scratch, the done word, BEST / BESTI addresses
void kernel_main() {
    const uint32_t nh = get_arg_val<uint32_t>(0), h0 = 1;
    const uint32_t hold = get_read_ptr(HOLD);
    for (uint32_t j = 0; j < nh; ++j) {
        DeviceZoneScopedN("SE-RHAND");
        cb_wait_front(FINC, j + 1);
        const uint32_t e = get_arg_val<uint32_t>(h0 + 5 * j), px = get_arg_val<uint32_t>(h0 + 5 * j + 1);
        const uint32_t py = get_arg_val<uint32_t>(h0 + 5 * j + 2);
        noc_async_write(hold + e * ET * 4096, get_noc_addr(px, py, get_arg_val<uint32_t>(h0 + 5 * j + 3)), ET * 4096);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_arg_val<uint32_t>(h0 + 5 * j + 4)), 1);
    }
    const uint32_t t0 = h0 + 5 * nh;
    const uint32_t o_addr = get_arg_val<uint32_t>(t0), page = get_arg_val<uint32_t>(t0 + 1);
    const float scale = f32(get_arg_val<uint32_t>(t0 + 2));
    const uint32_t o_l1 = get_arg_val<uint32_t>(t0 + 3);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(t0 + 4));
    // the best 32 of every token column, sorted descending: rows 0 .. K - 1 are its top K (the last best page)
    const uint32_t bv = get_arg_val<uint32_t>(t0 + 5) + (ET - 2) * 4096, bi = get_arg_val<uint32_t>(t0 + 6) + (ET - 2) * 2048;
    cb_wait_front(BIAS, ET);
    const uint32_t bias = get_read_ptr(BIAS);
    while (*done == 0) invalidate_l1_cache();
    *done = 0;  // (re-arm)
    for (uint32_t c = 0; c < 32; ++c) {  // token c of my owner row tile
        uint32_t ids[K];
        float w[K], sum = 0.0f;
        for (uint32_t q = 0; q < K; ++q) {
            const uint32_t eid = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bi)[fo(q, c)];
            ids[q] = eid;
            const float b = f32(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bv)[fo(q, c)]);
            w[q] = b - f32(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias + (eid >> 5) * 4096)[fo(eid & 31, 0)]);
            sum += w[q];
        }
        const float inv = scale / (sum + 1e-20f);
        volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(o_l1 + c * 4 * K);
        for (uint32_t q = 0; q < K; ++q) {
            o[q] = static_cast<uint16_t>(ids[q]);
            o[K + q] = static_cast<uint16_t>(bf16_rne(w[q] * inv));
        }
    }
    const auto og = TensorAccessor(o_args, o_addr, 32 * 4 * K);
    noc_async_write(o_l1, og.get_noc_addr(page), 32 * 4 * K);
    noc_async_write_barrier();
}
"""

# the port sender's poll (x_gather3 includes it): the router cores' hand-offs landed in this port's L1, sent in the
# plan's walk (unit, core) as each core's word shows it; fp32 packets of RPL tiles, a sub-block's last one with the
# fused increment of the downstream router core's arrival counter. One dedicated header, flushed per packet.
LSTREAM = r"""
struct LStream {
    uint32_t r0, nu, nc, buf, word, zone, actr, et, rpl, u, i, sent;
    uint32_t seen[8];
    void init(uint32_t a) {
        r0 = a; nu = get_arg_val<uint32_t>(a); nc = get_arg_val<uint32_t>(a + 1 + 2 * nu);
        const uint32_t t = a + 2 + 2 * nu + nc;
        buf = get_arg_val<uint32_t>(t); word = get_arg_val<uint32_t>(t + 1); zone = get_arg_val<uint32_t>(t + 2);
        actr = get_arg_val<uint32_t>(t + 3); et = get_arg_val<uint32_t>(t + 4); rpl = get_arg_val<uint32_t>(t + 5);
        u = 0; i = 0; sent = 0;
        for (uint32_t c = 0; c < 8; ++c) seen[c] = 0;
    }
    bool done() const { return u >= nu; }
    static uint32_t len(uint32_t a) { const uint32_t nu = get_arg_val<uint32_t>(a); return 2 + 2 * nu + get_arg_val<uint32_t>(a + 1 + 2 * nu) + 6; }
    // send what is ready; blocking: until everything is sent
    template <typename C, typename H>
    void poll(C& conn, H hdr, bool blocking) {
        while (u < nu) {
            const uint32_t w0 = get_arg_val<uint32_t>(r0 + 1 + 2 * u), w1 = get_arg_val<uint32_t>(r0 + 2 + 2 * u);
            const uint32_t mask = (w0 >> 16) & 3;
            if (i >= nc) { ++u; i = 0; continue; }
            const uint32_t cw = get_arg_val<uint32_t>(r0 + 2 + 2 * nu + i), hf = (cw >> 16) & 1;
            if (!((mask >> hf) & 1)) { ++i; continue; }
            volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16 * i);
            invalidate_l1_cache();
            if (*wd < seen[i] + 1) {
                if (!blocking) return;
                noc_semaphore_wait_min(wd, seen[i] + 1);
            }
            ++seen[i];
            const uint32_t x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            const uint32_t src0 = buf + ((u * nc + i) * et) * 4096;
            const uint32_t dst0 = zone + (hf ? (w1 >> 16) : (w1 & 0xFFFF)) * et * 4096;
            for (uint32_t t0 = 0; t0 < et; t0 += rpl) {
                const uint32_t n = et - t0 < rpl ? et - t0 : rpl;
                const uint64_t dst = get_noc_addr(x, y, dst0 + t0 * 4096);
                noc_async_writes_flushed();  // (the header's previous packet has left)
                if (t0 + rpl >= et) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(x, y, actr), 1, true}, n * 4096);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * 4096);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src0 + t0 * 4096, n * 4096, reinterpret_cast<uint32_t>(hdr));
            }
            ++i;
        }
    }
    void rearm() {
        for (uint32_t c = 0; c < nc; ++c) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16 * c) = 0;
    }
};
"""


class RouterStream:
    """The router chain on its own cores (module docstring). cores: the RB logical router cores (the mover column's
    free cells), xg: the gather (its ports carry the logits)."""

    def __init__(self, mesh_device, xg, *, rows, H, E, K, route_scale, cores, prio_first, arena, words):
        """arena: (tensor, base, bytes) - the engine's arena, also carved on these cores and the ports (lockstep L1);
        words: the port words' address (a zeroed L1 tensor on the ports, 16 B a router core)."""
        self.mesh, self.xg = mesh_device, xg
        G = self.G = xg.G
        CR = rows // 32
        assert CR % G == 0
        self.RB, self.KT, self.ET, self.K = CR // G, H // G // 32, E // 32, K
        assert len(cores) >= self.RB and E % 32 == 0 and K % 2 == 0 and 4 * K <= 64
        self.cores = cores[: self.RB]
        self.KBR = max(d for d in range(1, 5) if self.KT % d == 0)
        self.scale_bits = struct.unpack("<I", struct.pack("<f", route_scale))[0]
        self.prio_first = prio_first  # (the port sends the logits ahead of the x gather: GLM-type)
        self.plan = LogitsPlan(G=G, ring=xg.ring, RB=self.RB, ET=self.ET, nv=xg.nv)
        ET, RB = self.ET, self.RB
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        off, o = {}, 0
        for k, v in [
            ("xr", 2 * G * self.KBR * 2048),
            ("wr", 2 * self.KBR * ET * 2048),
            ("hold", G * ET * 4096),
            ("arrf", self.plan.slots["fwd"] * ET * 4096),
            ("arrb", self.plan.slots["bwd"] * ET * 4096),
            ("bias", ET * 4096),
            ("sc", ET * 4096),
            ("idx", ET * 2048),
            ("best", ET * 4096),
            ("besti", ET * 2048),
            ("finc", 16 * 8),
            ("psync", 16 * 2),
            ("out", 32 * 4 * K),
            ("words", 64),
        ]:
            off[k], o = o, al(o + v)
        self.off, self.bytes = off, o
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.crs = crs
        self.arena, self.base, abytes = arena
        assert o <= abytes, f"router cores need {o} B of the arena's {abytes}"
        # the ports' buffers (in the arena's range there): per port the sub-blocks of its units x cores
        nlink = -(-RB // xg.nv)
        self.pbuf_sub = (G - 1) * nlink * ET * 4096
        assert self.pbuf_sub <= abytes
        self.pbase, self.words = self.base, words
        self.sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]
        self.arr = [int(ttnn.get_global_semaphore_address(s)) for s in self.sems]
        ttnn.synchronize_device(mesh_device)

    def upload(self, w_router, bias):
        """w_router [E, H] (all chips the same layer), bias [E] -> (W_r^T [E, H / TP] a chip: the chip's H slice,
        bf16; the bias as [E, 32] fp32 tiles (every column the bias: the transposed scores' rows); the expert index
        tiles [E, 32] uint16, replicated)."""
        R, C = tuple(self.mesh.shape)
        E, H = w_router.shape
        wt = w_router.float()  # [E, H]
        glob = torch.cat([wt[:, c * H // C : (c + 1) * H // C] for c in range(C)], dim=1)  # [E, C H / C]
        glob = torch.cat([glob] * R, dim=0)
        w = ttnn.from_torch(
            glob,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(0, 1), mesh_shape=(R, C)),
        )
        b = ttnn.from_torch(
            bias.float().reshape(-1, 1).expand(-1, 32).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ix = ttnn.from_torch(
            torch.arange(E, dtype=torch.int32).reshape(-1, 1).expand(-1, 32).contiguous(),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        return w, b, ix

    def out_tensor(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.RB, 32 * 2 * self.K]),
            ttnn.uint16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def _port_layout(self, l):
        """This link's router cores (plan core indices) in the walk order."""
        return [j for j in range(self.RB) if self.plan.link(j) == l]

    def port_block(self, coord, d, v):
        """The port sender's logits args (LSTREAM): units, cores (NoC | half), its buffer / words, the downstream
        router cores' zone of direction d, their arrival counter, ET, tiles a packet."""
        plan, p = self.plan, self.xg.chips[coord]["rings"][0]["p"]
        if plan.down(p, d) is None:
            return [0, 0] + [0] * 6
        units = plan.port_units(p, d)
        js = self._port_layout(v)
        virt = self.mesh.worker_core_from_logical_core
        out = [len(units)] + [w for e, b, m, s0, s1 in units for w in (e | b << 8 | m << 16, s0 | s1 << 16)]
        out += [len(js)] + [virt(self.cores[j]).x | virt(self.cores[j]).y << 8 | plan.half(j) << 16 for j in js]
        rpl = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // 4096)
        zone = self.base + self.off["arrf" if d == "fwd" else "arrb"]
        return out + [self.pbase, self.words, zone, self.arr[DIRS.index(d)], self.ET, rpl]

    def attach_mcast(self, prog):
        """The W_r column family (rotating sender, handshake) on the router NCRISC: after prog.semaphores is set."""
        lo, hi = self.cores[0], self.cores[-1]
        assert lo.x == hi.x and hi.y - lo.y + 1 == len(self.cores), "the router cores must be one column rectangle"
        rect = ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi)])
        mc = ttnn.Mcast1D(
            self.mesh,
            rect,
            ttnn.Mcast1DShape.PerColumn,
            ttnn.Mcast1DRotatingSenderConfig(),
            ttnn.McastConfig(handshake=True, noc=ttnn.NOC.NOC_0),
        )
        mc.attach(prog, "rw", [self.ncrisc])

    def pieces(self, coord, x, w, b, ix, out):
        """(CBs, kernels) of the router cores on chip coord. x: the own shard [rows, H / TP] (DRAM)."""
        plan, G, ET, RB, K = self.plan, self.G, self.ET, self.RB, self.K
        p = self.xg.chips[coord]["rings"][0]["p"]
        virt = self.mesh.worker_core_from_logical_core
        ports = {(d, l): virt(c) for (_, d, l), c in self.xg.chips[coord]["ports"].items()}
        b0, off = self.base, self.off
        cbs = []

        def cb(i, key, size, dt, page):
            dd = ttnn.cb_descriptor_from_sharded_tensor(
                i, self.arena, address_offset=off[key], total_size=size, core_ranges=self.crs
            )
            dd.format_descriptors = [ttnn.CBFormatDescriptor(i, dt, page)]
            cbs.append(dd)

        cb(XR, "xr", 2 * G * self.KBR * 2048, ttnn.bfloat16, 2048)
        cb(WR, "wr", 2 * self.KBR * ET * 2048, ttnn.bfloat16, 2048)
        cb(HOLD, "hold", G * ET * 4096, ttnn.float32, 4096)
        cb(ARRF, "arrf", max(1, plan.slots["fwd"]) * ET * 4096, ttnn.float32, 4096)
        cb(ARRB, "arrb", max(1, plan.slots["bwd"]) * ET * 4096, ttnn.float32, 4096)
        cb(BIAS, "bias", ET * 4096, ttnn.float32, 4096)
        cb(SC, "sc", ET * 4096, ttnn.float32, 4096)
        cb(IDX, "idx", ET * 2048, ttnn.uint16, 2048)
        cb(BEST, "best", ET * 4096, ttnn.float32, 4096)
        cb(BESTI, "besti", ET * 2048, ttnn.uint16, 2048)
        cb(FINC, "finc", 16 * 8, ttnn.bfloat16, 16)
        cb(PSYNC, "psync", 16 * 2, ttnn.bfloat16, 16)
        n_rt, c_rt, b_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for j, core in enumerate(self.cores):
            ents = plan.entries(p, plan.half(j))
            arr = plan.arrivals(p, j, 0)
            n_rt[core.x][core.y] = (
                [int(x.buffer_address()), int(w.buffer_address()), int(b.buffer_address()), int(ix.buffer_address())]
                + [bb * RB + j for bb in ents]
                + [len(arr["fwd"])]
                + [t for _, _, t in arr["fwd"]]
                + [len(arr["bwd"])]
                + [t for _, _, t in arr["bwd"]]
                + self.arr
                + [b0 + off["words"]]
            )
            ops = [(k, e) for k, e in plan.ops(p, j, 0) if k != "pass"]
            ce = []
            for e, bb in enumerate(ents):
                dd, _ = plan.role(p, bb, plan.half(j))
                sl = {d_: [s_ for e_, s_, _ in arr[d_] if e_ == e] for d_ in DIRS}
                mode = (1 if sl["fwd"] else 0) | (2 if sl["bwd"] else 0) | (4 if dd == "fin" else 0)
                ce += [mode, sl["fwd"][0] if sl["fwd"] else 0, sl["bwd"][0] if sl["bwd"] else 0]
            c_rt[core.x][core.y] = (
                [len(ops)] + [(1 if k == "hand" else 2) | e << 8 for k, e in ops] + ce + [b0 + off["words"]]
            )
            hs = []
            for e, d in plan.handoffs(p, j, 0):
                l = plan.link(j)
                units = plan.port_units(p, d)
                u = [ue for ue, *_ in units].index(e)
                js = self._port_layout(l)
                i = js.index(j)
                pc = ports[(d, l)]
                hs += [e, pc.x, pc.y, self.pbase + (u * len(js) + i) * ET * 4096, self.words + 16 * i]
            b_rt[core.x][core.y] = (
                [len(hs) // 5]
                + hs
                + [int(out.buffer_address()), j, self.scale_bits, b0 + off["out"], b0 + off["words"]]
                + [b0 + off["best"], b0 + off["besti"]]
            )
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        ct = [self.KT, ET, G, self.KBR, K]
        comp = ttnn.ComputeConfigDescriptor()
        comp.math_fidelity = ttnn.MathFidelity.HiFi4
        comp.fp32_dest_acc_en = True
        comp.dst_full_sync_en = True
        comp.math_approx_mode = False
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[SC] = modes[BEST] = ttnn.UnpackToDestMode.UnpackToDestFp32  # (the scores: fp32 into DEST, unrounded)
        comp.unpack_to_dest_mode = modes
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        self.ncrisc = ttnn.KernelDescriptor(
            kernel_source=_NCRISC,
            source_type=st,
            core_ranges=self.crs,
            compile_time_args=ct + acc(x) + acc(w) + acc(b) + acc(ix),
            runtime_args=n_rt,
            config=dm("RISCV_1", ttnn.NOC.NOC_0),
        )
        kernels = [
            self.ncrisc,
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=st,
                core_ranges=self.crs,
                compile_time_args=ct,
                runtime_args=c_rt,
                config=comp,
            ),
            ttnn.KernelDescriptor(
                kernel_source=_BRISC,
                source_type=st,
                core_ranges=self.crs,
                compile_time_args=ct + acc(out),
                runtime_args=b_rt,
                config=dm("RISCV_0", ttnn.NOC.NOC_1),
            ),
        ]
        return cbs, kernels
