# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""router_stream - the fused head's router on its own cores, under the x chain: logits = x_own [rows, H / TP] @
W_r[H / TP, E] (the router's H-split slice, bf16, HiFi3, fp32 out) -> the logits to their owners over TP (fp32 on
the wire, measured - bf16 changes 3-6% of the rows' picks): every chip's partial straight to its owner (multi-hop
unicast through the ports; no relay chain - one transfer of latency), summed there in fp32 DEST -> scores =
sigmoid(logits), biased b = scores + bias -> top-K by b, weights = the picked scores / their sum x route_scale -> the
joint routing rows (router_topk's format: per token K expert ids then K bf16 weights, [owned / 32, 32 x 2 K] uint16).

Cores: RB = CR / TP router cores (one chunk of CR row tiles; core j owns row tile j of every block, i.e. of every
chip's tokens), one column of the mover column. Core j: its x rows and W_r^T streamed in K blocks of KBR (W_r^T read
from DRAM once: a rotating column multicast), logits computed TRANSPOSED (W_r^T x x^T: a token is a tile column), the
G x ET fp32 partials accumulated in L1 (packer L1 accumulation), then its partials of the other owners handed to the
ports (farthest first), its own block + the G - 1 arrivals in fp32 DEST -> sigmoid -> + bias -> scores, the top-K on the
SFPU (ttnn.topk's bitonic network over each token column, fp32), the joint rows (BRISC) for its 32 owner tokens. Hand-offs are pushed into the port's L1
(the sub-block, then +1 on the port's word of this core); the port's sender polls them between its other packets (the
per-model priority) and sends them into the downstream router core's zone with a fused increment."""

import struct

import torch

import ttnn
from ttnn.bringup.combine_rf import persist_l1

XR, WR, HOLD, ARRF, ARRB, BIAS, SC, FINC, PSYNC, BEST, BESTI, IDX = range(12)  # (BESTI unused)
PART = 13  # (split: the partner's best K, half 0)
_NCRISC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
using namespace dataflow_kernel_lib;
constexpr uint32_t KT = get_compile_time_arg_val(0), ET = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr uint32_t KBR = get_compile_time_arg_val(3);
constexpr auto x_args = TensorAccessorArgs<5>();
constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto i_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr uint32_t XR = 0, WR = 1, ARRF = 3, ARRB = 4, BIAS = 5, IDX = 11, K = get_compile_time_arg_val(4), RDY = 12;
// args: x, W_r^T, bias^T, expert index tiles; my rows [G]; the arrivals (na: every other chip's partial of my owner
// rows, slot = its source index) and their words (a word a slot, 16 B apart); the slots in their fixed sum order [G - 1]
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1), b_addr = get_arg_val<uint32_t>(2);
    const uint32_t i_addr = get_arg_val<uint32_t>(3);
    const uint32_t r0 = 4, na = get_arg_val<uint32_t>(r0 + G);
    volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(r0 + G + 1));
    const uint32_t ab0 = r0 + G, nab = G - 1;  // (the order: ab0 + 2 ..; the tail: ab0 + nab + 2 ..)
    {  // (the top-K's done words: zero before anyone can write them - they are written after the scores; with the
       // split word 128 too, the partner's best K landed - the arena is call scratch: stale, it let the merge take a
       // stale page, the picks a call behind)
        const uint32_t w0 = get_arg_val<uint32_t>(ab0 + nab + 2);
        for (uint32_t i = 0; i < 9; ++i) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w0 + 16 * i) = 0;
    }
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const auto wg = TensorAccessor(w_args, w_addr, 2048);
    const auto bg = TensorAccessor(b_args, b_addr, 4096);
    const auto ig = TensorAccessor(i_args, i_addr, 2048);
    cb_reserve_back(BIAS, RT_ETF);
    cb_reserve_back(IDX, ET);
    for (uint32_t t = 0; t < RT_ETF; ++t) noc_async_read(bg.get_noc_addr(t), get_write_ptr(BIAS) + t * 4096, 4096);
    for (uint32_t t = 0; t < ET; ++t) noc_async_read(ig.get_noc_addr(RT_HB + t), get_write_ptr(IDX) + t * 2048, 2048);
    noc_async_read_barrier();
    cb_push_back(BIAS, RT_ETF);
    cb_push_back(IDX, ET);
#ifdef RT_ENG
    {  // the engine computed my partial logits (every engine core its expert tiles x its rows, written into my HOLD
       // with an increment of this counter a tile): all G x ET in -> the compute (RDY)
        DeviceZoneScopedN("SE-RENGWAIT");
        volatile tt_l1_ptr uint32_t* ec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_ENG);
        noc_semaphore_wait_min(ec, ET);  // (my own block's tiles: the others' go straight to the ports)
        noc_semaphore_inc(get_noc_addr(RT_ENG), 0u - ET);  // (re-arm)
        noc_async_atomic_barrier();
        cb_reserve_back(RDY, 1);
        cb_push_back(RDY, 1);
        // (and word 112 for the compute's PACK thread: the RDY wait holds only UNPACK, and PACK's hand-off pushes
        // (FINC) would hand HOLD to the ports before the engine's partials landed - stale HOLD: the last call's
        // planner overlay in it, K3's picks 99.10%)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ab0 + nab + 2) + 112) = 1;
#ifdef RT_GATE
        if (get_arg_val<uint32_t>(4) == 0) {  // (router core 0: every engine core's partials written (its count) -> the
                                              // gate word on every engine core: their W streams start)
            volatile tt_l1_ptr uint32_t* gc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_GATE_CTR);
            noc_semaphore_wait_min(gc, RT_GATE_N);
            noc_semaphore_inc(get_noc_addr(RT_GATE_CTR), 0u - RT_GATE_N);  // (re-arm)
            noc_async_atomic_barrier();
            volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_GATE_SRC);
            *one = 1;
            noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(one),
                                        get_noc_multicast_addr(RT_GATE_X0, RT_GATE_Y0, RT_GATE_X1, RT_GATE_Y1, RT_GATE), RT_GATE_N);
            noc_async_write_barrier();
        }
#endif
    }
#else
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
#endif
    if (na) {  // the sources' partials as they land (a fused increment of the source slot's own word, 16 B apart): each
               // slot to the compute in a fixed order (RT_ARR_ORDERED: the order args - the predicted landing order, so
               // the same sum every run at ~no wait), else in landing order (ARRB) - it sums the early ones while the
               // last is on its way
        DeviceZoneScopedN("SE-RLWAIT");
        for (uint32_t a = 0, seen = 0; a < na;) {
            invalidate_l1_cache();
            for (uint32_t s = 0; s < na; ++s) {
#ifdef RT_ARR_ORDERED
                if (s != get_arg_val<uint32_t>(ab0 + 2 + a)) {  // (the fixed order)
                    continue;
                }
#endif
                if ((seen >> s & 1) || ctr[4 * s] == 0) {
                    continue;
                }
                ctr[4 * s] = 0;  // (re-arm: one increment a call - the source's next one comes after this chip's exchange)
                seen |= 1u << s;
                ++a;
                cb_reserve_back(ARRB, 1);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(ARRB)) = s;
                cb_push_back(ARRB, 1);
            }
        }
    }
#ifdef RT_SPLIT
#if RT_HALF == 0
    {  // (my partner's best K - the row tile's other expert half - written into my PART: to the compute's merge)
        volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RT_PWORD);
        while (*pw == 0) invalidate_l1_cache();
        *pw = 0;  // (re-arm: one a call)
        cb_reserve_back(RT_PART_CB, 1);
        cb_push_back(RT_PART_CB, 1);
    }
#endif
#endif
    {  // (my share of the joint rows - tokens 8 .. 15 - once the top-K is done; split half 1: of its picks' scores)
        const uint32_t t2 = ab0 + nab + 2, words = get_arg_val<uint32_t>(t2);
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 64);
        while (*done == 0) invalidate_l1_cache();
        *done = 0;
        DeviceZoneScopedN("SE-RROWS");
        set_l1_data_cache<true>();
        invalidate_l1_cache();
#if RT_HALF == 0
        rows_::make<K, RT_ETF, ET>(get_arg_val<uint32_t>(t2 + 1), get_arg_val<uint32_t>(t2 + 4), get_arg_val<uint32_t>(t2 + 2),
                                   get_arg_val<uint32_t>(t2 + 3), 8, 16);
#else
        rows_::picked<K, RT_ETF, ET>(get_arg_val<uint32_t>(t2 + 1), get_arg_val<uint32_t>(t2 + 4), 8, 16);
#endif
        set_l1_data_cache<false>();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16) = 1;
    }
#ifdef RA_PLAN
    {  // the second planner instance on this core (the routing exchange + planner: routing_stream)
        const uint32_t rr = ab0 + nab + 2 + 5, rows2 = get_arg_val<uint32_t>(rr), B = get_arg_val<uint32_t>(rr + 1);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows2), RA_NP_);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows2) = 0;
        if (B) ra::plan(rr + 2, B);  // (no instance here when the chip has fewer experts than planner RISCs)
    }
#endif
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
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#ifdef RT_SQRTSOFTPLUS
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#endif
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t KT = get_compile_time_arg_val(0), ET = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr uint32_t KBR = get_compile_time_arg_val(3), K = get_compile_time_arg_val(4);
constexpr uint32_t XR = 0, WR = 1, HOLD = 2, ARRF = 3, ARRB = 4, BIAS = 5, SC = 6, FINC = 7, PSYNC = 8;
constexpr uint32_t BEST = 9, IDX = 11;
constexpr uint32_t EM = rows_::em_of(RT_ETF);
#ifdef TRISC_MATH
namespace ckernel::sfpu {
// a score tile's top-K keys: the fp32 score with its low mantissa bits (EM) replaced by the expert id (the uint16 id
// tile two DEST tiles on): ttnn.topk's fused keys at fp32 - 9 bits instead of bf16, raw INT32 words
template <uint32_t M>
inline void rt_key() {
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = M;
    for (int i = 0; i < 32; i++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 128);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);  // the id (the high half of a uint16 in 32-bit DEST: junk)
        TTI_SFPOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // the score's low bits cleared
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
}  // namespace ckernel::sfpu
#endif
// args: nops, ops (kind | entry << 8: 1 hand, 2 fin), per entry: arrivals to add (the fin: G - 1, from the zone's
// slots 0 ..; a hand-off: 0); then the top-K's done word. Every tile here is TRANSPOSED: logits^T [ET][32 tokens] - a token is a column.
void kernel_main() {
    // (the PACK -> MATH word: zero before anything - L1 scratch, PACK sets it only after the top-K)
    MATH((*reinterpret_cast<volatile uint32_t*>(RT_MSYNC) = 0));
    compute_kernel_hw_startup<SrcOrder::Reverse>(WR, XR, HOLD);
    pack_reconfig_data_format(HOLD);
#ifdef RT_ENG
    // (the matmul below is the router cores' own path: off here)
    cb_wait_front(12, 1);  // (the engine's partial logits are in my HOLD: RDY - UNPACK; PACK on word 112)
    PACK(({
        volatile uint32_t* hw = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(1 + get_arg_val<uint32_t>(0) + G) + 112);
        while (*hw == 0) {
        }
        *hw = 0;  // (re-arm)
    }));
    if (false)
#endif
    {
    DeviceZoneScopedN("SE-RMMUL");
    for (uint32_t kb = 0; kb < KT / KBR; ++kb) {  // K-outer: every token tile's partial accumulated in L1 (fp32)
        cb_wait_front(XR, G * KBR);
        cb_wait_front(WR, KBR * ET);
        PACK((llk_pack_reconfig_l1_acc(kb > 0 ? 1 : 0)));
        for (uint32_t e0 = 0; e0 < ET; ++e0) {  // (logits^T [e0] x [G token tiles]: W_r^T x x^T; half-sync DEST: the
            constexpr uint32_t re = 1;           // packs of one block under the next block's math)
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
    cb_pop_front(PSYNC, 1);
    cb_wait_front(BIAS, RT_ETF);
    const uint32_t nops = get_arg_val<uint32_t>(0), et0 = 1 + nops;
    for (uint32_t o = 0; o < nops; ++o) {
        const uint32_t op = get_arg_val<uint32_t>(1 + o), kind = op & 0xFF, e = op >> 8;
        if (kind == 1) {
            cb_reserve_back(FINC, 1);
            cb_push_back(FINC, 1);
            continue;
        }
        const uint32_t na = get_arg_val<uint32_t>(et0 + e);  // (the one fin: the owner's - G - 1 sources)
        cb_reserve_back(SC, ET);
        cb_wait_front(IDX, ET);
        pack_reconfig_data_format(HOLD);
        // the sources' partials in landing order (ARRB: the slot): all but the last summed into my block's HOLD as they
        // land (fp32 on the SFPU, packed back in place), the last in the passes below - after the last arrival only its
        // add is left
        uint32_t last = 0;
        for (uint32_t a = 0; a < na; ++a) {
            cb_wait_front(ARRB, 1);
            const uint32_t sl = read_tile_value(ARRB, 0, 0);
            cb_pop_front(ARRB, 1);
            if (a + 1 == na) {
                last = sl;
                break;
            }
            for (uint32_t i0 = 0; i0 < ET; i0 += 2) {
                const uint32_t n = ET - i0 < 2 ? ET - i0 : 2;
                tile_regs_acquire();
                reconfig_data_format_srca(HOLD);
                copy_init(HOLD);
                for (uint32_t j = 0; j < n; ++j) copy_tile(HOLD, e * ET + i0 + j, j);
                reconfig_data_format_srca(HOLD, ARRF);
                copy_init(ARRF);
                for (uint32_t j = 0; j < n; ++j) copy_tile(ARRF, sl * ET + i0 + j, 2 + j);
                add_binary_tile_init();
                for (uint32_t j = 0; j < n; ++j) add_binary_tile(j, 2 + j, j);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < n; ++j) pack_tile<true>(j, HOLD, e * ET + i0 + j);
                tile_regs_release();
            }
            cb_reserve_back(PSYNC, 1);  // (PACK -> UNPACK: the sums are in HOLD before the next pass reads them)
            cb_push_back(PSYNC, 1);
            cb_wait_front(PSYNC, 1);
            cb_pop_front(PSYNC, 1);
        }
        { DeviceZoneScopedN("SE-RFARR"); }  // (a mark: every arrival in - the fin from here to the top-K)
        // (all in DEST at fp32: the partials, the bias unpacked straight to DEST, the adds on the SFPU - the FPU's
        // srcA / srcB hold TF32, a 10-bit mantissa: the sums and the bias add through them moved scores by up to ~1e-3
        // and the picks of close experts with them)
        // (a score tile a pass: DEST 0 its score, 1 its key, 2 / 3 the operands - the score is kept: packed back over
        // its logits in HOLD, where the joint rows read the picks' weights exactly; the key's low bits are the id)
        for (uint32_t i = 0; i < ET; ++i) {
            tile_regs_acquire();
            reconfig_data_format_srca(HOLD);
            copy_init(HOLD);
            copy_tile(HOLD, e * ET + i, 0);
            if (na) {  // (+ the last source's partial)
                reconfig_data_format_srca(HOLD, ARRF);
                copy_init(ARRF);
                copy_tile(ARRF, last * ET + i, 2);
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
            }
            {  // the owner: scores = sigmoid(logits) (RT_SQRTSOFTPLUS: sqrt(softplus)), biased = scores + bias (bias^T: per row)
#ifdef RT_SQRTSOFTPLUS
                softplus_tile_init();  // (INP_FLOAT32: the fp32 polynomial, as stock ttnn.softplus on fp32; beta 1, threshold 20)
                softplus_tile(0, 0x3F800000u, 0x3F800000u, 0x41A00000u);
                sqrt_tile_init();
                sqrt_tile(0);
#else
                sigmoid_tile_init();
                sigmoid_tile(0);
#endif
                reconfig_data_format_srca(ARRF, BIAS);
                copy_init(BIAS);
                copy_tile(BIAS, RT_HB + i, 2);
                add_binary_tile_init();
                add_binary_tile(0, 2, 1);
                // -> the top-K's keys: the expert ids into the low mantissa bits (rt_key: the id tile two on, DEST 3)
                reconfig_data_format_srca(BIAS, IDX);
                copy_init(IDX);
                copy_tile(IDX, i, 3);
                MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::rt_key<EM>, 1, VectorMode::RC_custom)));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(1, SC, i);
            pack_tile<true>(0, HOLD, e * ET + i);
            tile_regs_release();
        }
        cb_push_back(SC, ET);
        // the owner's 32 tokens' top-K on the SFPU (ttnn.topk's bitonic network), each token a column, fp32 straight
        // to DEST (no srcA rounding): the running top K of every column in DEST tile 0, each next 32 experts copied
        // into tile 1 as keys (rt_key: the expert id in the low mantissa bits - one word a datum, no id tiles riding
        // the swaps); the 64-wide column sorted into K-runs (bitonic phases 0 .. log K - 1), then merged run against
        // run down to one sorted K-run on top (log(64 / K) merge + rebuild rounds) - no full 64-wide sorts, nothing
        // packed until the end
        DeviceZoneScopedN("SE-RTOPK");
        cb_wait_front(SC, ET);
        // (the network's run KS: K's power of two, at least 4 - K 6 sorts the top 8, the rows take its first K)
        constexpr uint32_t KS = K <= 4 ? 4 : K <= 8 ? 8 : K <= 16 ? 16 : 32;
        constexpr uint32_t LK = KS == 8 ? 3 : KS == 16 ? 4 : KS == 32 ? 5 : 2, NM = 6 - LK;
        constexpr auto TIE = ckernel::TopkTieOrder::Unset;
        tile_regs_acquire();
        reconfig_data_format_srca(SC);
        copy_init(SC);
        ckernel::topk_tile_init<true, false>();
        for (uint32_t t = 0; t < ET; t += 2) {
            const bool odd = t + 1 == ET;  // (a last tile alone: against the running top K, a full round)
            const bool fold = t > 0 && !odd;  // (a pair after the first: its run folded into the running top K)
            if (odd) {
                copy_dest_values<DataFormat::Int32>(2, 0);
                copy_tile(SC, t, 1);
            } else {
                copy_tile(SC, t, 0);
                copy_tile(SC, t + 1, 1);
            }
            ckernel::topk_local_sort<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, LK - 1);
            for (uint32_t m = 0; m + 1 < NM; ++m) {
                ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, m, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, m, KS, LK, false);
            }
            if (!fold) {  // the top K of the 64 a column on top: tile 0's run, descending
                ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, NM - 1, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, NM - 1, KS, LK, true);
            } else {  // the pair's top K into tile 1's run, ascending; the running top K (descending) into tile 0 ->
                      // one bitonic merge (max on top) + rebuild of K
                ckernel::topk_merge<true, false, DST_ACCUM_MODE, true, false, TIE>(0, NM - 1, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, NM - 1, KS, LK, false);
                copy_dest_values<DataFormat::Int32>(2, 0);
                ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, NM - 1, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, NM - 1, KS, LK, true);
            }
            if (t + 2 < ET) copy_dest_values<DataFormat::Int32>(0, 2);  // (parked in DEST tile 2: fused keys leave the id tiles unused)
        }
#if defined(RT_SPLIT) && RT_HALF == 0
        {  // my best K (DEST tile 0) against my partner's (its BEST page - the other expert half, its candidates
           // below it - into DEST tile 1): one round as a first pair, the K best of the 64 on top (keys unique - the
           // expert id in the low bits: the same picks as one core over every tile)
            DeviceZoneScopedN("SE-RMERGE");
            cb_wait_front(RT_PART_CB, 1);
            reconfig_data_format_srca(RT_PART_CB);
            copy_init(RT_PART_CB);
            copy_tile(RT_PART_CB, 0, 1);
            ckernel::topk_tile_init<true, false>();
            ckernel::topk_local_sort<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, LK - 1);
            for (uint32_t m = 0; m + 1 < NM; ++m) {
                ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, m, KS);
                ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, m, KS, LK, false);
            }
            ckernel::topk_merge<false, false, DST_ACCUM_MODE, true, false, TIE>(0, NM - 1, KS);
            ckernel::topk_rebuild<false, DST_ACCUM_MODE, true, false, TIE>(0, 0, NM - 1, KS, LK, true);
            cb_pop_front(RT_PART_CB, 1);
        }
#endif
        tile_regs_commit();
        cb_reserve_back(BEST, 1);
        tile_regs_wait();
        pack_reconfig_data_format(BEST);
        pack_tile(0, BEST);
        tile_regs_release();
        cb_push_back(BEST, 1);
        const uint32_t t1 = et0 + G, words = get_arg_val<uint32_t>(t1);
        PACK(({  // (the last best pair packed: the joint rows can go - BRISC and NCRISC told)
            *reinterpret_cast<volatile uint32_t*>(words) = 1;
            *reinterpret_cast<volatile uint32_t*>(words + 64) = 1;
        }));
        {  // the joint rows' last two shares (tokens 16 .. 23 the MATH thread, 24 .. 31 the PACK thread; soft float):
           // the BEST page is in L1 once PACK passed its push - PACK tells MATH (RT_MSYNC)
            const uint32_t bvc = get_arg_val<uint32_t>(t1 + 1), olc = get_arg_val<uint32_t>(t1 + 2);
            const uint32_t scl = get_arg_val<uint32_t>(t1 + 3), sv = get_arg_val<uint32_t>(t1 + 4);
#if RT_HALF == 0
            PACK(({
                *reinterpret_cast<volatile uint32_t*>(RT_MSYNC) = 1;
                rows_::make<K, RT_ETF, ET>(bvc, sv, olc, scl, 24, 32);
                *reinterpret_cast<volatile uint32_t*>(words + 48) = 1;
            }));
            MATH(({
                volatile uint32_t* ms = reinterpret_cast<volatile uint32_t*>(RT_MSYNC);
                while (*ms == 0) {
                }
                rows_::make<K, RT_ETF, ET>(bvc, sv, olc, scl, 16, 24);
                *reinterpret_cast<volatile uint32_t*>(words + 32) = 1;
            }));
#else  // (half 1 hands its best K to its partner, with its picks' scores: no rows)
            (void)olc, (void)scl;
            PACK(({
                *reinterpret_cast<volatile uint32_t*>(RT_MSYNC) = 1;
                rows_::picked<K, RT_ETF, ET>(bvc, sv, 24, 32);
                *reinterpret_cast<volatile uint32_t*>(words + 48) = 1;
            }));
            MATH(({
                volatile uint32_t* ms = reinterpret_cast<volatile uint32_t*>(RT_MSYNC);
                while (*ms == 0) {
                }
                rows_::picked<K, RT_ETF, ET>(bvc, sv, 16, 24);
                *reinterpret_cast<volatile uint32_t*>(words + 32) = 1;
            }));
#endif
        }
#ifdef RT_ROUTE
        // dispatch prep: the route words of my 32 owned tokens from the joint rows (BRISC's word 80: all shares
        // written), into L1 for the BRISC's write (word 96); off the routing chain (the UNPACK thread is idle here)
        const uint32_t rl1 = get_arg_val<uint32_t>(t1 + 5), tok0 = get_arg_val<uint32_t>(t1 + 6);
        const uint32_t ol1 = get_arg_val<uint32_t>(t1 + 2);
        UNPACK(({
            volatile uint32_t* rd = reinterpret_cast<volatile uint32_t*>(words + 80);
            while (*rd == 0) {
            }
            *rd = 0;
            volatile uint32_t* o = reinterpret_cast<volatile uint32_t*>(rl1);
            for (uint32_t c = 0; c < 32; ++c) rw_::word(o, c, reinterpret_cast<volatile const uint16_t*>(ol1 + c * 4 * K), tok0 + c);
            *reinterpret_cast<volatile uint32_t*>(words + 96) = 1;
        }));
#endif
    }
}
"""

# the joint rows of owner tokens [c0, c1) of the row tile from the top-K's last best pair (any RISC: soft float, so
# the 32 tokens are split over four): ids = rows 0 .. K - 1 of the token's column, weights = the picks' scores / their
# sum x scale, bf16 (round to nearest even). The scores are the fin's fp32 sigmoids, read by id where the fin packed
# them (over its logits in HOLD) - not the keys' biased scores less the bias: a key keeps 13-15 mantissa bits of score +
# bias, which lost the score whenever the bias dwarfed it (a score 6e-6 under a bias 0.125: weight 0)
_ROWS = r"""
#ifndef RT_FOLD
#define RT_FOLD 1
#endif
// (RT_SPLIT: a row tile's experts over two router cores - RT_HALF 0 / 1, its expert tiles from RT_HB, ET of them;
// RT_ETF every expert tile: the bias, the keys' id bits and the joint rows are the whole row tile's)
#ifndef RT_ETF
#define RT_ETF ET
#endif
#ifndef RT_HB
#define RT_HB 0
#endif
#ifndef RT_HALF
#define RT_HALF 0
#endif
namespace rows_ {
// (r, c) of a tile: face (r / 16) * 2 + c / 16, then row-major within the face
FORCE_INLINE uint32_t fo(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
// the index of x's top bit (x > 0): five steps (no Zbb on the RISCs: __builtin_clz is a library loop)
FORCE_INLINE int32_t msb32(uint32_t x) {
    int32_t m = 0;
    if (x >> 16) x >>= 16, m += 16;
    if (x >> 8) x >>= 8, m += 8;
    if (x >> 4) x >>= 4, m += 4;
    if (x >> 2) x >>= 2, m += 2;
    if (x >> 1) m += 1;
    return m;
}
// x (> 0, < 2^31): a value scaled by 2^f -> bf16 bits, round to nearest even
FORCE_INLINE uint32_t bf16_of(uint32_t x, int32_t f) {
    if (!x) return 0;
    const int32_t msb = msb32(x);
    uint32_t t;
    int32_t e = msb;
    if (msb > 7) {  // (keep 8 bits: the hidden one + 7)
        const int32_t drop = msb - 7;
        const uint32_t half = 1u << (drop - 1), rem = x & ((1u << drop) - 1);
        t = x >> drop;
        if (rem > half || (rem == half && (t & 1))) ++t;
        if (t >> 8) t >>= 1, ++e;
    } else {
        t = x << (7 - msb);
    }
    return ((uint32_t)(e - f + 127) << 7) | (t & 0x7F);
}
// the expert id bits of a top-K key (the low mantissa bits of its fp32 score)
constexpr uint32_t em_of(uint32_t et) { return et * 32 <= 256 ? 0xFF : et * 32 <= 512 ? 0x1FF : 0x3FF; }
// the fp32 score of expert eid (one of mine: expert tiles RT_HB .., ETL of them) for token column c - the fin's score
// tiles at sv, transposed like the logits: (eid % 32, c) of tile eid / 32
template <uint32_t ETL>
FORCE_INLINE uint32_t score_of(uint32_t sv, uint32_t eid, uint32_t c) {
    return reinterpret_cast<const uint32_t*>(sv + ((eid >> 5) - RT_HB) * 4096)[fo(eid & 31, c)];
}
// (split half 1) its picks' scores, for its partner's rows: the score of the key at (q, c) of the best page at (q, c)
// of the page after its own PART page (sent with the best page)
template <uint32_t K, uint32_t ET, uint32_t ETL>
inline void picked(uint32_t bv, uint32_t sv, uint32_t c0, uint32_t c1) {
#ifdef RT_PART
    constexpr uint32_t EM = em_of(ET);
    for (uint32_t c = c0; c < c1; ++c)
        for (uint32_t q = 0; q < K; ++q) {
            const uint32_t key = reinterpret_cast<const uint32_t*>(bv)[fo(q, c)];
            reinterpret_cast<uint32_t*>(RT_PART + 4096)[fo(q, c)] = score_of<ETL>(sv, key & EM, c);
        }
#else
    (void)bv, (void)sv, (void)c0, (void)c1;
#endif
}
template <uint32_t K, uint32_t ET, uint32_t ETL>
inline void make(uint32_t bv, uint32_t sv, uint32_t o_l1, uint32_t scale_bits, uint32_t c0, uint32_t c1) {
    // weights = score / their sum x scale in integers (the RISCs have no FPU, no 64-bit divide): the K scores in one
    // block exponent (the largest's top bit at 25: a sum of up to 32 below 2^31), the sum normalized to bit 30, its
    // reciprocal by one 32-bit divide (16 bits: a bf16 weight needs 8 + rounding), x the scale's mantissa, then one
    // 32 x 32 -> 64 product a weight
    const uint32_t sc = (scale_bits & 0x7FFFFF) | 0x800000;  // (scale = sc x 2^se)
    const int32_t se = (int32_t)((scale_bits >> 23) & 0xFF) - 150;
    constexpr uint32_t EM = em_of(ET);
    for (uint32_t c = c0; c < c1; ++c) {
        uint32_t ids[K], w[K], sum = 0;  // (w: the scores' fp32 bits, then their block-exponent integers)
        int32_t emax = 0;
        for (uint32_t q = 0; q < K; ++q) {
            const uint32_t key = reinterpret_cast<const uint32_t*>(bv)[fo(q, c)];  // (the top-K's key: id in the low bits)
            const uint32_t eid = key & EM;
            ids[q] = eid;
            if ((eid >> 5) - RT_HB < ETL) {
                w[q] = score_of<ETL>(sv, eid, c);
            } else {  // (split: the partner half's pick - its key and score at the same place of its two PART pages)
                w[q] = 0;
#ifdef RT_PART
                for (uint32_t r = 0; r < K; ++r)
                    if (reinterpret_cast<const uint32_t*>(RT_PART)[fo(r, c)] == key) {
                        w[q] = reinterpret_cast<const uint32_t*>(RT_PART + 4096)[fo(r, c)];
                        break;
                    }
#endif
            }
            if (w[q] >> 31) w[q] = 0;  // (sigmoid and sqrt(softplus) scores are >= 0)
            const int32_t e = (w[q] >> 23) & 0xFF;
            if (e > emax) emax = e;
        }
        for (uint32_t q = 0; q < K; ++q) {
            const int32_t e = (w[q] >> 23) & 0xFF, sh = e - emax + 2;  // (mantissa top bit 23 -> 25 for the largest)
            const uint32_t m = (w[q] & 0x7FFFFF) | 0x800000;
            w[q] = e == 0 ? 0 : sh >= 0 ? (m << sh) : (sh > -31 ? (m >> -sh) : 0);
            sum += w[q];
        }
        uint16_t* o = reinterpret_cast<uint16_t*>(o_l1 + c * 4 * K);
        const int32_t sh = sum ? 30 - msb32(sum) : 0;
        const uint32_t sn = sh >= 0 ? sum << sh : sum >> -sh;
        const uint32_t r1 = sn ? 0xFFFFFFFFu / (sn >> 14) : 0;  // (2^32 / (sn / 2^14): 2^62 / sn = r1 x 2^16)
        const uint32_t rs = (uint32_t)(((uint64_t)r1 * sc) >> 8);  // (weight = wn x rs / 2^32 x 2^(se - 6))
        for (uint32_t q = 0; q < K; ++q) {
            o[q] = static_cast<uint16_t>(ids[q] / RT_FOLD);  // (candidate -> expert: RouterSpec.fold)
            const uint32_t wn = sh >= 0 ? (uint32_t)w[q] << sh : (uint32_t)w[q] >> -sh;
            o[K + q] = static_cast<uint16_t>(bf16_of((uint32_t)(((uint64_t)wn * rs) >> 32), 6 - se));
        }
    }
}
}  // namespace rows_
"""

_BRISC = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t ET = get_compile_time_arg_val(1), K = get_compile_time_arg_val(4);
constexpr auto o_args = TensorAccessorArgs<5>();
constexpr uint32_t HOLD = 2, BIAS = 5, FINC = 7, BEST = 9;
"""
    + _ROWS
    + r"""
// args: nh, per hand-off (entry, port x, port y, port L1 offset of the sub-block, port word address); the out address,
// my owner page, route scale bits, the joint rows' L1 scratch, the done word, the BEST page
void kernel_main() {
    const uint32_t nh = get_arg_val<uint32_t>(0), h0 = 1;
    const uint32_t hold = get_read_ptr(HOLD);
    for (uint32_t j = 0; j < nh; ++j) {
        DeviceZoneScopedN("SE-RHAND");
        cb_wait_front(FINC, j + 1);
        const uint32_t e = get_arg_val<uint32_t>(h0 + 6 * j), px = get_arg_val<uint32_t>(h0 + 6 * j + 1);
        const uint32_t py = get_arg_val<uint32_t>(h0 + 6 * j + 2), sw = get_arg_val<uint32_t>(h0 + 6 * j + 5);
        if (sw) {  // (the port's sub-block slot is shared with an earlier core of my link: once the port sent its)
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sw);
            while (*f == 0) invalidate_l1_cache();
            *f = 0;  // (re-arm: one a call - the next comes after this call's hand-offs)
        }
        noc_async_write(hold + e * ET * 4096, get_noc_addr(px, py, get_arg_val<uint32_t>(h0 + 6 * j + 3)), ET * 4096);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_arg_val<uint32_t>(h0 + 6 * j + 4)), 1);
    }
    const uint32_t t0 = h0 + 6 * nh;
    const uint32_t o_addr = get_arg_val<uint32_t>(t0), page = get_arg_val<uint32_t>(t0 + 1);
    const uint32_t o_l1 = get_arg_val<uint32_t>(t0 + 3);
    const uint32_t words = get_arg_val<uint32_t>(t0 + 4);
    // the top K of every token column, sorted descending in rows 0 .. K - 1 of the best page;
    // tokens 0 .. 7 here, 8 .. 15 the NCRISC, 16 .. 23 the compute's MATH thread, 24 .. 31 its PACK (done words 16 / 32 / 48)
    const uint32_t bv = get_arg_val<uint32_t>(t0 + 5);
    cb_wait_front(BIAS, RT_ETF);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);
    while (*done == 0) invalidate_l1_cache();
    *done = 0;  // (re-arm)
#if RT_HALF == 1
    {  // my best K (the row tile's other expert half) into my partner's PART, its picks' scores (built here by my four
       // RISCs, beside my own PART page) after it, then its word (args: the partner's NoC x | y << 16 in the out
       // address's place)
        DeviceZoneScopedN("SE-RHALF");
        set_l1_data_cache<true>();
        invalidate_l1_cache();
        rows_::picked<K, RT_ETF, ET>(bv, get_arg_val<uint32_t>(t0 + 9), 0, 8);
        set_l1_data_cache<false>();
        for (uint32_t i = 1; i < 4; ++i) {
            volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * i);
            while (*wd == 0) invalidate_l1_cache();
            *wd = 0;  // (re-arm)
        }
        const uint32_t pxy = get_arg_val<uint32_t>(t0);
        noc_async_write(bv, get_noc_addr(pxy & 0xFFFF, pxy >> 16, RT_PART), 4096);
        noc_async_write(RT_PART + 4096, get_noc_addr(pxy & 0xFFFF, pxy >> 16, RT_PART + 4096), 4096);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(pxy & 0xFFFF, pxy >> 16, RT_PWORD), 1);
        noc_async_atomic_barrier();
        (void)o_addr, (void)page, (void)o_l1;
    }
#else
    {
        {
        DeviceZoneScopedN("SE-RROWS");
        set_l1_data_cache<true>();  // (write-through; the packed tiles read after an invalidate)
        invalidate_l1_cache();
        rows_::make<K, RT_ETF, ET>(bv, get_arg_val<uint32_t>(t0 + 9), o_l1, get_arg_val<uint32_t>(t0 + 2), 0, 8);
        set_l1_data_cache<false>();
        }
        for (uint32_t i = 1; i < 4; ++i) {
            volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * i);
            while (*wd == 0) invalidate_l1_cache();
            *wd = 0;  // (re-arm)
        }
#ifdef RT_ROUTE
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 80) = 1;  // (every share of the rows written)
#endif
    }
    const auto og = TensorAccessor(o_args, o_addr, 32 * 4 * K);
    noc_async_write(o_l1, og.get_noc_addr(page), 32 * 4 * K);
    noc_async_write_barrier();
#endif
#ifdef RA_PLAN
    {  // the routing exchange + planner (routing_stream): every router core's rows written, then routing_a2a's planner
        DeviceZoneScopedN("SE-RPLAN");
        const uint32_t r0 = t0 + 10, rows = get_arg_val<uint32_t>(r0), rows2 = get_arg_val<uint32_t>(r0 + 1);
        const uint32_t B = get_arg_val<uint32_t>(r0 + 2);
#if RT_HALF == 0  // (the rows' makers: RA_NP_ of the RA_NC_ router cores - with RT_SPLIT the halves 0)
        for (uint32_t i = 0; i < RA_NC_; ++i) {  // (my rows are written: every router core's two planner RISCs told)
            const uint32_t x = get_arg_val<uint32_t>(r0 + 3 + 2 * i), y = get_arg_val<uint32_t>(r0 + 4 + 2 * i);
            noc_semaphore_inc(get_noc_addr(x, y, rows), 1);
            noc_semaphore_inc(get_noc_addr(x, y, rows2), 1);
        }
#endif
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), RA_NP_);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows) = 0;  // (re-arm: all increments landed)
        if (B) ra::plan(r0 + 3 + 2 * RA_NC_, B);  // (no instance here: fewer planner instances than router cores)
    }
#endif
#ifdef RT_ROUTE
    {  // dispatch prep: my owned tokens' route words (the compute's, word 96) to the route tensor
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 96);
        while (*rd == 0) invalidate_l1_cache();
        *rd = 0;  // (re-arm)
        const uint32_t rl1 = get_arg_val<uint32_t>(t0 + 6), tok0 = get_arg_val<uint32_t>(t0 + 8);
        const auto rg = TensorAccessor(TensorAccessorArgs<RT_CT0>(), get_arg_val<uint32_t>(t0 + 7), RT_PAGE);
        noc_async_write(rl1, rg.get_noc_addr(0) + tok0 * 4 * RT_RW, 32 * 4 * RT_RW);
        noc_async_write_barrier();
    }
#endif
}
"""
)

# the port sender's poll (x_gather3 includes it): the router cores' hand-offs landed in this port's L1, sent in the
# plan's walk (unit, core) as each core's word shows it; fp32 packets of RPL tiles, a sub-block's last one with the
# fused increment of the downstream router core's arrival word for this source's slot. One dedicated header, flushed per packet.
LSTREAM = r"""
struct LStream {
    uint32_t r0, nu, nc, buf, word, zone, actr, et, rpl, need, ns, sfree, u, i, sent;
    void init(uint32_t a) {
        r0 = a; nu = get_arg_val<uint32_t>(a); nc = get_arg_val<uint32_t>(a + 1 + 2 * nu);
        const uint32_t t = a + 2 + 2 * nu + nc;
        buf = get_arg_val<uint32_t>(t); word = get_arg_val<uint32_t>(t + 1); zone = get_arg_val<uint32_t>(t + 2);
        actr = get_arg_val<uint32_t>(t + 3); et = get_arg_val<uint32_t>(t + 4); rpl = get_arg_val<uint32_t>(t + 5);
        need = get_arg_val<uint32_t>(t + 6);  // (a sub-block's word when in: 1 hand-off, or et engine tiles)
        ns = get_arg_val<uint32_t>(t + 7);  // (sub-block slots a unit: core i in slot i % ns, then core i + ns told)
        sfree = get_arg_val<uint32_t>(t + 8);
        u = 0; i = 0; sent = 0;
    }
    bool done() const { return u >= nu; }
    static uint32_t len(uint32_t a) { const uint32_t nu = get_arg_val<uint32_t>(a); return 2 + 2 * nu + get_arg_val<uint32_t>(a + 1 + 2 * nu) + 9; }
    // send what is ready (unit u: an owner chip, multi-hop; core i of my link: its partial of that owner's rows);
    // blocking: until everything is sent
    template <typename C, typename H>
    void poll(C& conn, H hdr, bool blocking) {
        while (u < nu) {
            if (i >= nc) { ++u; i = 0; continue; }
            const uint32_t w0 = get_arg_val<uint32_t>(r0 + 1 + 2 * u), mesh = get_arg_val<uint32_t>(r0 + 2 + 2 * u);
            volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16 * (u * nc + i));
            invalidate_l1_cache();
            if (*wd < need) {
                if (!blocking) return;
                noc_semaphore_wait_min(wd, need);
            }
            const uint32_t cw = get_arg_val<uint32_t>(r0 + 2 + 2 * nu + i), x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            const uint32_t src0 = buf + ((u * ns + i % ns) * et) * 4096;
            const uint32_t dst0 = zone + (w0 >> 16) * et * 4096;
            for (uint32_t t0 = 0; t0 < et; t0 += rpl) {
                const uint32_t n = et - t0 < rpl ? et - t0 : rpl;
                const uint64_t dst = get_noc_addr(x, y, dst0 + t0 * 4096);
                noc_async_writes_flushed();  // (the header's previous packet has left)
                (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(w0 & 0xFFFF), static_cast<uint16_t>(mesh));
                if (t0 + rpl >= et) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(x, y, actr + 16 * (w0 >> 16)), 1, true}, n * 4096);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * 4096);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src0 + t0 * 4096, n * 4096, reinterpret_cast<uint32_t>(hdr));
            }
            if (i + ns < nc) {  // (the slot's next core may write it once the sub-block left)
                noc_async_writes_flushed();
                const uint32_t cn = get_arg_val<uint32_t>(r0 + 2 + 2 * nu + i + ns);
                noc_semaphore_inc(get_noc_addr(cn & 0xFF, (cn >> 8) & 0xFF, sfree + 16 * u), 1);
            }
            ++i;
        }
    }
    void rearm() {
        for (uint32_t c = 0; c < nu * nc; ++c) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16 * c) = 0;
    }
};
"""


class RouterStream:
    """The router chain on its own cores (module docstring). cores: the RB logical router cores (the mover column's
    free cells), xg: the gather (its ports carry the logits)."""

    @staticmethod
    def layout_of(G, ET, K, kbr, ETF=None):
        """The router cores' arena regions (offsets, bytes) at K block kbr: G chips, ET expert tiles a core, top-K;
        ETF every expert tile (split: the bias is the whole row tile's; ET a half's) - then a PART page too."""
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        off, o = {}, 0
        ETF = ETF or ET
        for k, v in [
            ("xr", 2 * G * kbr * 2048),
            ("wr", 2 * kbr * ET * 2048),
            ("hold", G * ET * 4096),
            ("sc", ET * 4096),
            ("idx", ET * 2048),
            ("best", 4096),
            ("bias", ETF * 4096),
            # (above the planner's overlay - everything below the bias, dead once the rows are made: another chip's
            # partials for the next call may land while this chip's planner of this call still runs - launch skew)
            ("arr", (G - 1) * ET * 4096),
            ("finc", 16 * 8),
            ("psync", 16 * 2),
            ("arrb", 16 * max(1, G - 1)),  # (the arrivals' slots in landing order)
            ("rdy", 16 * 2),
            ("out", 32 * 4 * K),
            ("words", 160),  # (word 128: split, the partner's best K landed)
            ("route", 32 * 4 * 2),
            ("gate", 16),  # (the W-stream gate's source word: 1, 16-byte aligned like its destination)
            ("msync", 16),  # (the compute's PACK -> MATH: the BEST page is in L1 - their shares of the rows)
        ] + (
            [("part", 2 * 4096)] if ETF != ET else []
        ):  # (split: the partner's best K, then its picks' scores)
            off[k], o = o, al(o + v)
        return off, o

    @staticmethod
    def min_bytes(G, E, K, split=False):
        """The least arena the router cores need (K block 1)."""
        ET = E // 32
        return RouterStream.layout_of(G, ET // 2 if split else ET, K, 1, ETF=ET)[1]

    def __init__(
        self,
        mesh_device,
        xg,
        *,
        rows,
        H,
        E,
        K,
        route_scale,
        cores,
        prio_first,
        arena,
        words,
        port_words=16,
        fidelity=ttnn.MathFidelity.HiFi3,
        on_engine=False,
        ordered=True,
        fold=1,
        split=False,
        score="sigmoid",
    ):
        """arena: (tensor, base, bytes) - the engine's arena, also carved on these cores and the ports (lockstep L1);
        words: the port words' address (a zeroed L1 tensor on the ports, 16 B a router core). ordered: the fin sums the
        sources' partials in a fixed order, the predicted landing order (_arrival_order: bit-identical runs), else in
        landing order (A/B only: the logits' last bits, a near tie's pick and the routing weights vary between runs).
        score: the scores of the logits, "sigmoid" or "sqrtsoftplus" (DeepSeek-V4: sqrt(softplus), fp32 softplus)."""
        assert score in ("sigmoid", "sqrtsoftplus"), f"router score {score}"
        self.score = score
        self.mesh, self.xg = mesh_device, xg
        self.ordered = ordered
        self.fold = fold
        G = self.G = xg.G
        CR = rows // 32
        assert CR % G == 0
        self.RB, self.KT, self.ET, self.K = CR // G, H // G // 32, E // 32, K
        # (split: a row tile's experts over two router cores - core h RB + j its half h's ETL tiles: the fin and the
        # top-K over half the tiles each, half 1's best K merged on half 0 - one more pair round)
        self.S = 2 if split else 1
        assert not split or (
            on_engine and self.ET % 2 == 0
        ), "split: the logits on the engine, an even expert tile count"
        self.ETL = self.ET // self.S
        assert len(cores) >= self.S * self.RB and E % 32 == 0 and K % 2 == 0 and 4 * K <= 64
        self.cores = cores[: self.S * self.RB]
        self.on_engine = on_engine  # (the logits computed by the engine cores, se_engine3's SE_RTR: no matmul here)
        self.scale_bits = struct.unpack("<I", struct.pack("<f", route_scale))[0]
        self.prio_first = prio_first
        self.fidelity = fidelity  # (the port sends the logits ahead of the x gather: GLM-type)
        self.nv = xg.nv
        ET, RB, ETL = self.ET, self.RB, self.ETL
        self.arena, self.base, abytes = arena
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731

        def layout(kbr):
            return RouterStream.layout_of(G, ETL, K, kbr, ETF=ET)

        # the K block: the largest <= 4 tiles dividing the chip's K whose buffers fit (8: no faster, measured - the
        # matmul is bound by its unpack / math, not by the blocks' L1-accumulated packs)
        kbrs = [d for d in range(1, 5) if self.KT % d == 0]
        fit = [d for d in kbrs if layout(d)[1] <= abytes]
        self.KBR = max(fit) if fit else 1
        self.off, self.bytes = off, o = layout(self.KBR)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.crs = crs
        assert o <= abytes, f"router cores need {o} B of the arena's {abytes}"
        # the ports' buffers (in the arena's range there): per port the sub-blocks of its units x cores
        nlink = -(-len(self.cores) // xg.nv)
        # (sub-block slots a unit: every core of the link's, or - on the router cores, too much L1 (K3 E 896 at 1024
        # rows: 3 x 4 x 28 fp32 tiles = 1.34 MB) - fewer, core i in slot i % NS behind the port's word that core i - NS's
        # left; on the engine its cores write the sub-blocks themselves, all of them)
        self.NS = nlink
        while not on_engine and self.NS > 1 and (G - 1) * self.NS * ETL * 4096 > abytes // 2:
            self.NS -= 1
        self.pbuf_sub = (G - 1) * self.NS * ETL * 4096
        assert self.pbuf_sub <= abytes, f"the ports' logits sub-blocks: {self.pbuf_sub} B > the arena's {abytes}"
        self.sfree_words = ttnn.from_torch(  # (the slots' "free" words on the router cores, 16 B a (direction, unit))
            torch.zeros(len(self.cores), 4 * 8, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                persist_l1.buffer_type(),
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores]),
                    [1, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.pbase, self.words = self.base, words
        self.port_words = port_words  # (the words a port has: its units x its link's router cores)
        # (arrivals / the engine's own-block tiles / the engine cores' "partials written", router core 0: the W gate)
        self.sems = [persist_l1.semaphore(mesh_device, crs, 0) for _ in range(3)]
        self.arr = [persist_l1.address(s) for s in self.sems]
        # (the arrivals: a word a source slot, 16 B apart, zero between calls - each re-armed by the router core that
        # saw it; arr[0] is their base)
        self.slot_words = ttnn.from_torch(
            torch.zeros(len(self.cores), 4 * max(1, G - 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                persist_l1.buffer_type(),
                ttnn.ShardSpec(crs, [1, 4 * max(1, G - 1)], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.arr[0] = int(self.slot_words.buffer_address())
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
        """This link's router cores (core indices: h RB + j, split) in the walk order."""
        return [j for j in range(len(self.cores)) if j % self.nv == l]

    def _pos(self, coord):
        return self.xg.chips[coord]["rings"][0]["p"]

    def _dir(self, p, b):
        """The direction chip p sends owner b's partial (shortest; ring ties the fabric's way - X+ on the
        LoudBox square and the Galaxy: XGather3.ring_dir)."""
        if not self.xg.ring:
            return "fwd" if b > p else "bwd"
        return self.xg.ring_dir(p, b)

    def _hops(self, p, b):
        G = self.G
        return min((b - p) % G, (p - b) % G) if self.xg.ring else abs(b - p)

    def _arrival_order(self, p):
        """Owner p's source slots in the fin's fixed sum order: the predicted landing order - a source sends its
        partials farthest first (_dests), so by (its send position, hops), then source. Slot of source q: q if q < p
        else q - 1 (port_block)."""
        srcs = [q for q in range(self.G) if q != p]
        key = lambda q: (self._dests(q, self._dir(q, p)).index(p), self._hops(q, p), q)  # noqa: E731
        return [q if q < p else q - 1 for q in sorted(srcs, key=key)]

    def _dests(self, p, d):
        """Owner blocks chip p sends in direction d, the farthest first - the router cores' hand-off order too (the
        port sends unit u once a core's word counts u + 1 hand-offs: both orders must agree)."""
        bs = [b for b in range(self.G) if b != p and self._dir(p, b) == d]
        return sorted(bs, key=lambda b: -self._hops(p, b))

    def port_block(self, coord, d, v):
        """The port sender's logits args (LSTREAM): its destinations (chip | slot there << 16, mesh), its link's router
        cores (NoC), its buffer / words, the owners' zone, their arrival counter, ET, tiles a packet. The partials go
        straight to their owners (multi-hop unicast; the owner sums them): no relay chain."""
        p = self._pos(coord)
        if self.xg.chips[coord]["rings"][0][("next" if d == "fwd" else "prev")] is None:
            return [0, 0] + [0] * 9
        bs = self._dests(p, d)
        js = self._port_layout(v)
        virt = self.mesh.worker_core_from_logical_core
        out = [len(bs)]
        for b in bs:
            n = self.xg._node_of_pos(coord, b)
            out += [int(n.chip_id) | (p if p < b else p - 1) << 16, int(n.mesh_id)]
        out += [len(js)] + [virt(self.cores[j]).x | virt(self.cores[j]).y << 8 for j in js]
        rpl = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // 4096)
        assert len(bs) * len(js) <= self.port_words, f"the port's sub-block words: {self.port_words} a port"
        need = self.ETL if self.on_engine else 1  # (engine tiles straight in, or one hand-off of the sub-block)
        sf = int(self.sfree_words.buffer_address()) + 64 * ("fwd", "bwd").index(d)  # (a direction's units: 4 words)
        return out + [
            self.pbase,
            self.words,
            self.base + self.off["arr"],
            self.arr[0],
            self.ETL,
            rpl,
            need,
            min(self.NS, len(js)),
            sf,
        ]

    def _ra_src(self, coord, ct0):
        if self.ra is None:
            return ""
        from models.demos.moe_block.tt.routing_stream import _plan_fn

        ra = self.ra
        self._ra_pargs = ra.planner_args(coord)
        return (
            '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n#define RA_PLAN\n'
            + ra.defines(coord)
            + _plan_fn(ra.kernel_subs(coord, ct0))
        )

    def _ra_ct(self):
        return list(ttnn.TensorAccessorArgs(self.ra.router_out).get_compile_time_args()) + list(
            ttnn.TensorAccessorArgs(self.ra.slots).get_compile_time_args()
        )

    def _ra_args(self, coord, j):
        virt = self.mesh.worker_core_from_logical_core
        if not hasattr(self, "_ra_pargs") or self._ra_coord != coord:
            self._ra_pargs, self._ra_coord = self.ra.planner_args(coord), coord
        pxy = [w for c in self.cores for v in [virt(c)] for w in (v.x, v.y)]
        if j >= self.ra.M:  # (fewer planner instances than router cores - the accumulators bound them: none here)
            return [self.ra.addr["rows"], self.ra.addr["rows2"], 0] + pxy
        return [self.ra.addr["rows"], self.ra.addr["rows2"], self.ra.pbases[j]] + pxy + self._ra_pargs(j)

    def _ra_args2(self, coord, j):
        """The second planner instance (q = j + router cores) on this router core's NCRISC."""
        if not hasattr(self, "_ra_pargs") or self._ra_coord != coord:
            self._ra_pargs, self._ra_coord = self.ra.planner_args(coord), coord
        q = j + len(self.cores)
        if q >= self.ra.M:
            return [self.ra.addr["rows2"], 0]
        return [self.ra.addr["rows2"], self.ra.pbases[q]] + self._ra_pargs(q)

    def engine_args(self, coord):
        """What the engine needs to put chip coord's partial logits in place (se_engine3 pieces(rtr=)): per row tile R
        (owner block R // RB, router core j = R % RB) its destination - my own block's into router core j's HOLD
        (its count: the engine counter), another owner's straight into the port sub-block the router core would have
        handed it to (its word: ET tiles) - as (NoC x | y << 16, base, count address); and the W gate's counter
        (router core 0: NoC, address)."""
        virt = self.mesh.worker_core_from_logical_core
        p, G, RB, ETL = self._pos(coord), self.G, self.RB, self.ETL
        ports = {(d, l): virt(c) for (_, d, l), c in self.xg.chips[coord]["ports"].items()}
        dests = []
        for R in range(G * RB):  # (split: a destination a half - the expert tiles h ETL .. - one after the other)
            bb, j = R // RB, R % RB
            dr = ()
            for h in range(self.S):
                cj = h * RB + j
                if bb == p:
                    c = virt(self.cores[cj])
                    dr += (c.x | c.y << 16, self.base + self.off["hold"] + bb * ETL * 4096, self.arr[1])
                else:
                    l = cj % self.nv
                    js = self._port_layout(l)
                    d = self._dir(p, bb)
                    k = self._dests(p, d).index(bb) * len(js) + js.index(cj)
                    pc = ports[(d, l)]
                    dr += (pc.x | pc.y << 16, self.pbase + k * ETL * 4096, self.words + 16 * k)
            dests.append(dr)
        c0 = virt(self.cores[0])
        return dict(dests=dests, gate_ctr=(c0.x | c0.y << 16, self.arr[2]), split=ETL if self.S > 1 else 0)

    def attach_mcast(self, prog):
        """The W_r column family (rotating sender, handshake) on the router NCRISC: after prog.semaphores is set."""
        if self.on_engine:
            return
        lo, hi = self.cores[0], self.cores[-1]
        col = lo.x == hi.x and hi.y - lo.y + 1 == len(self.cores)
        row = lo.y == hi.y and hi.x - lo.x + 1 == len(self.cores)
        assert col or row, "the router cores must be one column or one row rectangle"
        from models.demos.moe_block.tt.se_engine3 import next_semaphore_id

        rect = ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi)])
        mc = ttnn.Mcast(
            self.mesh,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
            rect,
            len(self.cores),  # (one group: the column or the row)
            ttnn.McastRotatingSenderConfig(),
            ttnn.McastCoreOrder.ColumnMajor if col else ttnn.McastCoreOrder.RowMajor,
        )
        mc.attach(prog, "rw", [self.ncrisc], next_semaphore_id(prog))

    ra = None  # (the routing exchange + planner riding the router BRISC: routing_stream.RoutingStream)

    gate = None  # (the engine's W-stream gate: se_engine3's gate_target - fused_head sets it with RouterSpec.w_gate)
    prep = None  # (dispatch prep's route words: dict(EPC, torus) - fused_head sets it with a RoutingSpec's prep)

    def pieces(self, coord, x, w, b, ix, out, route=None):
        """(CBs, kernels) of the router cores on chip coord. x: the own shard [rows, H / TP] (DRAM); route: the route
        words' tensor [1, 1, 1, S] uint32 (dispatch prep, with self.prep)."""
        G, ET, RB, K, ETL = self.G, self.ETL, self.RB, self.K, self.ETL
        p = self._pos(coord)
        virt = self.mesh.worker_core_from_logical_core
        ports = {(d, l): virt(c) for (_, d, l), c in self.xg.chips[coord]["ports"].items()}
        b0, off = self.base, self.off
        cbs = []

        def cb(i, key, size, dt, page):
            dd = ttnn.cb_descriptor_from_sharded_tensor(
                i, self.arena, address_offset=off[key] + getattr(self, "arena_delta", 0), total_size=size, core_ranges=self.crs
            )
            dd.format_descriptors = [ttnn.CBFormatDescriptor(i, dt, page)]
            cbs.append(dd)

        cb(XR, "xr", 2 * G * self.KBR * 2048, ttnn.bfloat16, 2048)
        cb(WR, "wr", 2 * self.KBR * ET * 2048, ttnn.bfloat16, 2048)
        cb(HOLD, "hold", G * ET * 4096, ttnn.float32, 4096)
        cb(ARRF, "arr", (G - 1) * ET * 4096, ttnn.float32, 4096)
        cb(BIAS, "bias", self.ET * 4096, ttnn.float32, 4096)
        if self.S > 1:
            cb(PART, "part", 4096, ttnn.float32, 4096)
        cb(SC, "sc", ET * 4096, ttnn.float32, 4096)
        cb(IDX, "idx", ET * 2048, ttnn.uint16, 2048)
        cb(BEST, "best", 4096, ttnn.float32, 4096)
        cb(FINC, "finc", 16 * 8, ttnn.bfloat16, 16)
        cb(PSYNC, "psync", 16 * 2, ttnn.bfloat16, 16)
        cb(ARRB, "arrb", 16 * max(1, G - 1), ttnn.uint32, 16)
        if self.on_engine:
            cb(12, "rdy", 16 * 2, ttnn.bfloat16, 16)
        n_rt, c_rt, b_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        tail = [
            b0 + off["words"],
            b0 + off["best"],
            b0 + off["out"],
            self.scale_bits,
        ]
        # (my partials of the other owners - the router cores hand them to the ports; on the engine its cores write
        # them straight into the ports' sub-blocks: none here)
        hand = [] if self.on_engine else [b for d in ("fwd", "bwd") for b in self._dests(p, d)]
        sv = (
            b0 + off["hold"] + p * ET * 4096
        )  # (my owner block's HOLD: the fin packs the scores there - the rows read them)
        hand.sort(key=lambda b: -self._hops(p, b))  # (the farthest first: their way is longest)
        for j, core in enumerate(self.cores):
            # entries = blocks (HOLD slot b: token tile b of my rows); the hand-offs, then my own block's fin
            n_rt[core.x][core.y] = (
                [int(x.buffer_address()), int(w.buffer_address()), int(b.buffer_address()), int(ix.buffer_address())]
                + [bb * RB + j % RB for bb in range(G)]
                + [G - 1, self.arr[0]]
                + self._arrival_order(p)
                + tail
                + [sv]
                + (self._ra_args2(coord, j) if self.ra is not None else [])
            )
            ops = [1 | bb << 8 for bb in hand] + [2 | p << 8]
            c_rt[core.x][core.y] = (
                [len(ops)]
                + ops
                + [G - 1 if bb == p else 0 for bb in range(G)]
                + tail
                + [sv, b0 + off["route"], 32 * (j % RB)]
            )
            hs = []
            l = j % self.nv
            js = self._port_layout(l)
            i = js.index(j)
            for bb in hand:
                d = self._dir(p, bb)
                u = self._dests(p, d).index(bb)
                pc = ports[(d, l)]
                ns = min(self.NS, len(js))  # (this link's slots a unit: the port's walk, LStream)
                sw = int(self.sfree_words.buffer_address()) + 64 * ("fwd", "bwd").index(d) + 16 * u if i >= ns else 0
                hs += [bb, pc.x, pc.y, self.pbase + (u * ns + i % ns) * ET * 4096]
                hs += [self.words + 16 * (u * len(js) + i), sw]
            jr, h = j % RB, j // RB  # (split: row tile jr's half h)
            if h:  # (half 1: no rows - its partner's NoC in the out address's place)
                pv = virt(self.cores[jr])
                o_w = [pv.x | pv.y << 16, jr]
            else:
                o_w = [int(out.buffer_address()), jr]
            b_rt[core.x][core.y] = (
                [len(hs) // 6]
                + hs
                + o_w
                + [self.scale_bits, b0 + off["out"], b0 + off["words"]]
                + [
                    b0 + off["best"],
                    b0 + off["route"],
                    int(route.buffer_address()) if route is not None else 0,
                    32 * jr,
                    sv,
                ]
                + (self._ra_args(coord, j) if self.ra is not None else [])
            )
        for j, core in enumerate(self.cores):  # (a kernel's runtime args a core: tt-metal's 341 - more are dropped)
            for nm, rt in (("NCRISC", n_rt), ("compute", c_rt), ("BRISC", b_rt)):
                n = len(list(rt[core.x][core.y]))
                assert n <= 341, f"router core {j}: {n} {nm} runtime args > 341"
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        ct = [self.KT, ET, G, self.KBR, K]
        comp = ttnn.ComputeConfigDescriptor()
        # HiFi3: the same picks as HiFi4 (GLM 99.45%, K3 99.30% of the host's fp32 routing; HiFi2 99.22 / 99.18), K3's
        # router 25 us shorter
        comp.math_fidelity = self.fidelity
        comp.fp32_dest_acc_en = True
        comp.dst_full_sync_en = False  # (half-sync: a block's packs under the next block's math)
        comp.math_approx_mode = False
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[SC] = modes[BEST] = modes[
            PART
        ] = ttnn.UnpackToDestMode.UnpackToDestFp32  # (the scores: fp32 into DEST, unrounded)
        modes[HOLD] = modes[ARRF] = modes[BIAS] = ttnn.UnpackToDestMode.UnpackToDestFp32  # (the fin's partials, bias)
        comp.unpack_to_dest_mode = modes
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        eng = f"#define RT_ENG {self.arr[1]}u\n" if self.on_engine else ""
        eng += f"#define RT_MSYNC {self.base + self.off['msync']}u\n#define RT_FOLD {self.fold}u\n"
        eng += "#define RT_ARR_ORDERED\n" if self.ordered else ""
        if self.on_engine and self.gate is not None:  # (the engine's W-stream gate: se_engine3 gate_target)
            g = self.gate
            eng += f"#define RT_GATE {g['addr']}u\n#define RT_GATE_N {g['n']}u\n#define RT_GATE_SRC {self.base + self.off['gate']}u\n"
            eng += f"#define RT_GATE_CTR {self.arr[2]}u\n"
            eng += "".join(f"#define RT_GATE_{k} {v}u\n" for k, v in zip(("X0", "Y0", "X1", "Y1"), g["rect"]))
        rt_c, rt_b, rt_ct = "", "", []
        if route is not None:
            from ttnn.bringup.combine_rf.sparse_all_gather_2d import route_words

            from models.demos.moe_block.tt.prep_stream import route_src

            R_, C_ = tuple(self.mesh.shape)
            rt_c = "#define RT_ROUTE\n" + route_src(
                (R_, C_),
                coord[0] * C_ + coord[1],
                E=self.ET * 32 // self.fold,
                EPC=self.prep["EPC"],
                K=K,
                torus=self.prep["torus"],
            )
            rt_ct = acc(route)
            n0 = len(ct + acc(out) + (self._ra_ct() if self.ra is not None else []))
            rw = route_words(R_ * C_)
            rt_b = f"#define RT_ROUTE\n#define RT_CT0 {n0}\n#define RT_PAGE {int(route.buffer_aligned_page_size())}\n"
            rt_b += f"#define RT_RW {rw}\n"
        kernels = []

        def sub(rt, cs):  # (a kernel group's runtime args: its cores' only)
            if self.S == 1:
                return rt
            o = ttnn.RuntimeArgs()
            for c in cs:
                o[c.x][c.y] = list(rt[c.x][c.y])
            return o

        for h in range(self.S):  # (split: a kernel group a half - its defines; the route words only half 0's)
            hc = self.cores[h * RB : (h + 1) * RB]
            hcrs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in hc]) if self.S > 1 else self.crs
            sd = ""
            if self.S > 1:
                sd = f"#define RT_SPLIT\n#define RT_HALF {h}\n#define RT_HB {h * ETL}u\n#define RT_ETF {self.ET}u\n"
                sd += f"#define RT_PWORD {b0 + off['words'] + 128}u\n#define RT_PART {b0 + off['part']}u\n"
                sd += f"#define RT_PART_CB {PART}\n"
            hrt_c, hrt_b, hrt_ct = (rt_c, rt_b, rt_ct) if h == 0 else ("", "", [])
            nk = ttnn.KernelDescriptor(
                kernel_source=eng
                + sd
                + self._ra_src(coord, len(ct + acc(x) + acc(w) + acc(b) + acc(ix)))
                + _NCRISC.replace("#include <cstdint>\n", "#include <cstdint>\n" + _ROWS, 1),
                source_type=st,
                core_ranges=hcrs,
                compile_time_args=ct
                + acc(x)
                + acc(w)
                + acc(b)
                + acc(ix)
                + (self._ra_ct() if self.ra is not None else []),
                runtime_args=sub(n_rt, hc),
                config=dm("RISCV_1", ttnn.NOC.NOC_0),
            )
            if h == 0:
                self.ncrisc = nk
            kernels += [
                nk,
                ttnn.KernelDescriptor(
                    kernel_source=eng
                    + sd
                    + ("#define RT_SQRTSOFTPLUS\n#define INP_FLOAT32 1\n" if self.score == "sqrtsoftplus" else "")
                    + _COMPUTE.replace("#include <cstdint>\n", "#include <cstdint>\n" + _ROWS + hrt_c, 1),
                    source_type=st,
                    core_ranges=hcrs,
                    compile_time_args=ct,
                    runtime_args=sub(c_rt, hc),
                    config=comp,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=f"#define RT_FOLD {self.fold}u\n"
                    + sd
                    + hrt_b
                    + self._ra_src(coord, len(ct + acc(out)))
                    + _BRISC,
                    source_type=st,
                    core_ranges=hcrs,
                    compile_time_args=ct + acc(out) + (self._ra_ct() if self.ra is not None else []) + hrt_ct,
                    runtime_args=sub(b_rt, hc),
                    config=dm("RISCV_0", ttnn.NOC.NOC_1),
                ),
            ]
        return cbs, kernels
