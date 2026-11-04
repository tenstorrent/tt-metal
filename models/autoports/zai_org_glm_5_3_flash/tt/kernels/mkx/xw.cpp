// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel routed experts — weight streamer + routing (BRISC, NoC0) of a DS bank reader (mkx_common.hpp):
//   1. routing (fused_experts' compute_expert_ids, per reader): the ids tile + the score tiles -> the union of the
//      rows' selections in ascending id order and the per-(expert, row) weights scale * s / (sum + eps), bf16 RNE
//      -> the list page (CB_LIST: pushed for the compute; the NCRISC reads it in place after MKX_SEM_LIST);
//   2. for every expert of the union, in order: this reader's gate_up shard blocks, then its down shard blocks (KB
//      rows x 2 tiles each, contiguous in the bank), read by address (expert e's tensor base from the address table)
//      into the ring — the compute's consumption order.
// Compile-time args: E, TOPK, KT_GU, KT_DN, KB, TB, GU_COLS, DN_COLS, SCALING_BITS, EPS_BITS, E_TILES, RING_SLOTS,
// E_LO, E_LOC, then the
//   TensorAccessorArgs of ids, scores, table, a gate_up and a down expert tensor (all experts share the spec).
// Runtime args: ids, scores, table addresses, T (rows), NSH (owned shards), the NSH shard indices (<= 3, padded),
//   VC, then (MKX_WAIT_SEM) the router-tail producer count; MKX_XTP (at MKX_XTP_RTW): this reader's bank, its K range
//   [K0, K1) (tiles) of the bank's gate_up shard XTP_C * 8 + bank. MKX_XTP compile-time arg (after the accessors):
//   XTP_C, the slice (Stage 14 patches it per chip like E_LO); the down reads rows [XTP_C * KT_DN, + KT_DN).
//   MKX_XSL compile-time arg (after the accessors): XSL_C, the slice (patched per chip); KT_DN = the slice width W.
//   MKX_LIVE (Stage 14b, coordinator 2026-10-22 10:51Z): the TensorAccessorArgs of the stack's position page (int32
//   [1, 32], row b < 0 = an idle decode row) after the XTP / XSL slot, its address at runtime arg MKX_LIVE_RT: an
//   idle row selects nothing (no union entry, no routing weight, no expert streamed for it), so a live row's bits
//   and the step's expert stream do not depend on idle rows' (stale) content.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/mkx_common.hpp"

using namespace mkx;

namespace {

FORCE_INLINE float bf16_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

FORCE_INLINE uint16_t f32_to_bf16_rne(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return static_cast<uint16_t>(((bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u) >> 16);
}

}  // namespace

void kernel_main() {
#if defined(MK_LAYERS) && MK_LAYERS > 1
    // Stage 14b stack program (kernels/mk/stack.hpp): this phase runs once per layer; its CBs start at their bases
    // every time (all empty at a layer boundary; the ring's slot addressing restarts at 0 below)
    mk::cb_rewind_range(MKX_CB0, 11);
#endif
    constexpr uint32_t E = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t TOPK = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t KT_GU = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t KT_DN = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t TB = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t GU_COLS = get_compile_time_arg_val(PH_CT0 + 6);  // gate_up tile columns (2I / 32)
    constexpr uint32_t DN_COLS = get_compile_time_arg_val(PH_CT0 + 7);  // down tile columns (H / 32)
    constexpr uint32_t SCALING_BITS = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr uint32_t E_TILES = get_compile_time_arg_val(PH_CT0 + 10);
    constexpr uint32_t RING_SLOTS = get_compile_time_arg_val(PH_CT0 + 11);
    // expert parallelism: this chip holds experts [E_LO, E_LO + E_LOC); only those enter the union (the routing weights
    // still normalise over the row's whole selection), the address table holds the local experts only. MKX_LOC (a
    // load-aware placement): the chip's E_LOC experts are any ids; the table row is [gate_up addresses (E_LOC) | down
    // addresses (E_LOC) | loc (E): the expert's local slot, E_LOC if not on this chip] and E_LO is unused
    constexpr uint32_t E_LO = get_compile_time_arg_val(PH_CT0 + 12);
    constexpr uint32_t E_LOC = get_compile_time_arg_val(PH_CT0 + 13);
    constexpr auto ids_args = TensorAccessorArgs<PH_CT0 + 14>();
    constexpr auto sc_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto tab_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto gu_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr uint32_t BLOCK_TILES = KB * BLOCK_TILES_COLS;
    constexpr uint32_t BLOCK_BYTES = BLOCK_TILES * TB;
    static_assert(KT_GU % KB == 0 && KT_DN % KB == 0, "K blocks");
    static_assert(E <= MAX_EXPERTS && TOPK <= 16, "routing sizes");

    const auto ids_t = TensorAccessor(ids_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto sc_t = TensorAccessor(sc_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto tab_t = TensorAccessor(tab_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const uint32_t T = get_arg_val<uint32_t>(ph_rt0 + 3);
    const uint32_t nsh = get_arg_val<uint32_t>(ph_rt0 + 4);
    uint32_t shard[3];
    for (uint32_t s = 0; s < 3; ++s) {
        shard[s] = get_arg_val<uint32_t>(ph_rt0 + 5 + s);
    }
    MK_MARK();

    // ---- 1. routing inputs -> raw scratch: ids tile | the weights tile (MKX_RW) or E_TILES score tiles | the address
    // table row. The table row and the zero fill of the list page's weight rows do not depend on the router: both are
    // issued before the hand-off wait, so the routing step after it is the ids / weights read + the union (no stores of
    // a dense [n][32] table: the fill is a NoC read of the hardware zeros)
    cb_reserve_back(CB_RAW, 1);
    const uint32_t raw = get_write_ptr(CB_RAW);
    constexpr uint32_t TILE16 = 2048;
#if defined(MKX_RW)
    // the router tail computed the rows' routing weights ([B, 8] bf16, selection order; router_tail/writer.cpp
    // RTAIL_WEIGHTS): the second input is that one tile instead of the E_TILES score tiles
    constexpr uint32_t N_SC = 1;
#else
    constexpr uint32_t N_SC = E_TILES;
#endif
    const uint32_t tab_l1 = raw + (1 + N_SC) * TILE16;
    noc_async_read_page(0, tab_t, tab_l1);
#if defined(MKX_LIVE)
#if defined(MKX_XTP) || defined(MKX_XSL)
    constexpr auto live_args = TensorAccessorArgs<dn_args.next_compile_time_args_offset() + 1>();
#else
    constexpr auto live_args = TensorAccessorArgs<dn_args.next_compile_time_args_offset()>();
#endif
    const uint32_t live_l1 = raw + LIVE_OFF;  // retired by the barrier after the router's ids / weights
    noc_async_read_page(0, TensorAccessor(live_args, get_arg_val<uint32_t>(ph_rt0 + MKX_LIVE_RT)), live_l1);
#endif
    cb_reserve_back(CB_LIST, 1);
    const uint32_t list_l1 = get_write_ptr(CB_LIST);
    {
        // weight rows [0, min(E_LOC, T * TOPK)) x 32 bf16 (xa.cpp reads all 32 rows of every union rank). The rows
        // start 8 B past a 16-B boundary (list + LIST_IDS + MAX_EXPERTS words): the fill starts at the boundary, i.e.
        // it also zeroes the last two id slots, which the union writes only after the read barrier below
        constexpr uint32_t WL_OFF = (LIST_IDS + MAX_EXPERTS) * 4;
        constexpr uint32_t Z0 = WL_OFF & ~15u;
        const uint32_t rows = T * TOPK < E_LOC ? T * TOPK : E_LOC;
        const uint32_t bytes = (WL_OFF - Z0 + rows * 64 + MEM_ZEROS_SIZE - 1) & ~(MEM_ZEROS_SIZE - 1);
        static_assert(
            Z0 + ((WL_OFF - Z0 + E_LOC * 64 + MEM_ZEROS_SIZE - 1) & ~(MEM_ZEROS_SIZE - 1)) <= LIST_BYTES,
            "list page: weight rows");
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, list_l1 + Z0 + off, MEM_ZEROS_SIZE);
        }
    }
#if defined(MKX_WAIT_SEM)
    mk::wait_handoff(MKX_WAIT_SEM, get_arg_val<uint32_t>(ph_rt0 + 9));  // the router tail's rows landed
#endif
    MK_MARK();
    noc_async_read_page(0, ids_t, raw);
    for (uint32_t i = 0; i < N_SC; ++i) {
        noc_async_read_page(i, sc_t, raw + (1 + i) * TILE16);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    MK_MARK();
    // this RISC's own data from here to the list push: plain (cacheable, schedulable) loads; the list page is written
    // through volatile pointers and published by the push + the semaphore below
    const uint16_t* ids16 = reinterpret_cast<const uint16_t*>(raw);
    const uint16_t* sc16 = reinterpret_cast<const uint16_t*>(raw + TILE16);
    const uint32_t* tab = reinterpret_cast<const uint32_t*>(tab_l1);

    // union of the rows' selections (a repeated id within a row counts once), ascending; work arrays in the scratch
    // CB behind the table (the data-movement stack is small). stamp[e] = the last row (b + 1) that selected e: the
    // duplicate check is one load per selection and the union is stamp != 0 (scanned a word at a time)
#if defined(MKX_LOC)
    const uint32_t* loc = tab + 2 * E_LOC;
    auto local = [&](uint32_t e) -> uint32_t { return loc[e]; };
#if defined(MKX_REP)
    // EP replicas: the table row also holds home[e] = primary chip | second chip << 8 (0xFF: none), then this chip
    const uint32_t* home = loc + E;
    const uint32_t me = home[E];
    const uint32_t work = tab_l1 + (((2 * E_LOC + 2 * E + 1) * 4 + 63) & ~63u);
#else
    const uint32_t work = tab_l1 + (((2 * E_LOC + E) * 4 + 63) & ~63u);
#endif
#else
    auto local = [&](uint32_t e) -> uint32_t { return e - E_LO; };
    const uint32_t work = tab_l1 + ((2 * E * 4 + 63) & ~63u);
#endif
    static_assert(E % 4 == 0, "word-wise stamp scan");
#if defined(MKX_LIVE)
    // the work arrays' end (the layout above: table row, then sel / rank / stamp [/ mine]) stays below the page
#if defined(MKX_LOC) && defined(MKX_REP)
    constexpr uint32_t WORK_OFF = (1 + N_SC) * TILE16 + (((2 * E_LOC + 2 * E + 1) * 4 + 63) & ~63u), WORK_BYTES = 2 * E;
#elif defined(MKX_LOC)
    constexpr uint32_t WORK_OFF = (1 + N_SC) * TILE16 + (((2 * E_LOC + E) * 4 + 63) & ~63u), WORK_BYTES = E;
#else
    constexpr uint32_t WORK_OFF = (1 + N_SC) * TILE16 + ((2 * E * 4 + 63) & ~63u), WORK_BYTES = E;
#endif
    static_assert(
        WORK_OFF + 32 * TOPK * 4 + 2 * E + WORK_BYTES <= LIVE_OFF, "MKX_LIVE: the page overlaps the work arrays");
    const int32_t* live = reinterpret_cast<const int32_t*>(live_l1);
#endif
    uint32_t* sel = reinterpret_cast<uint32_t*>(work);                              // [32 * TOPK]
    uint16_t* rank = reinterpret_cast<uint16_t*>(work + 32 * TOPK * 4);             // [E]
    uint32_t* stamp32 = reinterpret_cast<uint32_t*>(work + 32 * TOPK * 4 + 2 * E);  // [E] bytes
    uint8_t* stamp = reinterpret_cast<uint8_t*>(stamp32);
    for (uint32_t w = 0; w < E / 4; ++w) {
        stamp32[w] = 0;
    }
    for (uint32_t b = 0; b < T; ++b) {
        const uint8_t tag = static_cast<uint8_t>(b + 1);
#if defined(MKX_LIVE)
        if (live[b] < 0) {  // an idle row: no selection (0xFFFF >= MAX_EXPERTS: xa's scatter skips it too)
            for (uint32_t j = 0; j < TOPK; ++j) {
                sel[b * TOPK + j] = 0xFFFFu;
            }
            continue;
        }
#endif
        for (uint32_t j = 0; j < TOPK; ++j) {
            uint32_t e = ids16[tile_elem(b, j)];
            if (e < E) {
                if (stamp[e] == tag) {
                    e = E;  // duplicate within the row: dropped
                } else {
                    stamp[e] = tag;
                }
            }
            sel[b * TOPK + j] = e;
        }
    }
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_l1);
    volatile tt_l1_ptr uint16_t* wl = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(list + LIST_IDS + MAX_EXPERTS);
#if defined(MKX_REP)
    // which touched experts THIS chip streams: single-copy ones where they live; each touched replicated one on the
    // lighter of its two homes (ascending id, after the single-copy ones are counted; ties -> the primary). The same
    // deterministic greedy runs on every chip over the replicated router output, so every touched expert is streamed
    // on exactly one chip (scripts/ep_replica_sim.py: B=32 busiest chip -2.5 experts at 36 replicas)
    static_assert(E <= 256 * 8, "home words");
    uint8_t* mine = stamp + E;  // [E] bytes behind the stamps
    {
        uint32_t cost[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t w = 0; w < E / 4; ++w) {
            const uint32_t m = stamp32[w];
            if (m == 0) {
                continue;
            }
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t e = 4 * w + k;
                if (((m >> (8 * k)) & 0xFFu) != 0 && ((home[e] >> 8) & 0xFFu) == 0xFFu) {
                    ++cost[home[e] & 7u];
                }
            }
        }
        for (uint32_t w = 0; w < E / 4; ++w) {
            const uint32_t m = stamp32[w];
            if (m == 0) {
                continue;
            }
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t e = 4 * w + k;
                if (((m >> (8 * k)) & 0xFFu) == 0) {
                    continue;
                }
                const uint32_t h = home[e], sec = (h >> 8) & 0xFFu;
                if (sec == 0xFFu) {
                    mine[e] = 1;  // single copy: here() decides by loc
                } else {
                    const uint32_t pri = h & 7u;
                    const uint32_t c = cost[pri] <= cost[sec & 7u] ? pri : (sec & 7u);
                    ++cost[c];
                    mine[e] = c == me ? 1 : 0;
                }
            }
        }
    }
    auto here = [&](uint32_t e) -> bool { return loc[e] < E_LOC && mine[e] != 0; };
#elif defined(MKX_LOC)
    auto here = [&](uint32_t e) -> bool { return loc[e] < E_LOC; };
#else
    auto here = [&](uint32_t e) -> bool { return e - E_LO < E_LOC; };  // unsigned: e < E_LO wraps
#endif
    uint32_t n = 0;
    for (uint32_t w = 0; w < E / 4; ++w) {
        const uint32_t m = stamp32[w];
        if (m == 0) {
            continue;
        }
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t e = 4 * w + k;
            if (((m >> (8 * k)) & 0xFFu) == 0) {
                continue;
            }
            if (here(e)) {
                rank[e] = static_cast<uint16_t>(n);
                list[LIST_IDS + n] = e;
                ++n;
            } else {
                rank[e] = 0xFFFFu;  // touched, streamed on another chip / not local (MKX_RW_XA's scatter skips it)
            }
        }
    }
#if defined(MKX_RW) && defined(MKX_RW_XA)
    // the scatter of the tail's weights runs on xa (the activation mover, idle until the first expert's m gather):
    // the list goes out right after the union; xa reads sel / rank / the weights tile from this RISC's scratch
    static_assert(LIST_XA_WORD * 4 >= (LIST_IDS + MAX_EXPERTS) * 4 + 8 + MAX_EXPERTS * 64, "list page: xa words");
    static_assert(TOPK == RW_TOPK_XA, "MKX_RW_XA: top-8");
    list[LIST_XA_WORD] = work;
    list[LIST_XA_WORD + 1] = raw + TILE16;
#elif defined(MKX_RW)
    for (uint32_t b = 0; b < T; ++b) {  // the tail's weights, scattered to the union's ranks (no arithmetic here)
        for (uint32_t j = 0; j < TOPK; ++j) {
            const uint32_t e = sel[b * TOPK + j];
            if (e < E && here(e)) {
                wl[rank[e] * 32 + b] = sc16[tile_elem(b, j)];
            }
        }
    }
#else
    constexpr float scaling = __builtin_bit_cast(float, SCALING_BITS);
    constexpr float eps = __builtin_bit_cast(float, EPS_BITS);
    for (uint32_t b = 0; b < T; ++b) {
        float s[TOPK];
        float sum = 0.0f;
        for (uint32_t j = 0; j < TOPK; ++j) {
            const uint32_t e = sel[b * TOPK + j];
            if (e >= E) {
                s[j] = 0.0f;
                continue;
            }
            s[j] = bf16_to_f32(sc16[(e >> 5) * (TILE16 / 2) + tile_elem(b, e & 31u)]);
            sum += s[j];
        }
        const float inv = scaling / (sum + eps);
        for (uint32_t j = 0; j < TOPK; ++j) {
            const uint32_t e = sel[b * TOPK + j];
            if (e < E && here(e)) {
                wl[rank[e] * 32 + b] = f32_to_bf16_rne(s[j] * inv);
            }
        }
    }
#endif
    asm volatile("" ::: "memory");  // the plain-pointer work above is done before the list is published
    list[0] = n;
    list[1] = T;
    cb_push_back(CB_LIST, 1);  // the compute reads n
    // the NCRISC reads the list in place (its own CB_LIST read pointer is at the same base)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MKX_SEM_LIST)) = 1;
    MK_MARK();

    // ---- 2. the weight stream: per expert, gate_up blocks of the owned shards, then their down blocks
    const uint32_t ring = get_local_cb_interface(CB_RING).fifo_limit - get_local_cb_interface(CB_RING).fifo_size;
    uint32_t slot = 0;
    // MKX_INFLIGHT blocks in flight (transaction ids 1 .. MKX_INFLIGHT, oldest waited for and pushed first); before a
    // block is issued, ring space for it and every block still in flight is reserved
    constexpr uint32_t TXN_BYTES = MKX_TPT * TB;
    constexpr uint32_t DEPTH = MKX_INFLIGHT;
    static_assert(BLOCK_BYTES % TXN_BYTES == 0 && TXN_BYTES <= NOC_MAX_BURST_SIZE, "transactions");
    static_assert(DEPTH >= 1 && DEPTH <= 8 && DEPTH < RING_SLOTS, "in-flight depth");
    const uint32_t vc = get_arg_val<uint32_t>(ph_rt0 + 8);
    uint32_t trid_issue = 1, trid_wait = 1, inflight = 0;
    auto retire = [&]() {
        noc_async_read_barrier_with_trid(trid_wait);
        cb_push_back(CB_RING, BLOCK_TILES);
        trid_wait = trid_wait == DEPTH ? 1 : trid_wait + 1;
        --inflight;
    };
    auto block = [&](uint64_t src) {
        cb_reserve_back(CB_RING, (inflight + 1) * BLOCK_TILES);
        const uint32_t dst = ring + slot * BLOCK_BYTES;
        const uint32_t lo = static_cast<uint32_t>(src);
        noc_async_read_one_packet_set_state<true>(src, TXN_BYTES, vc);  // this reader's bank (every owned shard)
        noc_async_read_set_trid(trid_issue);
        for (uint32_t off = 0; off < BLOCK_BYTES; off += TXN_BYTES) {
            noc_async_read_one_packet_with_state_with_trid<false, false>(0, lo + off, dst + off, trid_issue);
        }
        slot = slot + 1 == RING_SLOTS ? 0 : slot + 1;
        trid_issue = trid_issue == DEPTH ? 1 : trid_issue + 1;
        if (++inflight == DEPTH) {
            retire();
        }
    };
    // the compute's order: gate_up(0), then for pos >= 1 gate_up(pos) before down(pos - 1) (expert pos - 1's m gather
    // overlaps expert pos's gate_up), down(n - 1) last
#if defined(MKX_XTP)
    // expert TP: one gate_up shard (this bank's slice tile), this reader's K range; the down rows of the slice
    constexpr uint32_t XTP_C = get_compile_time_arg_val(dn_args.next_compile_time_args_offset());
    const uint32_t xgu = XTP_C * MKX_XTP_W + get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTW + 0);
    const uint32_t xk0 = get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTW + 1);
    const uint32_t xk1 = get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTW + 2);
    constexpr uint32_t dn_k0 = XTP_C * KT_DN;
#elif defined(MKX_XSL)
    // batched expert TP (mkx_common.hpp MKX_XSL): slice XSL_C (width W = KT_DN m tiles) of every selected expert, the
    // union taken GS = 64 / W experts at a time; group g's shard j is slice tile j % W of the group's expert j / W: its
    // gate_up shard W XSL_C + j % W sits in bank j % 8 like EP's shard j (full K), its down rows [XSL_C W, + W) of
    // output shard j; pos = the group
    constexpr uint32_t XSL_C = get_compile_time_arg_val(dn_args.next_compile_time_args_offset());
    constexpr uint32_t XW = KT_DN, GS = 64 / KT_DN;
    constexpr uint32_t dn_k0 = XSL_C * KT_DN;
    static_assert(KT_DN % 8 == 0 && 64 % KT_DN == 0 && KT_DN % KB == 0, "MKX_XSL: slice width 8 / 16 / 32");
#else
    constexpr uint32_t dn_k0 = 0;
#endif
    auto gate_up = [&](uint32_t pos) {
#if defined(MKX_XSL)
        for (uint32_t s = 0; s < nsh; ++s) {
            const uint32_t q = GS * pos + shard[s] / XW;  // the group's expert of this shard (none past the union)
            if (q < n) {
                const auto gu = TensorAccessor(gu_args, tab[local(list[LIST_IDS + q])]);
                for (uint32_t kb = 0; kb < KT_GU; kb += KB) {
                    block(gu.get_noc_addr(kb * GU_COLS + 2 * (XW * XSL_C + shard[s] % XW)));
                }
            }
        }
        return;
#endif
        const auto gu = TensorAccessor(gu_args, tab[local(list[LIST_IDS + pos])]);
#if defined(MKX_XTP)
        for (uint32_t kb = xk0; kb < xk1; kb += KB) {
            block(gu.get_noc_addr(kb * GU_COLS + 2 * xgu));
        }
#else
        for (uint32_t s = 0; s < nsh; ++s) {
            for (uint32_t kb = 0; kb < KT_GU; kb += KB) {
                block(gu.get_noc_addr(kb * GU_COLS + 2 * shard[s]));
            }
        }
#endif
    };
    auto down = [&](uint32_t pos) {
#if defined(MKX_XSL)
        const uint32_t nq = n - GS * pos < GS ? n - GS * pos : GS;  // the group's experts (K = KT_DN each)
        for (uint32_t s = 0; s < nsh; ++s) {
            for (uint32_t q = 0; q < nq; ++q) {
                const auto dn = TensorAccessor(dn_args, tab[E_LOC + local(list[LIST_IDS + GS * pos + q])]);
                for (uint32_t kb = 0; kb < KT_DN; kb += KB) {
                    block(dn.get_noc_addr((dn_k0 + kb) * DN_COLS + 2 * shard[s]));
                }
            }
        }
        return;
#endif
        const auto dn = TensorAccessor(dn_args, tab[E_LOC + local(list[LIST_IDS + pos])]);
        for (uint32_t s = 0; s < nsh; ++s) {
            for (uint32_t kb = 0; kb < KT_DN; kb += KB) {
                block(dn.get_noc_addr((dn_k0 + kb) * DN_COLS + 2 * shard[s]));
            }
        }
    };
#if defined(MKX_XSL)
    const uint32_t npos = (n + GS - 1) / GS;  // groups of GS experts
#else
    const uint32_t npos = n;
#endif
    if (npos > 0) {
        gate_up(0);
        for (uint32_t pos = 1; pos < npos; ++pos) {
            gate_up(pos);
            down(pos - 1);
        }
        down(npos - 1);
    }
    while (inflight > 0) {
        retire();
    }
    noc_async_read_set_trid(0);
    MK_MARK();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
