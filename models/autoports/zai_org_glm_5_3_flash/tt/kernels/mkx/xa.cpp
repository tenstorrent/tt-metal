// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel routed experts — activation mover (NCRISC, NoC1) of a DS bank reader (mkx_common.hpp):
//   x (all K tiles) -> CB_X once (resident for every expert's gate_up);
//   per expert pos of the union (the list page, read in place after MKX_SEM_LIST):
//     the routing weight tile (row b = w[pos, b] in every column) -> CB_RS;
//     this reader's m tiles (CB_MO, from the compute) -> pages of the shared m tensor's buffer p = pos & 1 (all-gather:
//     every reader writes its shards' tiles, signals MKX_SEM_MW<p> on every reader, waits for all of them), then the
//     whole m of the expert -> CB_M (the down's in0), then MKX_SEM_MR<p> on every reader (buffer p consumed: before
//     writing buffer p again, two experts later, a reader waits until every reader has copied it);
//   the final output tiles (CB_Y, 2 per owned shard) -> y columns 2j, 2j + 1.
// Compile-time args: KT_GU, I_TILES, NREAD (readers signalled), then the TensorAccessorArgs of x, m, y.
// Runtime args: x, m, y addresses, NSH, the NSH shard indices (<= 3, padded to 3), then NREAD reader (x, y) NoC
//   coordinates; MKX_XTP (at MKX_XTP_RTA): role (0 = the bank's leader, 1 / 2 = sender), the leader's NoC (x, y), the
//   bank (= this slice's m tile index; mkx_common.hpp CB_PART). MKX_XSL: per group of 8 experts, one routing weight
//   tile per owned shard with an expert (CB_RS, MKX_XSL_RS_PAGES pages), then the m all-gather of the group as EP.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mkx/mkx_common.hpp"

using namespace mkx;

#if defined(MKX_XROWS) && defined(MKX_XROWS_PRE)
// megakernel composite (mk/reader_moe_a.cpp): the x tiles' zero fill (a local DMA of the hardware zeros, 256 KB = 512
// reads, ~10+ us issue-bound) is issued at PROGRAM START, while this RISC waits for mhc_pre's output anyway; CB_X's L1
// is this phase's alone, so nothing else touches it before the expert phase. Not waited for here: the dense phases'
// first read barrier retires it (tens of us later), the expert phase's barrier before its row reads at the latest.
// From CB_X's base, not its current write pointer: with MK_QKV_ALIAS (mk/reader_moe_a.cpp) the fill runs once per
// layer of a stack program (MK_LAYERS > 1), before the expert phase rewinds its CBs, and CB_X's FIFO is the merged
// qkv-ring region (larger than KT_GU pages, e.g. 278,528 vs 262,144 B), so after layer 0 the write pointer no longer
// wraps to the base and the fill ran past the CB's end. CB_X is empty here (the previous layer's expert phase
// consumed it).
FORCE_INLINE void xrows_prologue() {
    constexpr uint32_t KT_GU = get_compile_time_arg_val(PH_CT0 + 0);
    mk::cb_rewind(CB_X);
    const uint32_t xl1 = get_write_ptr(CB_X), xpage = get_tile_size(CB_X);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < KT_GU * xpage; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, xl1 + off, MEM_ZEROS_SIZE);
    }
}
#endif

void kernel_main() {
    constexpr uint32_t KT_GU = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t I_TILES = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t NREAD = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr auto x_args = TensorAccessorArgs<PH_CT0 + 3>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto m = TensorAccessor(m_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(ph_rt0 + 2));
#if defined(MKX_Y2_INIT)
    constexpr auto y2_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
#endif
#if defined(MKX_XINV)
#if defined(MKX_Y2_INIT)
    constexpr auto xinv_args = TensorAccessorArgs<y2_args.next_compile_time_args_offset()>();
#else
    constexpr auto xinv_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
#endif
#endif
    const uint32_t nsh = get_arg_val<uint32_t>(ph_rt0 + 3);
#if defined(MK_LAYERS) && MK_LAYERS > 1
    // Stage 14b stack program (kernels/mk/stack.hpp): this phase runs once per layer; its CBs start at their bases
    // every time (all empty at a layer boundary; the ring's slot addressing restarts at 0 below)
#if defined(MKX_Y2_INIT)
    // + CB_Y2 (MKX_CB0 + 11, fold_y2 / TP_SHARED): a reader owning fewer shards than the CB's depth pushes a part of it
    // per layer, so without the rewind its pointers drift past fifo_limit from layer 1 on (2026-10-20 review)
    mk::cb_rewind_range(MKX_CB0, 12);
#else
    mk::cb_rewind_range(MKX_CB0, 11);
#endif
#endif
    uint32_t shard[3];
    for (uint32_t s = 0; s < 3; ++s) {
        shard[s] = get_arg_val<uint32_t>(ph_rt0 + 4 + s);
    }
    constexpr uint32_t COORDS = 7;
#if defined(MKX_XTP)
    const uint32_t xrole = get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTA + 0);
    const uint32_t xbank = get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTA + 3);
    // CB_PART sits at the same L1 address on every reader: this RISC's write pointer is at the base before any push
    const uint32_t part_base = get_write_ptr(CB_PART);
    const uint64_t leader_part = get_noc_addr(
        get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTA + 1), get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTA + 2), part_base);
    const uint64_t leader_sem = get_noc_addr(
        get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTA + 1),
        get_arg_val<uint32_t>(ph_rt0 + MKX_XTP_RTA + 2),
        get_semaphore(MKX_XTP_SEM));
    const uint32_t part_page = get_tile_size(CB_PART);
#endif
    MK_MARK();
#if defined(MKX_Y2_INIT)
    // routed + shared expert in ONE partial: this reader's shared-down output tiles have landed (the dense phase before
    // this one wrote them, write barrier done) -> tell every reader; the y2 tiles of this reader's shards (written by
    // other readers) are read below, once every reader has signalled
    mk::signal_handoff(MKX_Y2_SEM, ph_rt0 + COORDS, NREAD);
#endif

    // x, resident
    cb_reserve_back(CB_X, KT_GU);
    const uint32_t xl1 = get_write_ptr(CB_X), xpage = get_tile_size(CB_X);
#if defined(MKX_XROWS)
    // decode rows only (MKX_XROWS = T <= 16): every reader used to pull all KT_GU full tiles of x (32x the bytes at
    // T = 1, all readers on the same L1 pages: most of the B=1 'routing list' interval); now the tiles are zeroed by a
    // local DMA of the hardware zeros and only the T valid rows are read: rows 0..15 sit in faces 0 / 1 (bf16 face =
    // 16 x 16 x 2 B = 512 B, face 1 at +512), one 32-B face row per face per row (L1 source: 16-B aligned)
    {
        static_assert(MKX_XROWS >= 1 && MKX_XROWS <= 16, "MKX_XROWS: decode rows 0..15 (faces 0 / 1)");
#if !defined(MKX_XROWS_PRE)
        // standalone program: the zero fill here (in a megakernel composite it is the prologue: xrows_prologue())
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < KT_GU * xpage; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, xl1 + off, MEM_ZEROS_SIZE);
        }
#endif
        noc_async_read_barrier();  // the zeros land before the row reads into the same tiles
        for (uint32_t k = 0; k < KT_GU; ++k) {
            const uint32_t dst = xl1 + k * xpage;
            for (uint32_t r = 0; r < MKX_XROWS; ++r) {
                noc_async_read(x.get_noc_addr(k, r * 32), dst + r * 32, 32);
                noc_async_read(x.get_noc_addr(k, 512 + r * 32), dst + 512 + r * 32, 32);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_X, KT_GU);
    }
#elif defined(MKX_XPROG)
    // progressive: pushed in chunks of 8 tiles, so the first expert's gate_up starts on its first K block instead of
    // after all KT_GU tiles (xc.cpp waits cumulatively per K block)
    constexpr uint32_t XCH = 8;
    static_assert(KT_GU % XCH == 0, "x chunks");
    for (uint32_t k0 = 0; k0 < KT_GU; k0 += XCH) {
        for (uint32_t k = k0; k < k0 + XCH; ++k) {
            noc_async_read_page(k, x, xl1 + k * xpage);
        }
        noc_async_read_barrier();
        cb_push_back(CB_X, XCH);
    }
#else
    for (uint32_t k = 0; k < KT_GU; ++k) {
        noc_async_read_page(k, x, xl1 + k * xpage);
    }
    noc_async_read_barrier();
    cb_push_back(CB_X, KT_GU);
#endif
#if defined(MKX_Y2_INIT)
    mk::wait_handoff(MKX_Y2_SEM, NREAD);
    {
        const auto y2 = TensorAccessor(y2_args, get_arg_val<uint32_t>(ph_rt0 + MKX_Y2_RT));
        cb_reserve_back(CB_Y2, 2 * nsh);
        const uint32_t l1 = get_write_ptr(CB_Y2), page = get_tile_size(CB_Y2);
        for (uint32_t s = 0; s < nsh; ++s) {
            noc_async_read_page(2 * shard[s], y2, l1 + (2 * s) * page);
            noc_async_read_page(2 * shard[s] + 1, y2, l1 + (2 * s + 1) * page);
        }
        noc_async_read_barrier();
        cb_push_back(CB_Y2, 2 * nsh);
    }
#endif

    // the list (written by this core's BRISC)
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MKX_SEM_LIST));
    while (*flag == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
#if defined(MK_LAYERS) && MK_LAYERS > 1
    *flag = 0;  // stack: the next layer's list is published by a fresh 1 (its BRISC rebuilds the list only after this
                // layer's routed output was consumed: the layer's P10 waits for every reader's shared down)
#endif
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_LIST));
    const uint32_t n = list[0];
    volatile tt_l1_ptr uint16_t* wl = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(list + LIST_IDS + MAX_EXPERTS);
#if defined(MKX_XINV)
    {
        // the FFN 1/rms(h) tile: already landed (this RISC's dense router phase waited for its hand-off) -> CB_XINV,
        // held by the compute for every expert's gate | up (also with no local expert: the compute pops it at the end)
        const auto xinv = TensorAccessor(xinv_args, get_arg_val<uint32_t>(ph_rt0 + MKX_XINV_RT));
        cb_reserve_back(CB_XINV, 1);
        noc_async_read_page(0, xinv, get_write_ptr(CB_XINV));
        noc_async_read_barrier();
        cb_push_back(CB_XINV, 1);
    }
#endif
    MK_MARK();
#if defined(MKX_RW) && defined(MKX_RW_XA)
    {
        // xw.cpp left its (deduplicated) selections, the union ranks (0xFFFF: not streamed here) and the tail's
        // weights tile in its scratch; the weight rows were zeroed by xw's NoC fill (completed before the list went
        // out)
        const uint32_t rows = list[1];
        const uint32_t* sel = reinterpret_cast<const uint32_t*>(list[LIST_XA_WORD]);
        const uint16_t* rank = reinterpret_cast<const uint16_t*>(list[LIST_XA_WORD] + 32 * RW_TOPK_XA * 4);
        const uint16_t* sc16 = reinterpret_cast<const uint16_t*>(list[LIST_XA_WORD + 1]);
        for (uint32_t b = 0; b < rows; ++b) {
            for (uint32_t j = 0; j < RW_TOPK_XA; ++j) {
                const uint32_t e = sel[b * RW_TOPK_XA + j];
                if (e < MAX_EXPERTS && rank[e] != 0xFFFFu) {
                    wl[rank[e] * 32 + b] = sc16[tile_elem(b, j)];
                }
            }
        }
    }
#endif

    const uint32_t mpage = get_tile_size(CB_M);
#if defined(MKX_XSL)
    // batched expert TP: groups of GS experts; per group one routing weight tile per owned shard whose expert exists
    // (the compute scales that m tile by it, xc.cpp), rows [0, T) written as 32-bit pairs (rows >= T: m is 0 there)
    constexpr uint32_t XW = MKX_XSL_W, GS = 64 / MKX_XSL_W;  // slice width (m tiles), experts per group
    const uint32_t npos = (n + GS - 1) / GS;
    const uint32_t rows_t = list[1];
    {
        // every CB_RS page zeroed once (rows >= T are never written; finite values times m's zero rows)
        const uint32_t rs0 = get_write_ptr(CB_RS);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < MKX_XSL_RS_PAGES * 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, rs0 + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
    // group pos's weight tiles; pushed one group ahead of the gather (the compute's gate_up(pos) needs them before
    // down(pos - 1), i.e. before this RISC finished the gather of pos - 1): 2 groups x <= 3 pages
    auto push_rs = [&](uint32_t pos) {
        for (uint32_t s = 0; s < nsh; ++s) {
            const uint32_t q = GS * pos + shard[s] / XW;
            if (q >= n) {
                continue;
            }
            cb_reserve_back(CB_RS, 1);
            volatile tt_l1_ptr uint32_t* rs32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_RS));
            for (uint32_t b = 0; b < rows_t; ++b) {
                const uint32_t w = wl[q * 32 + b];
                const uint32_t ww = w | (w << 16);
                const uint32_t f = tile_elem(b, 0) / 2;  // face (b / 16, 0) row b % 16; face (b / 16, 1) at + 128 words
                for (uint32_t c = 0; c < 8; ++c) {
                    rs32[f + c] = ww;
                    rs32[f + 128 + c] = ww;
                }
            }
            cb_push_back(CB_RS, 1);
        }
    };
    static_assert(MKX_XSL_RS_PAGES >= 6, "MKX_XSL: 2 groups of routing weight tiles");
    if (npos > 0) {
        push_rs(0);
    }
#else
    const uint32_t npos = n;
#endif
    for (uint32_t pos = 0; pos < npos; ++pos) {
        const uint32_t p = pos & 1;
        const uint32_t sem_mw = p ? MKX_SEM_MW1 : MKX_SEM_MW0;
        const uint32_t sem_mr = p ? MKX_SEM_MR1 : MKX_SEM_MR0;
#if defined(MKX_XSL)
        if (pos + 1 < npos) {
            push_rs(pos + 1);
        }
#else
        // routing weight tile of this expert
        cb_reserve_back(CB_RS, 1);
        volatile tt_l1_ptr uint16_t* rs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_RS));
        for (uint32_t b = 0; b < 32; ++b) {
            const uint16_t w = wl[pos * 32 + b];
            for (uint32_t c = 0; c < 32; ++c) {
                rs[tile_elem(b, c)] = w;
            }
        }
        cb_push_back(CB_RS, 1);
#endif
#if defined(MKX_XTP)
        // the gate | up partials: senders write theirs into the leader's CB_PART slot (pos & 1) and count; the leader
        // hands the slot to its compute once both of this expert's partials landed (a sender's partial of pos + 2
        // follows its down of pos, which needs the m tile the leader made after popping slot pos & 1)
        if (xrole != 0) {
            cb_wait_front(CB_PS, 2);
            noc_async_write(
                get_read_ptr(CB_PS),
                leader_part + ((pos & 1) * XTP_PART_SLOT_TILES + (xrole - 1) * 2) * part_page,
                2 * part_page);
            noc_async_write_barrier();
            noc_semaphore_inc(leader_sem, 1);
            cb_pop_front(CB_PS, 2);
        } else {
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MKX_XTP_SEM)), 2 * (pos + 1));
            cb_reserve_back(CB_PART, XTP_PART_SLOT_TILES);
            cb_push_back(CB_PART, XTP_PART_SLOT_TILES);
            // m all-gather through buffer p: the leaders write the slice's m tiles (one per bank)
            if (pos >= 2) {
                mk::wait_handoff(sem_mr, NREAD);  // every reader copied buffer p (expert pos - 2)
            }
            cb_wait_front(CB_MO, 1);
            noc_async_write_page(p * I_TILES + xbank, m, get_read_ptr(CB_MO));
            noc_async_write_barrier();
            cb_pop_front(CB_MO, 1);
            mk::signal_handoff(sem_mw, ph_rt0 + COORDS, NREAD);
        }
        mk::wait_handoff(sem_mw, MKX_XTP_W);  // the slice's MKX_XTP_W (8) m tiles landed
#else
        // m all-gather through buffer p
        if (pos >= 2) {
            mk::wait_handoff(sem_mr, NREAD);  // every reader copied buffer p (expert pos - 2)
        }
        for (uint32_t s = 0; s < nsh; ++s) {
            cb_wait_front(CB_MO, 1);
            noc_async_write_page(p * I_TILES + shard[s], m, get_read_ptr(CB_MO));
            noc_async_write_barrier();
            cb_pop_front(CB_MO, 1);
        }
        mk::signal_handoff(sem_mw, ph_rt0 + COORDS, NREAD);
        mk::wait_handoff(sem_mw, NREAD);
#endif
        cb_reserve_back(CB_M, I_TILES);
        const uint32_t ml1 = get_write_ptr(CB_M);
        for (uint32_t i = 0; i < I_TILES; ++i) {
            noc_async_read_page(p * I_TILES + i, m, ml1 + i * mpage);
        }
        noc_async_read_barrier();
        cb_push_back(CB_M, I_TILES);
        mk::signal_handoff(sem_mr, ph_rt0 + COORDS, NREAD);
    }
#if defined(MK_LAYERS) && MK_LAYERS > 1
#if defined(MKX_XTP)
#error "Stage 14b stack: expert TP (MKX_XTP) is not stackable (its partial counter MKX_XTP_SEM runs on across layers)"
#endif
    // stack: consume the read acknowledgements of the last two positions (within one call they are waited for two
    // positions later only), so the next layer's parity semaphores start at 0
    // (npos = the gather positions this call ran: n experts, or n / 8 groups under MKX_XSL)
    for (uint32_t pos = npos >= 2 ? npos - 2 : 0; pos < npos; ++pos) {
        mk::wait_handoff((pos & 1) ? MKX_SEM_MR1 : MKX_SEM_MR0, NREAD);
    }
#endif
#if defined(MKX_XTP)
    noc_async_atomic_barrier();  // a sender's last partial count landed
#endif
    MK_MARK();

    // the output: 2 tiles per owned shard
    cb_wait_front(CB_Y, 2 * nsh);
    const uint32_t yl1 = get_read_ptr(CB_Y), ypage = get_tile_size(CB_Y);
#if defined(MKX_FAB_RT)
    // Stage 14 TP / EP (tt/tp8_sparse.py): this chip's routed PARTIAL goes to the in-program fabric all-reduce instead
    // of y: the valid rows of each owned tile -> lane (tile / 64)'s own buffer at (tile % 64) * V (mk/reader_a.cpp's
    // packing), then OWN_READY on both lanes from EVERY reader (also one owning no shard); the lanes signal the owners.
    // Runtime args at MKX_FAB_RT: lane 0 (x, y), lane 1 (x, y), own address, lane sync address.
    {
        constexpr uint32_t FCH = MKX_FAB_BROWS * 32;
        constexpr uint32_t FV = MKX_FAB_BROWS >= 32 ? 2048 : 2 * FCH;
        const uint32_t own = get_arg_val<uint32_t>(MKX_FAB_RT + 4);
        for (uint32_t s = 0; s < nsh; ++s) {
            for (uint32_t k = 0; k < 2; ++k) {
                const uint32_t tt = 2 * shard[s] + k, lane = tt / 64;
                const uint32_t src = yl1 + (2 * s + k) * ypage;
                const uint64_t dst = get_noc_addr(
                    get_arg_val<uint32_t>(MKX_FAB_RT + 2 * lane),
                    get_arg_val<uint32_t>(MKX_FAB_RT + 2 * lane + 1),
                    own + (tt % 64) * FV);
                if constexpr (MKX_FAB_BROWS >= 32) {
                    noc_async_write(src, dst, 2048);
                } else {
                    noc_async_write(src, dst, FCH);
                    noc_async_write(src + 512, dst + FCH, FCH);
                }
            }
        }
        noc_async_write_barrier();
        const uint32_t lsync = get_arg_val<uint32_t>(MKX_FAB_RT + 5);  // OWN_READY = word 0 of the lane's sync words
        for (uint32_t lane = 0; lane < 2; ++lane) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(MKX_FAB_RT + 2 * lane),
                    get_arg_val<uint32_t>(MKX_FAB_RT + 2 * lane + 1),
                    lsync),
                1);
        }
        noc_async_atomic_barrier();
    }
#else
    for (uint32_t s = 0; s < nsh; ++s) {
        noc_async_write_page(2 * shard[s], y, yl1 + (2 * s) * ypage);
        noc_async_write_page(2 * shard[s] + 1, y, yl1 + (2 * s + 1) * ypage);
    }
    noc_async_write_barrier();
#endif
    cb_pop_front(CB_Y, 2 * nsh);
#if defined(MKX_SIGNAL_SEM)  // the routed output landed: the post-mix owners (count, then their NoC (x, y))
    mk::signal_handoff(
        MKX_SIGNAL_SEM, ph_rt0 + COORDS + 2 * NREAD + 1, get_arg_val<uint32_t>(ph_rt0 + COORDS + 2 * NREAD));
#endif
    MK_MARK();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
}
