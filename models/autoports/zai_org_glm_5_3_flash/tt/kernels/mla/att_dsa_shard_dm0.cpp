// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel attention phase, sequence-sharded DSA + split-K (kernels/mla/dsa_shard.hpp) — BRISC (NoC0) of owner
// core (g, c) on chip `rank` of C. The chip holds 1 / C of every context (unit = 128 tokens, owner u mod C): its local
// page table, its local pooled keys. Per launch of T tokens:
//   pre-sweep: per token the I4 inputs (q_idx row, wsc, this core's LOCAL pooled-key tiles) + the score plan
//              (npt, 0, 0x5C) -- dense tokens: (0, 0, 0x5C), nothing scored;
//   pass 0 (the other head tile: the PEER's heads at TP2, the other mesh row's at TP8) and pass 1 (this chip's / this
//              row's tile): per token the K tiles of this chip's selected pools (CB_ASEL page t: the local ids after
//              the global merge, + the open pool's tail rows when this chip owns the open pool) -- or, below the
//              sparse bound, this chip's local rows 0 .. n_local - 1 -- in batches of <= NKMAX tiles, each announced
//              by (bk, bm, last) after the pass header (nk, mi, 0x5D); the q tile row of the pass.
// The two passes share one CB_ASEL page per token (T pages; popped at the end). Both K gathers read the same local
// rows: the latents never leave this chip. The q tile: TP2 pass 0 from the all-gathered y_ag (the peer's block), pass 1
// from q_lat; TP8 (MLA_DSA_SHARD_LINE = 4, kernels/mla/dsa_shard.hpp) whole from q_asm (rev 59: the index core
// assembles both 32-head tiles once per chip and token from the 4 chips' 8-row pieces in y_ag; rev 58 gathered 128 runs
// of 256 B per push on every owner).
// Compile-time args: att_dsa_dm0.cpp's, then the TensorAccessorArgs of y_ag (TP8: q_asm).
// Runtime args: att_dsa_dm0.cpp's (q_lat, cache, page table, cur_pos, mask, g, c, T, q_idx, wsc, pooled keys), then
//   rank, C, peer block (the peer's position in y_ag; unused at TP8), the y_ag address (TP8: q_asm's) [, the token lane
//   with MLA_ATT_LANES].
// Rev 60 (W, T > 4): token lanes (att_common.hpp LANES) -- this core's lane takes tokens t0, t0 + LANES, ..; c / C are
//   lane-local; CB_ASEL holds one page per lane token (page j = t / LANES).
// Rev 61 MLA_DSA_ONE_SLOT (Stage 17's speculative verify on the shard, dsa/spec_on_shard_design.md): the rows are
//   consecutive positions of the launch's slot -> every row scores that slot's local pooled keys (att_dsa_dm0.cpp's
//   rev 56 addressing); each row's closed-pool bound stays its own (n_closed_local(p)), and IXR_SEM (ix_dm1: every
//   row's ring / pooled-key writes, in row order) precedes the first sparse row's pooled-key reads.
// Rev 63 MLA_DSA_PAGED_POOLS (doc/ttnn/12-fast-start/paged_pooled_keys.md rev 3, the padded layout): `pools` is the
//   per-chip paged store [nb_local, 1, 32, 128] (local latent block lb = page lb, its 16 pools at rows 0..15); a unit
//   tile = the top halves (faces 0-1) of pages row[2u], row[2u + 1] of the token's local page-table row. TP_TILES and
//   the slot are unused (the table row is the slot's).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_shard.hpp"

namespace {
FORCE_INLINE void info3(uint32_t cb, uint32_t a, uint32_t b, uint32_t c) {
    cb_reserve_back(cb, 1);
    volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
    info[0] = a;
    info[1] = b;
    info[2] = c;
    invalidate_l1_cache();
    cb_push_back(cb, 1);
}
constexpr uint32_t KIND_SCORE = 0x5C, KIND_PASS = 0x5D;
}  // namespace

void kernel_main() {
    constexpr uint32_t CB_AQ = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t CB_AK = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t CB_AMASK = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t CB_AINFO = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t CB_ASCR = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t C = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t HPT = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t WAIT_SEM = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr uint32_t LAT_SEM = get_compile_time_arg_val(PH_CT0 + 10);
    constexpr uint32_t LAT_COUNT = get_compile_time_arg_val(PH_CT0 + 11);
    constexpr uint32_t CB_AID = get_compile_time_arg_val(PH_CT0 + 12);
    constexpr uint32_t CB_AQI = get_compile_time_arg_val(PH_CT0 + 13);
    constexpr uint32_t CB_AWS = get_compile_time_arg_val(PH_CT0 + 14);
    constexpr uint32_t CB_APK = get_compile_time_arg_val(PH_CT0 + 15);
    constexpr uint32_t CB_ASEL = get_compile_time_arg_val(PH_CT0 + 16);
    constexpr uint32_t IXR_SEM = get_compile_time_arg_val(PH_CT0 + 17);
    constexpr uint32_t QI_SEM = get_compile_time_arg_val(PH_CT0 + 18);
    constexpr uint32_t QI_COUNT = get_compile_time_arg_val(PH_CT0 + 19);
    constexpr uint32_t TP_TILES = get_compile_time_arg_val(PH_CT0 + 20);
    constexpr auto q_args = TensorAccessorArgs<PH_CT0 + 21>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto qi_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<qi_args.next_compile_time_args_offset()>();
    constexpr auto pk_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto yag_args = TensorAccessorArgs<pk_args.next_compile_time_args_offset()>();
#if defined(MLA_DSA_DENSE_WINDOW)  // the replicated dense window
                                   // (doc/ttnn/14-production-tp8-megakernel/dsa/dense_window_design.md)
    constexpr auto win_args = TensorAccessorArgs<yag_args.next_compile_time_args_offset()>();
    constexpr uint32_t WIN_BLOCKS = MLA_DSA_WIN_BLOCKS;  // window pages per slot (>= DENSE_MAX / BLOCK)
#endif
    constexpr uint32_t DT = mlaatt::DT;
    constexpr uint32_t TPB = BLOCK / 32;  // row tiles per cache page
    static_assert(HPT == 1, "split-K: one head group per chip (TP2: 32 heads; TP8: 8 heads + the q assembly)");
    // (at function scope: a static_assert inside `if constexpr` of a non-template lambda is evaluated in every build)
    static_assert(mlashard::LINE == 1 || (mlashard::LINE == 4 && DT == 16), "the 2x4 q assembly: 4 pieces of 8 rows");
    const auto qlat = TensorAccessor(q_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto cache = TensorAccessor(k_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto pt = TensorAccessor(p_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto cur = TensorAccessor(c_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const auto mask = TensorAccessor(m_args, get_arg_val<uint32_t>(ph_rt0 + 4));
    const uint32_t g = get_arg_val<uint32_t>(ph_rt0 + 5);
    const uint32_t c = get_arg_val<uint32_t>(ph_rt0 + 6);
    const uint32_t t_rows = get_arg_val<uint32_t>(ph_rt0 + 7);
    const auto qidx = TensorAccessor(qi_args, get_arg_val<uint32_t>(ph_rt0 + 8));
    const auto wsc = TensorAccessor(w_args, get_arg_val<uint32_t>(ph_rt0 + 9));
    const auto pools = TensorAccessor(pk_args, get_arg_val<uint32_t>(ph_rt0 + 10));
    const uint32_t rank = get_arg_val<uint32_t>(ph_rt0 + 11);
    const uint32_t nchips = get_arg_val<uint32_t>(ph_rt0 + 12);
    const uint32_t peer_blk = get_arg_val<uint32_t>(ph_rt0 + 13);
    const auto yag = TensorAccessor(yag_args, get_arg_val<uint32_t>(ph_rt0 + 14));
#if defined(MLA_ATT_LANES)
    const uint32_t t0 = get_arg_val<uint32_t>(ph_rt0 + 15);
#else
    const uint32_t t0 = 0;
#endif
#if defined(MLA_DSA_DENSE_WINDOW)
#if defined(MLA_ATT_LANES)
    const auto win = TensorAccessor(win_args, get_arg_val<uint32_t>(ph_rt0 + 16));
#else
    const auto win = TensorAccessor(win_args, get_arg_val<uint32_t>(ph_rt0 + 15));
#endif
#endif
    const uint32_t tb = get_tile_size(CB_AK);
    ASSERT((get_write_ptr(CB_AK) & 63) == 0 && (tb & 63) == 0);
    ASSERT(tb == mlashard::TB);  // (review 2 item 7: the q tiles / pages are 2-KiB bf16 tiles)
    ASSERT(t_rows <= mlashard::T_MAX && mlashard::lane_tokens(t_rows, t0) <= mlashard::TL);

    MK_MARK();
    const uint32_t scr = get_write_ptr(CB_ASCR);  // cur_pos row at 0, one page-table row at 256
    noc_async_read_page(0, cur, scr);
    cb_reserve_back(CB_AID, 1);
    noc_async_read_page(32, mask, get_write_ptr(CB_AID));
    noc_async_read_barrier();
    cb_push_back(CB_AID, 1);
    invalidate_l1_cache();
    if constexpr (LAT_COUNT > 0) {
        mk::wait_handoff(LAT_SEM, LAT_COUNT);
    }
    bool waited = WAIT_COUNT == 0, ix_waited = false;
    auto wait_q = [&]() {
        if (!waited) {
            mk::wait_handoff(WAIT_SEM, WAIT_COUNT);
            waited = true;
        }
    };
    // the q tile row of token t: pass 0 the peer's heads from y_ag (landed: CB_ASEL page t was pushed after the global
    // merge, which waited for the all-gather), pass 1 this chip's heads from q_lat
    auto push_q = [&](uint32_t pass, uint32_t t, bool win = false) {
        cb_reserve_back(CB_AQ, DT);
        const uint32_t q1 = get_write_ptr(CB_AQ);
        if (win) {  // the dense window's pass: this chip's own head tile from q_lat (TP8: its 8 heads = rows 0 .. 7)
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page((t * HPT + g) * DT + d, qlat, q1 + d * tb);
            }
        } else if constexpr (mlashard::LINE > 1) {
            // TP8 (rev 59): head tile `tile` (pass 0 the other mesh row's, pass 1 this row's), assembled once per chip
            // by the index core's NCRISC into q_asm (kernels/mla/ix_dm1.cpp; the 10th message implies it landed): 16
            // pages
            const uint32_t row = mlashard::pair_pos(rank);
            const uint32_t tile = pass == 0 ? 1 - row : row;
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page(mlashard::qasm_page(t_rows, tile, t, d), yag, q1 + d * tb);
            }
        } else {
            for (uint32_t d = 0; d < DT; ++d) {
                if (pass == 0) {
                    noc_async_read_page(mlashard::ag_q_page(t_rows, peer_blk, t, d), yag, q1 + d * tb);
                } else {
                    noc_async_read_page((t * HPT + g) * DT + d, qlat, q1 + d * tb);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_AQ, DT);
    };
    auto pad_k = [&](uint32_t nk) {
        if (nk < mlaatt::NKMAX) {
            cb_reserve_back(CB_AK, (mlaatt::NKMAX - nk) * DT);
            cb_push_back(CB_AK, (mlaatt::NKMAX - nk) * DT);
        }
    };
    auto push_mask = [&](uint32_t lim) {
        cb_reserve_back(CB_AMASK, 1);
        noc_async_read_page(lim, mask, get_write_ptr(CB_AMASK));
        noc_async_read_barrier();
        cb_push_back(CB_AMASK, 1);
    };
    auto pos_of = [&](uint32_t t) {
        const int32_t pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[t];
        return pos < 0 ? 0u : static_cast<uint32_t>(pos);
    };
    const uint32_t sel_page = get_local_cb_interface(CB_ASEL).fifo_page_size;

    // ---- pre-sweep: I4 inputs of every token of the lane (the compute scores them in order; the NCRISC selects
    // locally)
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const uint32_t p = pos_of(t);
        if (!mladsa::sparse(p)) {
            info3(CB_AINFO, 0, 0, KIND_SCORE);
            continue;
        }
        const uint32_t n_closed = mlashard::n_closed_local(p, rank, nchips);
        const uint32_t npt = mladsa::pool_tiles(n_closed, c, C);
        info3(CB_AINFO, npt, 1, KIND_SCORE);  // word 1 = 1: a sparse token (its q_idx / wsc / pooled keys follow)
        if (!ix_waited) {
            mk::wait_handoff(IXR_SEM, 1);
            mk::wait_handoff(QI_SEM, QI_COUNT);
            ix_waited = true;
        }
        cb_reserve_back(CB_AQI, 4);
        for (uint32_t d = 0; d < 4; ++d) {
            noc_async_read_page(t * 4 + d, qidx, get_write_ptr(CB_AQI) + d * tb);
        }
        cb_reserve_back(CB_AWS, 1);
        noc_async_read_page(0, wsc, get_write_ptr(CB_AWS));
        noc_async_read_barrier();
        cb_push_back(CB_AQI, 4);
        cb_push_back(CB_AWS, 1);
#if defined(MLA_DSA_PAGED_POOLS)
        // rev 63: the pooled keys live with the latent blocks (one [32, 128] page per local block, rows 0..15 = its 16
        // pools = the FIRST 1 KiB of each of the page's 4 tiles): local unit u = blocks row[2u], row[2u + 1] of this
        // token's LOCAL page-table row (the verify rows of a ONE_SLOT launch repeat the slot's row)
        mlaatt::read_pt_prefix(
            pt,
            t,
            scr + 256,
            (mlashard::n_local_rows(p, rank, nchips) + BLOCK - 1) / BLOCK + 1);  // the visible local entries
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* prow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 256);
#endif
        for (uint32_t i0 = 0; i0 < npt; i0 += mladsa::SB) {
            const uint32_t nb = npt - i0 < mladsa::SB ? npt - i0 : mladsa::SB;
            cb_reserve_back(CB_APK, 4 * mladsa::SB);
            const uint32_t l1 = get_write_ptr(CB_APK);
            for (uint32_t j = 0; j < nb; ++j) {
#if defined(MLA_DSA_PAGED_POOLS)
                const uint32_t u = c + (i0 + j) * C;  // local unit (one pooled tile = 32 pools)
                const uint32_t lb0 = prow[2 * u], lb1 = prow[2 * u + 1];
                for (uint32_t d = 0; d < 4; ++d) {
                    const uint32_t dst = l1 + (4 * j + d) * tb;
                    noc_async_read(pools.get_noc_addr(lb0 * 4 + d, 0), dst, tb / 2);           // rows 0..15
                    noc_async_read(pools.get_noc_addr(lb1 * 4 + d, 0), dst + tb / 2, tb / 2);  // rows 16..31
                }
#else
#if defined(MLA_DSA_ONE_SLOT)  // rev 61: a speculative verify's rows all score the launch's slot (its local pools)
                const uint32_t tile0 = (c + (i0 + j) * C) * 4;
#else
                const uint32_t tile0 = (t * TP_TILES + c + (i0 + j) * C) * 4;
#endif
                for (uint32_t d = 0; d < 4; ++d) {
                    noc_async_read_page(tile0 + d, pools, l1 + (4 * j + d) * tb);
                }
#endif
            }
            noc_async_read_barrier();
            cb_push_back(CB_APK, 4 * mladsa::SB);
        }
    }
    MK_MARK();

    // ---- THE DENSE WINDOW (MLA_DSA_DENSE_WINDOW): a launch whose EVERY row is dense (p + 1 <= DENSE_MAX; inactive
    // rows count as dense) attends this chip's heads over rows 0 .. p of the replicated window in ONE pass (pass 1's q:
    // q_lat): no pass 0, no y_ag. Every core of every chip decides the same from cur_pos (all t_rows, not only the
    // lane's). The mode header tells the compute (att_c.cpp); the writer decides the same (att_dsa_shard_dm1.cpp).
#if defined(MLA_DSA_DENSE_WINDOW)
    bool all_dense = true;
    for (uint32_t t = 0; t < t_rows; ++t) {
        all_dense = all_dense && !mladsa::sparse(pos_of(t));
    }
    info3(CB_AINFO, all_dense ? 1u : 0u, 0, KIND_PASS);
#else
    constexpr bool all_dense = false;
#endif
    // ---- the two split-K passes
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t pass = all_dense ? 1 : 0; pass < 2; ++pass) {
        if (pass == 1) {
            MK_MARK();  // probe: pass 0's K / q pushed
        }
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
            const uint32_t p = pos_of(t);
            const uint32_t j = mlashard::lane_tok(t);  // this lane's CB_ASEL page of token t
#if defined(MLA_DSA_DENSE_WINDOW)
            if (all_dense) {  // the window pass: rows 0 .. p of slot t's window, chunked over the group (att_dsa_dm0
                              // D1)
                // an IDLE row (cur_pos < 0, a seat beyond the window's slots possibly) reads nothing: one empty batch
                const bool idle = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[t] < 0;
                const uint32_t nk = idle ? 0 : mlaatt::chunk_tiles(p, c, C);
                const uint32_t last = p / 32;
                const uint32_t mi = (!idle && last % C == c) ? (last - c) / C : 0xFFFFu;
                info3(CB_AINFO, nk, mi, KIND_PASS);
                if (nk == 0) {
                    info3(CB_AINFO, 0, 0xFFFFu, 1);  // one empty last batch: no K / q pages
                    continue;
                }
                for (uint32_t b0 = 0; b0 < nk; b0 += mlaatt::NKMAX) {
                    const uint32_t bk = nk - b0 < mlaatt::NKMAX ? nk - b0 : mlaatt::NKMAX;
                    const uint32_t bm = (mi != 0xFFFFu && mi >= b0 && mi < b0 + bk) ? mi - b0 : 0xFFFFu;
                    info3(CB_AINFO, bk, bm, b0 + bk == nk ? 1 : 0);
                    if (bm != 0xFFFFu) {
                        push_mask(p % 32);
                    }
                    for (uint32_t i = b0; i < b0 + bk; ++i) {
                        const uint32_t jt = c + i * C;
                        const uint32_t tile0 = (t * WIN_BLOCKS + jt / TPB) * (TPB * DT) + (jt % TPB) * DT;
                        cb_reserve_back(CB_AK, DT);
                        const uint32_t l1 = get_write_ptr(CB_AK);
                        for (uint32_t d = 0; d < DT; ++d) {
                            noc_async_read_page(tile0 + d, win, l1 + d * tb);
                        }
                        noc_async_read_barrier();
                        cb_push_back(CB_AK, DT);
                    }
                    pad_k(bk);
                    wait_q();
                    push_q(1, t, true);
                }
                continue;
            }
#endif
            mlaatt::read_pt_prefix(
                pt,
                t,
                scr + 256,
                (mlashard::n_local_rows(p, rank, nchips) + BLOCK - 1) / BLOCK +
                    1);                     // this chip's LOCAL page-table row (the visible entries)
            cb_wait_front(CB_ASEL, j + 1);  // page j: pushed after the global merge (the all-gather landed)
            noc_async_read_barrier();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 256);
            volatile tt_l1_ptr uint32_t* sel =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_ASEL) + j * sel_page);
            if (!mladsa::sparse(p)) {  // D1 over this chip's local rows 0 .. n_loc - 1
                const uint32_t n_loc = mlashard::n_local_rows(p, rank, nchips);
                const uint32_t nk = n_loc == 0 ? 0 : mlaatt::chunk_tiles(n_loc - 1, c, C);
                const uint32_t last = n_loc == 0 ? 0 : (n_loc - 1) / 32;
                const uint32_t mi = (n_loc > 0 && last % C == c) ? (last - c) / C : 0xFFFFu;
                info3(CB_AINFO, nk, mi, KIND_PASS);
                if (nk == 0) {
                    info3(CB_AINFO, 0, 0xFFFFu, 1);  // one empty last batch: no K / q pages
                    continue;
                }
                for (uint32_t b0 = 0; b0 < nk; b0 += mlaatt::NKMAX) {
                    const uint32_t bk = nk - b0 < mlaatt::NKMAX ? nk - b0 : mlaatt::NKMAX;
                    const uint32_t bm = (mi != 0xFFFFu && mi >= b0 && mi < b0 + bk) ? mi - b0 : 0xFFFFu;
                    info3(CB_AINFO, bk, bm, b0 + bk == nk ? 1 : 0);
                    if (bm != 0xFFFFu) {
                        push_mask((n_loc - 1) % 32);
                    }
                    for (uint32_t i = b0; i < b0 + bk; ++i) {
                        const uint32_t jt = c + i * C;
                        const uint32_t tile0 = row[jt / TPB] * (TPB * DT) + (jt % TPB) * DT;
                        cb_reserve_back(CB_AK, DT);
                        const uint32_t l1 = get_write_ptr(CB_AK);
                        for (uint32_t d = 0; d < DT; ++d) {
                            noc_async_read_page(tile0 + d, cache, l1 + d * tb);
                        }
                        noc_async_read_barrier();
                        cb_push_back(CB_AK, DT);
                    }
                    pad_k(bk);
                    if (pass == 1) {
                        wait_q();
                    }
                    push_q(pass, t);
                }
                continue;
            }
            // ---- S1 over this chip's selected pools (local ids) + the tail on core 0 of the open pool's owner
            const uint32_t n_sel = sel[0], tail_rows = sel[1], tail_pool = sel[2];
            const uint32_t entries = n_sel + (tail_rows > 0 ? 1 : 0);
            const uint32_t ntiles = (entries + mladsa::POOLS_PER_TILE - 1) / mladsa::POOLS_PER_TILE;
            info3(CB_AINFO, 0, 0, KIND_PASS);
            if (ntiles == 0) {
                info3(CB_AINFO, 0, 0xFFFFu, 1);
                continue;
            }
            for (uint32_t b0 = 0; b0 < ntiles; b0 += mlaatt::NKMAX) {
                const uint32_t nk = ntiles - b0 < mlaatt::NKMAX ? ntiles - b0 : mlaatt::NKMAX;
                const bool last_batch = b0 + nk == ntiles;
                const uint32_t e_end = last_batch ? entries : (b0 + nk) * mladsa::POOLS_PER_TILE;
                const uint32_t e_last0 = (b0 + nk - 1) * mladsa::POOLS_PER_TILE;
                const bool tail_here = last_batch && tail_rows > 0;
                const uint32_t rows_last =
                    (e_end - e_last0) * mladsa::KPOOL - (tail_here ? mladsa::KPOOL - tail_rows : 0);
                const uint32_t mi = rows_last < 32 ? nk - 1 : 0xFFFFu;
                info3(CB_AINFO, nk, mi, last_batch ? 1 : 0);
                if (mi != 0xFFFFu) {
                    push_mask(rows_last - 1);
                }
                for (uint32_t k = 0; k < nk; ++k) {
                    cb_reserve_back(CB_AK, DT);
                    const uint32_t l1 = get_write_ptr(CB_AK);
                    const uint32_t e0 = (b0 + k) * mladsa::POOLS_PER_TILE;
                    const uint32_t e1 = e0 + mladsa::POOLS_PER_TILE < e_end ? e0 + mladsa::POOLS_PER_TILE : e_end;
                    if (e1 - e0 < mladsa::POOLS_PER_TILE) {
                        for (uint32_t off = 0; off < DT * tb; off += MEM_ZEROS_SIZE) {
                            noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
                        }
                        noc_async_read_barrier();
                    }
                    for (uint32_t e = e0; e < e1;) {
                        const uint32_t q = e < n_sel ? sel[3 + e] : tail_pool;  // LOCAL pool ids
                        const uint32_t tok = q * mladsa::KPOOL;                 // local token row
                        const uint32_t in_page = tok % BLOCK;
                        const uint32_t tile0 = row[tok / BLOCK] * (TPB * DT) + (in_page / 32) * DT;
                        const uint32_t r0 = in_page % 32, s = e - e0;
                        const uint32_t src_off = (r0 / 16) * 1024 + (r0 % 16) * 32;
                        const uint32_t dst_off = ((s * mladsa::KPOOL) / 16) * 1024 + ((s * mladsa::KPOOL) % 16) * 32;
                        uint32_t run = 1;
#if defined(MLA_DSA_RUNS)
                        if (e < n_sel) {
                            while (e + run < e1 && e + run < n_sel && sel[3 + e + run] == q + run &&
                                   (q % 4) + run < 4 && (s % 4) + run < 4) {
                                ++run;
                            }
                        }
#endif
                        for (uint32_t d = 0; d < DT; ++d) {
                            const uint64_t src = cache.get_noc_addr(tile0 + d, src_off);
                            noc_async_read(src, l1 + d * tb + dst_off, 128 * run);
                            noc_async_read(src + 512, l1 + d * tb + dst_off + 512, 128 * run);
                        }
                        e += run;
                    }
                    noc_async_read_barrier();
                    cb_push_back(CB_AK, DT);
                }
                pad_k(nk);
                if (pass == 1) {
                    wait_q();
                }
                push_q(pass, t);
            }
        }
    }
    if (mlashard::lane_tokens(t_rows, t0) > 0) {
        if (all_dense) {  // (the window pass never waited for the merge's pages: they still arrive, after the AG)
            cb_wait_front(CB_ASEL, mlashard::lane_tokens(t_rows, t0));
        }
        cb_pop_front(CB_ASEL, mlashard::lane_tokens(t_rows, t0));
    }
    if (!waited) {
        mk::wait_handoff(WAIT_SEM, WAIT_COUNT);  // the protocol stays balanced when no token needed q_lat
    }
    if (!ix_waited) {
        mk::wait_handoff(IXR_SEM, 1);
        mk::wait_handoff(QI_SEM, QI_COUNT);
    }
    MK_MARK();
}
