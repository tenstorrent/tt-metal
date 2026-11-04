// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel latent core — NCRISC (NoC1): the KX latent tiles (rows = tokens) -> the latent tensor (validation /
// the op chain's `latent`) and row t into the paged latent cache at cur_pos[t] (paged_update_cache's contract: page =
// page_table[t][p / BLOCK], row p % BLOCK of the bf16 TILE page [BLOCK, KX * 32]; rows with p < 0 are inactive and
// skipped). Each touched cache tile is read, its two 32-B face rows of the token replaced, and written back (DRAM
// writes stay whole tiles). MLA_LAT_NB = NB > 1 (tt/mk_mla.py, t > 4): every active row's page-table row is read up
// front, then up to NB rows' cache tiles per read-modify-write round trip (a batch ends before a row whose cache tile
// is already in it: the same tile's read-modify-writes must not overlap).
// Then (QL_SEM / QL_COUNT): once the W_uk readers signal, copy the q_lat staging tensor (L1, written by the readers'
// face-row scatter) to the DRAM q_lat the flash-MLA op reads (it takes an interleaved Q from DRAM only), whole tiles.
// After the cache write, LAT_SEM is incremented on the LAT_NCONS attention cores (the whole-half program). BLOCK = 0:
// no cache write (the q-norm core of the whole-half program: RMSNorm_1536(q_a) -> q_r for the q_b readers).
// Compile-time args: CB_OUT, CB_SCR, KX, BLOCK (cache rows per page), QL_SEM, QL_COUNT, LAT_SEM, LAT_NCONS, ON (output
//   tiles per core: ON tiles at latent-tensor page OOFF, a runtime arg; the cache write needs the whole row), then the
//   TensorAccessorArgs of the latent tensor, the cache, the page table, cur_pos, the q_lat staging tensor and the DRAM
//   q_lat.
// Runtime args: latent, cache, page table, cur_pos addresses, T (rows), page-table row bytes, q_lat staging, q_lat,
//   q_lat tiles (0: no copy), OOFF, then LAT_NCONS consumer (x, y) pairs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/mla_common.hpp"
#if defined(MLA_DSA_SHARD)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_shard.hpp"
#endif

void kernel_main() {
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(1);
    constexpr uint32_t KX = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(3);
    constexpr uint32_t QL_SEM = get_compile_time_arg_val(4);
    constexpr uint32_t QL_COUNT = get_compile_time_arg_val(5);
    constexpr uint32_t LAT_SEM = get_compile_time_arg_val(6);
    constexpr uint32_t LAT_NCONS = get_compile_time_arg_val(7);
    constexpr uint32_t ON = get_compile_time_arg_val(8);
    static_assert(BLOCK == 0 || ON == KX, "the cache write takes the whole latent row");
    constexpr auto l_args = TensorAccessorArgs<9>();
    constexpr auto c_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
#if defined(MLA_DSA_DENSE_WINDOW)  // the replicated dense window
                                   // (doc/ttnn/14-production-tp8-megakernel/dsa/dense_window_design.md)
    constexpr auto w_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
#if !defined(MLA_DSA_SHARD)
#error "MLA_DSA_DENSE_WINDOW: the sharded line only"
#endif
#endif
    const auto lat = TensorAccessor(l_args, get_arg_val<uint32_t>(0));
    const auto cache = TensorAccessor(c_args, get_arg_val<uint32_t>(1));
    const auto pt = TensorAccessor(p_args, get_arg_val<uint32_t>(2));
    const auto cur = TensorAccessor(q_args, get_arg_val<uint32_t>(3));
    const uint32_t t_rows = get_arg_val<uint32_t>(4);
    const auto ql_src = TensorAccessor(s_args, get_arg_val<uint32_t>(6));
    const auto ql_dst = TensorAccessor(d_args, get_arg_val<uint32_t>(7));
    const uint32_t ql_tiles = get_arg_val<uint32_t>(8);
#if defined(MLA_DSA_SHARD)  // after the LAT_NCONS consumer pairs: rank, C, the all-gather lane core + its wait
                            // semaphore (rev 60: t > 4 too -- the batched cache write writes the owned rows only)
    const uint32_t rank = get_arg_val<uint32_t>(10 + 2 * LAT_NCONS), nchips = get_arg_val<uint32_t>(11 + 2 * LAT_NCONS);
    const uint32_t ag_x = get_arg_val<uint32_t>(12 + 2 * LAT_NCONS), ag_y = get_arg_val<uint32_t>(13 + 2 * LAT_NCONS);
    const uint32_t ag_sem = get_arg_val<uint32_t>(14 + 2 * LAT_NCONS);
#endif
#if defined(MLA_DSA_DENSE_WINDOW)  // after the shard block (the probe pair, if any, follows: tt/mk_mla.py)
    const auto win = TensorAccessor(w_args, get_arg_val<uint32_t>(15 + 2 * LAT_NCONS));
#endif
    const uint32_t OOFF = get_arg_val<uint32_t>(9);
    constexpr uint32_t BLK = BLOCK > 0 ? BLOCK : 32;  // BLOCK = 0: no cache (the arithmetic below is then unused)
    constexpr uint32_t TILES_PER_BLOCK = (BLK / 32) * KX;
    constexpr uint32_t TB = mla::TILE_BYTES_BF16;

    MK_MARK();
    constexpr uint32_t NOUT = ON > 0 ? ON : 1;  // ON = 0: the one fp32 row-scale tile (its own page size)
    cb_wait_front(CB_OUT, NOUT);
    const uint32_t src = get_read_ptr(CB_OUT);
    for (uint32_t k = 0; k < NOUT; ++k) {
        noc_async_write_page(OOFF + k, lat, src + k * TB);
    }
#if defined(MLA_LAT_NB)
    constexpr uint32_t NB = MLA_LAT_NB;
#else
    constexpr uint32_t NB = 1;
#endif
    // scratch: [0, NB x KX tiles) cache tiles, then the cur_pos row and the page-table row(s)
    const uint32_t scr = get_write_ptr(CB_SCR);
    const uint32_t cur_l1 = scr + NB * KX * TB;
    const uint32_t pt_l1 = cur_l1 + 256;
    if constexpr (BLOCK > 0) {
        noc_async_read_page(0, cur, cur_l1);
        noc_async_read_barrier();
        invalidate_l1_cache();  // CPU loads of NoC-written L1 below
        volatile tt_l1_ptr int32_t* curv = reinterpret_cast<volatile tt_l1_ptr int32_t*>(cur_l1);
#if defined(MLA_LAT_NB)
        const uint32_t pt_bytes = (get_arg_val<uint32_t>(5) + 63) & ~63u;  // a page-table row as its page read lands
        // THE VISIBLE-PREFIX READ (att_common.hpp read_pt_prefix; the latent core has no att_common): only the row's
        // entries through the written block -- the full row was a capacity-sized read per token (32 KB on the pair's 1M
        // local table); the one entry read below is the same bytes
        const uint32_t pt_page = pt.get_aligned_page_size();
        for (uint32_t t = 0; t < t_rows; ++t) {
            if (curv[t] >= 0) {
                const uint32_t p_ = static_cast<uint32_t>(curv[t]);
#if defined(MLA_DSA_SHARD)
                if (mlashard::owner(p_, nchips) != rank) {
                    continue;  // not written on this chip (the loop below skips it the same way)
                }
                const uint32_t pl_ = mlashard::local_pos(p_, nchips);
#else
                const uint32_t pl_ = p_;
#endif
                const uint32_t want = (4 * (pl_ / BLK + 1) + 63) & ~63u;
                noc_async_read(pt.get_noc_addr(t), pt_l1 + t * pt_bytes, want < pt_page ? want : pt_page);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        uint32_t t = 0;
        while (t < t_rows) {
            uint32_t rows[NB], tiles[NB], n = 0;
            for (; t < t_rows && n < NB; ++t) {
                const int32_t p = curv[t];
                if (p < 0) {
                    continue;  // inactive row: no cache write
                }
#if defined(MLA_DSA_SHARD)  // rev 60 (kernels/mla/dsa_shard.hpp): the owner chip writes the row at its LOCAL position
                if (mlashard::owner(static_cast<uint32_t>(p), nchips) != rank) {
                    continue;
                }
                const uint32_t pl = mlashard::local_pos(static_cast<uint32_t>(p), nchips);
#else
                const uint32_t pl = static_cast<uint32_t>(p);
#endif
                const uint32_t page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_l1 + t * pt_bytes)[pl / BLK];
                const uint32_t tile0 = page * TILES_PER_BLOCK + ((pl % BLK) / 32) * KX;
                bool dup = false;
                for (uint32_t j = 0; j < n; ++j) {
                    dup = dup || tiles[j] == tile0;
                }
                if (dup) {
                    break;  // row t opens the next batch
                }
                rows[n] = t;
                tiles[n] = tile0;
                ++n;
            }
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t k = 0; k < KX; ++k) {
                    noc_async_read_page(tiles[j] + k, cache, scr + (j * KX + k) * TB);
                }
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t j = 0; j < n; ++j) {
#if defined(MLA_DSA_SHARD)
                const uint32_t r = (mlashard::local_pos(static_cast<uint32_t>(curv[rows[j]]), nchips) % BLK) % 32;
#else
                const uint32_t r = (static_cast<uint32_t>(curv[rows[j]]) % BLK) % 32;
#endif
                for (uint32_t k = 0; k < KX; ++k) {
                    const uint32_t d = scr + (j * KX + k) * TB + mla::row_off(r);
                    const uint32_t s = src + k * TB + mla::row_off(rows[j]);
                    mla::copy_row32(s, d);
                    mla::copy_row32(s + 512, d + 512);
                }
            }
            invalidate_l1_cache();  // the stores land before the NoC writes the tiles back
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t k = 0; k < KX; ++k) {
                    noc_async_write_page(tiles[j] + k, cache, scr + (j * KX + k) * TB);
                }
            }
            noc_async_write_barrier();
        }
#if defined(MLA_DSA_DENSE_WINDOW)
        // the dense window (t > 4): every dense row also to its slot's window page on EVERY chip, one row at a time
        // (as the per-row path; the scratch's first KX tiles are free again after the batched write-back)
        for (uint32_t t = 0; t < t_rows; ++t) {
            const int32_t p = curv[t];
            if (p < 0 || static_cast<uint32_t>(p) + 1 > mladsa::DENSE_MAX) {
                continue;
            }
            const uint32_t pw = static_cast<uint32_t>(p);
            const uint32_t wt0 = (t * MLA_DSA_WIN_BLOCKS + pw / BLK) * TILES_PER_BLOCK + ((pw % BLK) / 32) * KX;
            const uint32_t wr = (pw % BLK) % 32;
            for (uint32_t k = 0; k < KX; ++k) {
                noc_async_read_page(wt0 + k, win, scr + k * TB);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t k = 0; k < KX; ++k) {
                const uint32_t d = scr + k * TB + mla::row_off(wr);
                const uint32_t s = src + k * TB + mla::row_off(t);
                mla::copy_row32(s, d);
                mla::copy_row32(s + 512, d + 512);
            }
            invalidate_l1_cache();
            for (uint32_t k = 0; k < KX; ++k) {
                noc_async_write_page(wt0 + k, win, scr + k * TB);
            }
            noc_async_write_barrier();
        }
#endif
#else
        for (uint32_t t = 0; t < t_rows; ++t) {
            const int32_t p = curv[t];
            if (p < 0) {
                continue;  // inactive row: no cache write (the op chain's paged_update_cache skips it the same way)
            }
#if defined(MLA_DSA_DENSE_WINDOW)
            // the dense window: rows p + 1 <= DENSE_MAX on EVERY chip (slot t's page p / BLK of the window), as the
            // cache write below (every chip holds the token's latent row; the sharded cache keeps its owner-only write)
            if (static_cast<uint32_t>(p) + 1 <= mladsa::DENSE_MAX) {
                const uint32_t pw = static_cast<uint32_t>(p);
                const uint32_t wt0 = (t * MLA_DSA_WIN_BLOCKS + pw / BLK) * TILES_PER_BLOCK + ((pw % BLK) / 32) * KX;
                const uint32_t wr = (pw % BLK) % 32;
                for (uint32_t k = 0; k < KX; ++k) {
                    noc_async_read_page(wt0 + k, win, scr + k * TB);
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t k = 0; k < KX; ++k) {
                    const uint32_t d = scr + k * TB + mla::row_off(wr);
                    const uint32_t s = src + k * TB + mla::row_off(t);
                    mla::copy_row32(s, d);
                    mla::copy_row32(s + 512, d + 512);
                }
                invalidate_l1_cache();
                for (uint32_t k = 0; k < KX; ++k) {
                    noc_async_write_page(wt0 + k, win, scr + k * TB);
                }
                noc_async_write_barrier();
            }
#endif
#if defined(MLA_DSA_SHARD)  // kernels/mla/dsa_shard.hpp: the owner chip writes the row at its LOCAL position
            if (mlashard::owner(static_cast<uint32_t>(p), nchips) != rank) {
                continue;
            }
            const uint32_t pl = mlashard::local_pos(static_cast<uint32_t>(p), nchips);
#else
            const uint32_t pl = static_cast<uint32_t>(p);
#endif
            {  // the visible prefix through entry pl / BLK (as above)
                const uint32_t pt_page = pt.get_aligned_page_size(), want = (4 * (pl / BLK + 1) + 63) & ~63u;
                noc_async_read(pt.get_noc_addr(t), pt_l1, want < pt_page ? want : pt_page);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            const uint32_t page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_l1)[pl / BLK];
            const uint32_t within = pl % BLK;
            const uint32_t tile0 = page * TILES_PER_BLOCK + (within / 32) * KX;
            const uint32_t r = within % 32;
            for (uint32_t k = 0; k < KX; ++k) {
                noc_async_read_page(tile0 + k, cache, scr + k * TB);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t k = 0; k < KX; ++k) {
                // two 32-B face rows: row t of latent tile k -> row r of cache tile k (RISC copies, local L1)
                const uint32_t d = scr + k * TB + mla::row_off(r);
                const uint32_t s = src + k * TB + mla::row_off(t);
                mla::copy_row32(s, d);
                mla::copy_row32(s + 512, d + 512);
            }
            invalidate_l1_cache();  // the stores land before the NoC writes the tiles back
            for (uint32_t k = 0; k < KX; ++k) {
                noc_async_write_page(tile0 + k, cache, scr + k * TB);
            }
            noc_async_write_barrier();
        }
#endif
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, NOUT);
    if constexpr (LAT_NCONS > 0) {
        mk::signal_handoff(LAT_SEM, 10, LAT_NCONS);
    }
    MK_MARK();
    if constexpr (QL_COUNT == 0) {
#if defined(MK_PROBE_ARG)
        MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
        return;
    }
    // q_lat: staging (L1) -> DRAM, KX tiles of scratch at a time
    mk::wait_handoff(QL_SEM, QL_COUNT);
    MK_MARK();
    for (uint32_t i0 = 0; i0 < ql_tiles; i0 += KX) {
        const uint32_t n = ql_tiles - i0 < KX ? ql_tiles - i0 : KX;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read_page(i0 + i, ql_src, scr + i * TB);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
#if defined(MLA_DSA_SHARD) && defined(MLA_DSA_SHARD_LINE)
            // TP8 (kernels/mla/dsa_shard.hpp AG8): this chip's 8 head rows of q_lat tile (t, d) = face 0 rows 0..7 +
            // face 1 rows 0..7 (two 256-B runs) -> the compact q piece of token t's x_ag chunk (page d / 4, offset
            // (d % 4) 512); the pad rows 8..31 of q_lat never leave the chip
            static_assert(KX == mlashard::PPC, "q_lat tiles per token = the latent tiles (512 / 32)");
            const uint32_t tt_ = (i0 + i) / mlashard::PPC, d = (i0 + i) % mlashard::PPC;
            const uint64_t dst = ql_dst.get_noc_addr(mlashard::agx_q_page(t_rows, 0, tt_, d), mlashard::agx_q_off(d));
            noc_async_write(scr + i * TB, dst, 256);
            noc_async_write(scr + i * TB + 512, dst + 256, 256);
#else
            noc_async_write_page(i0 + i, ql_dst, scr + i * TB);
#endif
        }
        noc_async_write_barrier();
    }
#if defined(MLA_DSA_SHARD)  // the q rows of x_ag are in place: the all-gather lane may send
    noc_semaphore_inc(get_noc_addr(ag_x, ag_y, get_semaphore(ag_sem)), 1);
    noc_async_atomic_barrier();
#endif
    MK_MARK();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
}
