// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel attention phase (kernels/mla/att_common.hpp) — BRISC (NoC0) of owner core (g, c), per token t: the
// row-tile plan (nk, the index of the tile holding p) into CB_AINFO for the compute, the additive mask tile of p % 32
// (tile p % 32 of the [32 x 32, 32] mask bank: 0, or -1e30 past p) when this core holds p's tile, this
// core's latent row tiles (K = V) from the paged cache (page = page_table[t][jt / (BLOCK / 32)]), then the group's
// q_lat tile row (after the q_lat hand-off when WAIT_COUNT > 0). Inactive rows (cur_pos < 0) attend row 0 (their
// output row is don't-care, finite). Rows past p inside p's tile are read and masked additively: they must hold finite
// values (a NaN / inf would survive the mask) -- the latent cache is zero-initialised (MLA.allocate_cache) and only
// ever written with finite latents.
// Compile-time args: CB_AQ, CB_AK, CB_AMASK, CB_AINFO, CB_ASCR, C (chunk cores per group), HPT, BLOCK, WAIT_SEM,
//   WAIT_COUNT (the q_lat hand-off), LAT_SEM, LAT_COUNT (the latent-cache write hand-off, before the tile of row p),
//   CB_AID (the identity tile = mask-bank tile 32, read once for the compute's diagonal coefficient tiles), then the
//   TensorAccessorArgs of q_lat, the cache, the page table, cur_pos and the mask bank.
// Runtime args: q_lat, cache, page table, cur_pos, mask addresses, g, c, T (+ the token lane with MLA_ATT_LANES).
// MLA_ATT_VERIFY (Stage 17 speculative verify, doc/ttnn/17-mtp-spec-decode/design.md §2.4): the T rows are consecutive
// positions of ONE user (row 0 first, one page-table row repeated): any row tile at or past row 0's position tile may
// hold a latent another row of this launch writes -> the latent-write hand-off is waited for before the first such
// tile (not only before the tile of the token's own row p; with token lanes a lane reads row 0's tile for its token
// without having waited otherwise).
// Token lanes (att_common.hpp LANES): this core's lane takes tokens lane, lane + LANES, ...; c / C are lane-local; the
// row tiles of a token go in batches of <= NKMAX (plan kind KIND_BATCHED, then one plan page per batch).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"

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
    constexpr auto q_args = TensorAccessorArgs<PH_CT0 + 13>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t DT = mlaatt::DT;
    constexpr uint32_t TPB = BLOCK / 32;  // row tiles per cache page
    const auto qlat = TensorAccessor(q_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto cache = TensorAccessor(k_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto pt = TensorAccessor(p_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto cur = TensorAccessor(c_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const auto mask = TensorAccessor(m_args, get_arg_val<uint32_t>(ph_rt0 + 4));
    const uint32_t g = get_arg_val<uint32_t>(ph_rt0 + 5);
    const uint32_t c = get_arg_val<uint32_t>(ph_rt0 + 6);
    const uint32_t t_rows = get_arg_val<uint32_t>(ph_rt0 + 7);
#if defined(MLA_ATT_LANES)
    const uint32_t t0 = get_arg_val<uint32_t>(ph_rt0 + 8);
#else
    const uint32_t t0 = 0;
#endif
    const uint32_t tb = get_tile_size(CB_AK);
    auto plan = [&](uint32_t w0, uint32_t w1, uint32_t w2) {
        cb_reserve_back(CB_AINFO, 1);
        volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_AINFO));
        info[0] = w0;
        info[1] = w1;
        info[2] = w2;
        invalidate_l1_cache();
        cb_push_back(CB_AINFO, 1);
    };

    MK_MARK();
    // scratch: cur_pos row at 0, one page-table row at 256
    const uint32_t scr = get_write_ptr(CB_ASCR);
    noc_async_read_page(0, cur, scr);
    cb_reserve_back(CB_AID, 1);
    noc_async_read_page(32, mask, get_write_ptr(CB_AID));  // the identity tile, kept by the compute for the phase
    noc_async_read_barrier();
    cb_push_back(CB_AID, 1);
    invalidate_l1_cache();
    // the latent-cache write hand-off is waited for only before reading the tile that holds a token's row p (this
    // step's latent row): every other row tile is in the cache already, so those reads start right after mhc_pre
    // (off the W_uk weight stream's DRAM window)
    bool lat_ok = LAT_COUNT == 0;
    bool waited = WAIT_COUNT == 0;
#if defined(MLA_ATT_VERIFY)
    const int32_t vpos0 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[0];
    const uint32_t vt0 = vpos0 < 0 ? 0xFFFFFFFFu : static_cast<uint32_t>(vpos0) / 32;  // row 0's position tile
#define MLA_LAT_WAIT_HERE(i_, jt_) ((i_) == mi || (jt_) >= vt0)
#else
#define MLA_LAT_WAIT_HERE(i_, jt_) ((i_) == mi)
#endif
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[t];
        const uint32_t p = pos < 0 ? 0u : static_cast<uint32_t>(pos);
        const uint32_t nk = mlaatt::chunk_tiles(p, c, C);
        const uint32_t last = p / 32;
        const uint32_t mi = (last % C == c) ? (last - c) / C : 0xFFFFu;
#if defined(MLA_ATT_LANES)
        plan(nk, mi, mlaatt::KIND_BATCHED);
        if (nk == 0) {
            plan(0, 0xFFFFu, 1);  // one empty last batch: no K / q pages
            continue;
        }
        mlaatt::read_pt_prefix(pt, t, scr + 256, p / BLOCK + 2);  // the entries of positions <= p
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* prow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 256);
        for (uint32_t b0 = 0; b0 < nk; b0 += mlaatt::NKMAX) {
            const uint32_t bk = nk - b0 < mlaatt::NKMAX ? nk - b0 : mlaatt::NKMAX;
            const uint32_t bm = (mi != 0xFFFFu && mi >= b0 && mi < b0 + bk) ? mi - b0 : 0xFFFFu;
            plan(bk, bm, b0 + bk == nk ? 1 : 0);
            if (bm != 0xFFFFu) {
                cb_reserve_back(CB_AMASK, 1);
                noc_async_read_page(p % 32, mask, get_write_ptr(CB_AMASK));
                noc_async_read_barrier();
                cb_push_back(CB_AMASK, 1);
            }
            for (uint32_t i = b0; i < b0 + bk; ++i) {
                const uint32_t jt = c + i * C;
                const uint32_t tile0 = prow[jt / TPB] * (TPB * DT) + (jt % TPB) * DT;
                cb_reserve_back(CB_AK, DT);
                const uint32_t l1 = get_write_ptr(CB_AK);
                if (MLA_LAT_WAIT_HERE(i, jt) && !lat_ok) {
                    mk::wait_handoff(LAT_SEM, LAT_COUNT);  // this step's latent row is in the cache
                    lat_ok = true;
                }
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_read_page(tile0 + d, cache, l1 + d * tb);
                }
                noc_async_read_barrier();
                cb_push_back(CB_AK, DT);
            }
            if (bk < mlaatt::NKMAX) {
                cb_reserve_back(CB_AK, (mlaatt::NKMAX - bk) * DT);
                cb_push_back(CB_AK, (mlaatt::NKMAX - bk) * DT);
            }
            if (!waited) {
                mk::wait_handoff(WAIT_SEM, WAIT_COUNT);
                waited = true;
            }
            cb_reserve_back(CB_AQ, DT);  // q per batch: the compute pops it with each partial
            const uint32_t qb = get_write_ptr(CB_AQ);
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page((t * HPT + g) * DT + d, qlat, qb + d * tb);
            }
            noc_async_read_barrier();
            cb_push_back(CB_AQ, DT);
        }
        continue;
#endif
        // the plan for the three compute threads: nk, the index of the tile holding p (0xFFFF: none here)
        cb_reserve_back(CB_AINFO, 1);
        volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_AINFO));
        info[0] = nk;
        info[1] = mi;
        invalidate_l1_cache();
        cb_push_back(CB_AINFO, 1);
        if (nk == 0) {
            continue;
        }
        if (mi != 0xFFFFu) {
            cb_reserve_back(CB_AMASK, 1);
            noc_async_read_page(p % 32, mask, get_write_ptr(CB_AMASK));
            noc_async_read_barrier();
            cb_push_back(CB_AMASK, 1);
        }
        mlaatt::read_pt_prefix(pt, t, scr + 256, p / BLOCK + 2);  // the entries of positions <= p
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 256);
        for (uint32_t i = 0; i < nk; ++i) {
            const uint32_t jt = c + i * C;
            const uint32_t tile0 = row[jt / TPB] * (TPB * DT) + (jt % TPB) * DT;
            cb_reserve_back(CB_AK, DT);
            const uint32_t l1 = get_write_ptr(CB_AK);
            if (MLA_LAT_WAIT_HERE(i, jt) && !lat_ok) {
                mk::wait_handoff(LAT_SEM, LAT_COUNT);  // this step's latent row is in the cache
                lat_ok = true;
            }
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page(tile0 + d, cache, l1 + d * tb);
            }
            noc_async_read_barrier();
            cb_push_back(CB_AK, DT);
        }
        if (nk < mlaatt::NKMAX) {  // pad CB_AK to its capacity: the compute's page-index reads start at the base
            cb_reserve_back(CB_AK, (mlaatt::NKMAX - nk) * DT);
            cb_push_back(CB_AK, (mlaatt::NKMAX - nk) * DT);
        }
        if (!waited) {
            mk::wait_handoff(WAIT_SEM, WAIT_COUNT);
            waited = true;
        }
        cb_reserve_back(CB_AQ, DT);
        const uint32_t q1 = get_write_ptr(CB_AQ);
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page((t * HPT + g) * DT + d, qlat, q1 + d * tb);
        }
        noc_async_read_barrier();
        cb_push_back(CB_AQ, DT);
    }
    if (!waited) {
        mk::wait_handoff(WAIT_SEM, WAIT_COUNT);  // keep the semaphore protocol balanced when every token was empty
    }
    if (!lat_ok) {
        mk::wait_handoff(LAT_SEM, LAT_COUNT);  // (balanced: this core held no token's row p)
    }
    MK_MARK();
#undef MLA_LAT_WAIT_HERE
}
