// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel attention phase with DSA (kernels/mla/dsa_common.hpp) — BRISC (NoC0) of owner core (g, c). Dense
// tokens (p + 1 <= 2051): exactly kernels/mla/att_dm0.cpp. Sparse tokens: the score plan + inputs (q_idx tile row,
// wsc, this core's pooled-key tiles), then — once the NCRISC published this core's selected pools in CB_ASEL — the
// selected pools' latent rows gathered from the paged cache into K tiles (8 pools per tile, 4 rows each; 128-B
// face-row runs, DRAM reads 64-B aligned) in batches of <= 3 tiles, each batch with its plan (nk, the partial tile,
// last) and the q_lat tile row. Unused rows of a K tile are zero (finite: the masked P = 0 times them).
// CB_AINFO words: dense / batch: (nk, mi, last), score plan: (npt, 0, 0x5C) (word 2 = the sparse marker).
// Compile-time args: att_dm0.cpp's 13 (.. CB_AID), then CB_AQI, CB_AWS, CB_APK, CB_ASEL, IXR_SEM, QI_SEM, QI_COUNT,
// TP_TILES, then
//   the TensorAccessorArgs of q_lat, the cache, the page table, cur_pos, the mask bank, q_idx, wsc, the pooled keys.
// Runtime args: q_lat, cache, page table, cur_pos, mask, g, c, T, q_idx, wsc, pooled-key addresses (+ the token lane
// with
//   MLA_ATT_LANES, rev 40: this core's lane takes tokens lane, lane + LANES, ...; c / C are lane-local; a dense token's
//   row tiles go in batches of <= NKMAX as kernels/mla/att_dm0.cpp's lanes path).
// MLA_DSA_ONE_SLOT (rev 56, Stage 17's speculative verify): every row scores the launch's slot (s0),
// kernels/mla/ix_dm1.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"

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
    constexpr uint32_t DT = mlaatt::DT;
    constexpr uint32_t TPB = BLOCK / 32;  // row tiles per cache page
#if defined(MLA_DSA_RUNS)
    static_assert(
        BLOCK % 32 == 0 && mladsa::KPOOL == 4, "rev 48 runs: 4 pools per 16-row face group, whole tiles per page");
#endif
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
#if defined(MLA_ATT_LANES)
    const uint32_t t0 = get_arg_val<uint32_t>(ph_rt0 + 11);
#else
    const uint32_t t0 = 0;
#endif
    const uint32_t tb = get_tile_size(CB_AK);
    // the S1 gather's 128-B face-row reads keep source and destination congruent mod 64 (mla_common.hpp): CB_AK's pages
    // must start 64-B aligned (watcher builds check it)
    ASSERT((get_write_ptr(CB_AK) & 63) == 0 && (tb & 63) == 0);

    MK_MARK();
    const uint32_t scr = get_write_ptr(CB_ASCR);  // cur_pos row at 0, one page-table row at 256
    noc_async_read_page(0, cur, scr);
    cb_reserve_back(CB_AID, 1);
    noc_async_read_page(32, mask, get_write_ptr(CB_AID));  // the identity tile, kept by the compute for the phase
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
    auto push_q = [&](uint32_t t) {
        cb_reserve_back(CB_AQ, DT);
        const uint32_t q1 = get_write_ptr(CB_AQ);
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page((t * HPT + g) * DT + d, qlat, q1 + d * tb);
        }
        noc_async_read_barrier();
        cb_push_back(CB_AQ, DT);
    };
    auto pad_k = [&](uint32_t nk) {  // CB_AK to its capacity: the compute's page-index reads start at the base
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
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        const int32_t pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[t];
        const uint32_t p = pos < 0 ? 0u : static_cast<uint32_t>(pos);
        mlaatt::read_pt_prefix(pt, t, scr + 256, p / BLOCK + 2);  // the entries of positions <= p
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 256);
        if (!mladsa::sparse(p)) {  // D1: kernels/mla/att_dm0.cpp
            const uint32_t nk = mlaatt::chunk_tiles(p, c, C);
            const uint32_t last = p / 32;
            const uint32_t mi = (last % C == c) ? (last - c) / C : 0xFFFFu;
#if defined(MLA_ATT_LANES)
            // a lane's C cores hold more row tiles each: batches of <= NKMAX (att_dm0.cpp's lanes plan)
            info3(CB_AINFO, nk, mi, mlaatt::KIND_BATCHED);
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
                wait_q();
                push_q(t);
            }
            continue;
#endif
            info3(CB_AINFO, nk, mi, 1);
            if (nk == 0) {
                continue;
            }
            if (mi != 0xFFFFu) {
                push_mask(p % 32);
            }
            for (uint32_t i = 0; i < nk; ++i) {
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
            pad_k(nk);
            wait_q();
            push_q(t);
            continue;
        }
        // ---- sparse: I4 inputs
        const uint32_t n_closed = (p + 1) / mladsa::KPOOL;
        const uint32_t npt = mladsa::pool_tiles(n_closed, c, C);
        info3(CB_AINFO, npt, 0, 0x5C);
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
        for (uint32_t i0 = 0; i0 < npt;
             i0 += mladsa::SB) {  // batches of SB pooled-key tiles (rev 37), one barrier each
            const uint32_t nb = npt - i0 < mladsa::SB ? npt - i0 : mladsa::SB;
            cb_reserve_back(
                CB_APK, 4 * mladsa::SB);  // rev 41: a full batch slot (the last one padded), att_c.cpp score
            const uint32_t l1 = get_write_ptr(CB_APK);
            for (uint32_t j = 0; j < nb; ++j) {
#if defined(MLA_DSA_ONE_SLOT)
                const uint32_t tile0 = (c + (i0 + j) * C) * 4;
#else
                const uint32_t tile0 = (t * TP_TILES + c + (i0 + j) * C) * 4;
#endif
                for (uint32_t d = 0; d < 4; ++d) {
                    noc_async_read_page(tile0 + d, pools, l1 + (4 * j + d) * tb);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_APK, 4 * mladsa::SB);
        }
        // ---- S1: this core's selected pools (+ the tail on core 0) -> K tiles, batches of <= NKMAX
        cb_wait_front(CB_ASEL, 1);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_ASEL));
        const uint32_t n_sel = sel[0], tail_rows = sel[1], tail_pool = sel[2];
        const uint32_t entries = n_sel + (tail_rows > 0 ? 1 : 0);
        const uint32_t ntiles = (entries + mladsa::POOLS_PER_TILE - 1) / mladsa::POOLS_PER_TILE;
        if (ntiles == 0) {
            info3(CB_AINFO, 0, 0xFFFFu, 1);
        }
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t b0 = 0; b0 < ntiles; b0 += mlaatt::NKMAX) {
            const uint32_t nk = ntiles - b0 < mlaatt::NKMAX ? ntiles - b0 : mlaatt::NKMAX;
            const bool last_batch = b0 + nk == ntiles;
            const uint32_t e_end = last_batch ? entries : (b0 + nk) * mladsa::POOLS_PER_TILE;
            const uint32_t e_last0 = (b0 + nk - 1) * mladsa::POOLS_PER_TILE;  // first entry of the batch's last tile
            const bool tail_here = last_batch && tail_rows > 0;
            const uint32_t rows_last = (e_end - e_last0) * mladsa::KPOOL - (tail_here ? mladsa::KPOOL - tail_rows : 0);
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
                    // unused rows must read as finite zeros (0 x inf = NaN in the PV product)
                    for (uint32_t off = 0; off < DT * tb; off += MEM_ZEROS_SIZE) {
                        noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
                    }
                    noc_async_read_barrier();
                }
                for (uint32_t e = e0; e < e1;) {
                    const uint32_t q = e < n_sel ? sel[3 + e] : tail_pool;
                    const uint32_t tok = q * mladsa::KPOOL;  // first row of the pool
                    const uint32_t in_page = tok % BLOCK;
                    const uint32_t tile0 = row[tok / BLOCK] * (TPB * DT) + (in_page / 32) * DT;
                    const uint32_t r0 = in_page % 32, s = e - e0;
                    const uint32_t src_off = (r0 / 16) * 1024 + (r0 % 16) * 32;
                    const uint32_t dst_off = ((s * mladsa::KPOOL) / 16) * 1024 + ((s * mladsa::KPOOL) % 16) * 32;
                    uint32_t run = 1;  // (rev 53: was `k`, which shadowed the K-tile loop's)
#if defined(MLA_DSA_RUNS)
                    // rev 48: a run of consecutive selected pools inside one 16-row face group of the source tile AND
                    // of the K tile being built (4 pools each) -> one read of 128 k B per face (was k reads of 128 B:
                    // the gather is read-count bound, ~80 ns per read; card0 u45). The tail pool never joins a run.
                    if (e < n_sel) {
                        while (e + run < e1 && e + run < n_sel && sel[3 + e + run] == q + run && (q % 4) + run < 4 &&
                               (s % 4) + run < 4) {
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
            wait_q();
            push_q(t);
        }
        cb_pop_front(CB_ASEL, 1);
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
