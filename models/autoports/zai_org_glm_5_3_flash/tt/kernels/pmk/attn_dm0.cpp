// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK attention phase (kernels/pmk/attn_common.hpp) — BRISC (NoC0) of an attention core: per work item (h, qt0, QG
// q tiles) of its item list: the item's QG Q tiles ONCE (CB_AQ), then the K batches with their plan pages, in key order
// [context] ++ [chunk tiles 0 .. qt0 + QG - 1]; a batch's reads are issued together and waited for once (rev 3: rev 2
// read tile by tile with a barrier each, and re-read Q every batch).
//   dense  : context tile j from the paged cache (page = page_table[j / TPB], tile row j % TPB); chunk tile k from the
//            chunk latent store (wait_min(LATB, k + 1): the 32-row block landed, §4b counter rule); no mask tiles (the
//            compute applies the causal triangle on k == qt from CB_AID page 0 and skips k > qt). Rev 7
//            (PMK_ATT_DUAL_NOC): the odd chunk tiles of a batch are read by the NCRISC on NoC1 -- this RISC clears the
//            slot's flag word, pushes a CB_AGO page (slot, k0, nk, flag), reads the even tiles, waits for its barrier
//            and for the flag, then pushes the batch.
//   sparse : the union of the item's 32 QG rows' selected CONTEXT pools (rev 8: per-row bitmaps, sel [1, 1, C, W]; rev
//   7
//            ids, sorted ascending) -> K tiles of 8 pools gathered from the ROW_MAJOR latent store (one 1-KB read
//            per row) into CB_AKRM (the compute tilizes them) with a membership mask tile per (tile, q tile); then
//            the chunk tiles with membership masks (the diagonal tile k == qt also causal + the open pool's tail
//            [p + 1 - (p + 1) % 4, p] of query p). Rev 8 (README §10): the selection comes as per-row BITMAPS in the
//            item's CB_ABM page (fetched + OR-ed into the union by the NCRISC one item ahead; attn_common.hpp
//            fetch_item_bitmaps); K tiles are FIXED 8-pool groups (group g <=> byte g of a row's bitmap), and a mask
//            operand is 32 words -- column 0 of a bf16 tile = the row's membership byte (attn_common.hpp
//            mask_col_row), expanded on the TRISC. Rev 6 wrote 1,024 fp32 elements per mask tile (97 % of a 35-ms
//            unit); rev 7 32 x 16 words from pool words built per id on the NCRISC (56-59 % / 64-67 % of a 12-ms unit:
//            every L1 load / store of a Tensix RISC costs ~8-10 cycles).
// Rev 4: the chunk batches of an item go in the per-core rotated order (attn_common.hpp batch_rot: all 104 readers
// spread over the store's owners -- README §8); the pool gather reads the row-major store one page per row when its
// page is one row (RM_ROWS = 1; the op path's lat_rm) or one page per pool (RM_ROWS = 4); probe: cycles waiting for a
// CB_AK slot (= the compute is slower) vs in the read barriers (= the NoC is slower) vs building masks vs Q reads vs
// waiting for the NCRISC (pool words / the dual-NoC flag).
// Compile-time args (PH_CT0 +): CB_AQ, CB_AK, CB_AKRM, CB_AMASK, CB_AINFO, CB_ASCR, CB_ABM, CB_AID, CT (chunk tiles),
//   PT_BYTES (the page-table row's 64-B rounded bytes), BLOCK (cache rows per page), SEM_QLAT, SEM_LATB, RM_ROWS
//   (rows per page of the row-major latent store: 1 or 4), CB_AGO, FLAG_OFF (byte offset of the slot flags in
//   CB_ASCR), then the TensorAccessorArgs of q_lat, the cache, the page table, the mask bank, the chunk latent
//   store, the row-major latent store, the context L1 store (rev 10 PMK_CTX_L1: the context rows [0, chunk_start)
//   TILE, HEIGHT-sharded over the MM-only cores; the sparse context batches read whole tiles from it over both NoCs
//   instead of gathering 1-KB rows from DRAM and tilizing on the TRISC -- README §12).
// Runtime args (ph_rt0 +): q_lat, cache, page table, mask, chunk latent, lat_rm, ctx store addresses, chunk_start,
//   n_items, core index (the rotation seed), then max_items pairs (h, qt0) (the first n_items valid)
//   [, MK_PROBE_ARG: probe address, slot].
// Defines: PMK_SPARSE, PMK_ATT_NK, PMK_QG, PMK_ATT_KSLOTS, PMK_LAT_L1 (the store is L1-sharded: whole-tile reads),
//   PMK_ATT_DUAL_NOC (the L1-store reads of a batch split over both NoCs: dense chunk tiles; rev 10 also the sparse
//   chunk and context-store tiles), PMK_CTX_L1, MK_PROBE_ARG / MK_PROBE_CB (probe builds).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/attn_common.hpp"

namespace {
constexpr uint32_t CB_AQ = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_AK = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_AKRM = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_AMASK = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_AINFO = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t CB_ASCR = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t CB_ABM = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t CB_AID = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t CT = get_compile_time_arg_val(PH_CT0 + 8);
constexpr uint32_t PT_BYTES = get_compile_time_arg_val(PH_CT0 + 9);
constexpr uint32_t BLOCK = get_compile_time_arg_val(PH_CT0 + 10);
constexpr uint32_t SEM_QLAT = get_compile_time_arg_val(PH_CT0 + 11);
constexpr uint32_t SEM_LATB = get_compile_time_arg_val(PH_CT0 + 12);
constexpr uint32_t RM_ROWS = get_compile_time_arg_val(PH_CT0 + 13);
constexpr uint32_t CB_AGO = get_compile_time_arg_val(PH_CT0 + 14);
constexpr uint32_t FLAG_OFF = get_compile_time_arg_val(PH_CT0 + 15);
#if defined(PMK_LAT_L1)
constexpr bool LAT_WHOLE = true;  // the chunk latent store is L1-sharded: a K tile is one contiguous 32-KB read
#else
constexpr bool LAT_WHOLE = false;
#endif
#if defined(PMK_ATT_DUAL_NOC)
constexpr bool DUAL = true;
#else
constexpr bool DUAL = false;
#endif
#if defined(PMK_CTX_L1)
constexpr bool CTX_L1 = true;  // sparse: the context tiles from the context L1 store (whole tiles, no tilize)
#else
constexpr bool CTX_L1 = false;
#endif
static_assert(RM_ROWS == 1 || RM_ROWS == 4, "the row-major latent store: one row or one pool per page");
constexpr auto q_args = TensorAccessorArgs<PH_CT0 + 16>();
constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
constexpr auto p_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
constexpr auto m_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
constexpr auto l_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
constexpr auto r_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
constexpr auto c_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
constexpr uint32_t DT = pmkatt::DT, NK = pmkatt::NK, QG = pmkatt::QG, TPB = BLOCK / 32, KSLOTS = pmkatt::KSLOTS;
static_assert(KSLOTS >= 2 && KSLOTS <= 8, "CB_AK slots");

#if defined(MK_PROBE_ARG)
// reader cycle counters (probe builds): 0 = waiting for a CB_AK / CB_AKRM slot (the compute holds both), 1 = in the
// read barriers (the NoC / DRAM), 2 = writing mask operands (sparse), 3 = Q reads (DRAM, per item), 4 = waiting for
// the NCRISC (the item's bitmaps in sparse, the dual-NoC flag in dense)
uint32_t g_rc[5] = {0, 0, 0, 0, 0};
FORCE_INLINE uint32_t now32() { return static_cast<uint32_t>(get_timestamp()); }
#define RC_DECL() uint32_t _rt0 = 0
#define RC_BEGIN() _rt0 = now32()
#define RC_END(k) g_rc[k] += now32() - _rt0
#else
#define RC_DECL() \
    do {          \
    } while (0)
#define RC_BEGIN() \
    do {           \
    } while (0)
#define RC_END(k) \
    do {          \
    } while (0)
#endif

FORCE_INLINE void plan(uint32_t w0, uint32_t w1, uint32_t w2) {
    cb_reserve_back(CB_AINFO, 1);
    volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_AINFO));
    info[0] = w0;
    info[1] = w1;
    info[2] = w2;
    invalidate_l1_cache();
    cb_push_back(CB_AINFO, 1);
}
}  // namespace

void kernel_main() {
    const auto qlat = TensorAccessor(q_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto cache = TensorAccessor(k_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto pt = TensorAccessor(p_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto mask = TensorAccessor(m_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const auto lat = TensorAccessor(l_args, get_arg_val<uint32_t>(ph_rt0 + 4));
    const auto latrm = TensorAccessor(r_args, get_arg_val<uint32_t>(ph_rt0 + 5));
    const auto ctxst = TensorAccessor(c_args, get_arg_val<uint32_t>(ph_rt0 + 6));
    const uint32_t chunk_start = get_arg_val<uint32_t>(ph_rt0 + 7);
    const uint32_t n_items = get_arg_val<uint32_t>(ph_rt0 + 8);
    const uint32_t cidx = get_arg_val<uint32_t>(ph_rt0 + 9);  // this core's index in the grid (the rotation seed)
    const uint32_t it0 = ph_rt0 + 10;
    (void)ctxst;
    RC_DECL();
    const uint32_t tb = get_tile_size(CB_AK);
    const uint32_t ct0 = chunk_start / 32;  // context tiles (chunk starts are multiples of 128)
    volatile tt_l1_ptr uint32_t* qlat_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_QLAT));
    volatile tt_l1_ptr uint32_t* latb_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_LATB));

    MK_MARK();
    // scratch (CB_ASCR, tt/pmk_attn.py scratch_bytes): the page-table row at 0 (its 64-B rounded page), the slot flag
    // words at FLAG_OFF (dual-NoC); the triangle / identity / E8 tiles into CB_AID (3 bf16 pages)
    const uint32_t scr = get_write_ptr(CB_ASCR);
    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + FLAG_OFF);
    noc_async_read_page(0, pt, scr);
    cb_reserve_back(CB_AID, 3);
    for (uint32_t t = 0; t < 3; ++t) {  // tile 0: the causal triangle, 1: the identity, 2: the E8 expansion row
        noc_async_read_page(t, mask, get_write_ptr(CB_AID) + t * get_tile_size(CB_AID));
    }
    noc_async_read_barrier();
    cb_push_back(CB_AID, 3);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* prow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    const bool sparse = chunk_start + CT * 32 > pmkatt::DENSE_MAX;
#if !defined(PMK_SPARSE)
    ASSERT(!sparse);
#endif
    uint32_t slot_seq = 0;  // dense dual-NoC: the batch sequence -> its CB_AK slot (KSLOTS ring) and flag

    // the reads of one K tile (16 latent tiles), no barrier: context tile j from the paged cache
    auto read_ctx_tile = [&](uint32_t j, uint32_t l1) {
        const uint32_t tile0 = prow[j / TPB] * (TPB * DT) + (j % TPB) * DT;
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page(tile0 + d, cache, l1 + d * tb);
        }
    };
    // chunk tile k (rows 32 k .. of the chunk) from the chunk latent store, no barrier: its DT pages k * DT .. are
    // one shard = contiguous in the owner's L1 (HEIGHT-sharded [32, 512] per core), so ONE 32-KB read (rev 3: 16
    // page requests per tile; the DRAM form of the store (lat_l1 false) is tile-interleaved: per page)
    auto read_chunk_tile = [&](uint32_t k, uint32_t l1) {
        noc_semaphore_wait_min(latb_sem, k + 1);  // §4b: the block landed (monotonic, reset once per layer)
        if constexpr (LAT_WHOLE) {
            noc_async_read(lat.get_noc_addr(k * DT), l1, DT * tb);
        } else {
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page(k * DT + d, lat, l1 + d * tb);
            }
        }
    };
    // context tile g (rows 32 g .. of the context) from the context L1 store (rev 10, CTX_L1): one 32-KB read
    auto read_ctx_l1_tile = [&](uint32_t g, uint32_t l1) { noc_async_read(ctxst.get_noc_addr(g * DT), l1, DT * tb); };
    // a GO page to the NCRISC (dual-NoC): w0 (the slot, or ITEM_END / GO_END), k0, nk, flag, src
    auto go_page = [&](uint32_t w0, uint32_t k0, uint32_t nk, uint32_t fl, uint32_t src) {
        cb_reserve_back(CB_AGO, 1);
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_AGO));
        go[0] = w0;
        go[1] = k0;
        go[2] = nk;
        go[3] = fl;
        go[4] = src;
        invalidate_l1_cache();
        cb_push_back(CB_AGO, 1);
    };
    // an L1-store batch's tiles k0 .. k0 + nk - 1 (src: the chunk store or the context store) into the reserved slot
    // at l1: alone (rev 4) or split with the NCRISC (rev 7 dual-NoC: even tiles here, odd ones on NoC1 announced by
    // a CB_AGO page; the flag word of the slot is cleared BEFORE the page goes out and set by the NCRISC after ITS
    // barrier)
    auto read_l1_batch = [&](uint32_t src, uint32_t k0, uint32_t nk, uint32_t l1) {
        uint32_t fl = 0;
        if constexpr (DUAL) {
            fl = slot_seq % KSLOTS;
            ++slot_seq;
            flags[fl] = 0;
            go_page(l1, k0, nk, fl, src);
        }
        for (uint32_t i = 0; i < nk; i += DUAL ? 2 : 1) {
            if (src == pmkatt::SRC_CTX) {
                read_ctx_l1_tile(k0 + i, l1 + i * DT * tb);
            } else {
                read_chunk_tile(k0 + i, l1 + i * DT * tb);
            }
        }
        RC_BEGIN();
        noc_async_read_barrier();
        RC_END(1);
        if constexpr (DUAL) {
            RC_BEGIN();
            invalidate_l1_cache();
            while (flags[fl] == 0) {
                invalidate_l1_cache();
            }
            RC_END(4);
        }
    };
    auto push_q = [&](uint32_t h, uint32_t qt0, uint32_t qg) {  // the item's Q tiles, once
        noc_semaphore_wait_min(qlat_sem, qt0 + qg);             // W_uk's 32-row blocks (all heads) are in q_lat
        RC_BEGIN();
        cb_reserve_back(CB_AQ, QG * DT);
        const uint32_t l1 = get_write_ptr(CB_AQ);
        for (uint32_t j = 0; j < qg; ++j) {
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page((h * CT + qt0 + j) * DT + d, qlat, l1 + (j * DT + d) * tb);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_AQ, QG * DT);
        RC_END(3);
    };

#if defined(PMK_SPARSE)
    const uint32_t ctx_pools = ct0 * pmkatt::POOLS_PER_TILE;
    const uint32_t ctx_groups = ct0;                       // fixed 8-pool groups of the context = its tiles
    const uint32_t rw = pmkatt::sel_row_words(ctx_pools);  // words per selection row
    // the mask operand ring (2 NK QG bf16 pages): zeroed ONCE; afterwards only column 0 of each row is written, so the
    // other 30 columns (x the zero rows of E8 on the TRISC) stay finite zeros for the phase
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_AMASK));
        for (uint32_t x = 0; x < 2 * NK * QG * 512; ++x) {
            z[x] = 0;
        }
        invalidate_l1_cache();
    }
    // the item's selection rows (CB_ABM page, fetched by the NCRISC) and their union over the context pools
    const volatile tt_l1_ptr uint8_t* rows = nullptr;
    volatile tt_l1_ptr uint32_t* uni = nullptr;
    // the mask operand of K tile = bitmap byte `gb` (context group g: byte g; chunk tile k: byte ctx_groups + k)
    // against q tile j of the item: column 0 of each row = the row's membership byte (the diagonal: mask_byte_diag)
    auto push_mask = [&](uint32_t gb, uint32_t j, bool diag) {
        cb_reserve_back(CB_AMASK, 1);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_AMASK));
        const volatile tt_l1_ptr uint8_t* rb = rows + (32 * j) * rw * 4 + gb;
        for (uint32_t r = 0; r < 32; ++r) {
            uint32_t b = rb[r * rw * 4];
            if (diag) {
                b = pmkatt::mask_byte_diag(b, r);
            }
            pmkatt::mask_col_row(m, r, b);
        }
        invalidate_l1_cache();
        cb_push_back(CB_AMASK, 1);
    };
    // a masked batch pushes EXACTLY NK x QG mask pages: the real ones (pushed above) then padding pages the compute
    // never reads (it indexes the real ones by popcount and pops NK x QG) -- the 2 NK QG-page ring stays aligned
    auto pad_masks = [&](uint32_t pushed) {
        for (uint32_t z = pushed; z < NK * QG; ++z) {
            cb_reserve_back(CB_AMASK, 1);
            cb_push_back(CB_AMASK, 1);
        }
    };
    // the reads of one K tile of 8 context pools from the ROW_MAJOR latent store into a CB_AKRM slot (32 rows x 1 KB;
    // unused slots zero: finite, masked), no barrier. The store is DRAM-interleaved by PAGE: with one row per page
    // (the op path's lat_rm [max_seq, 512]) a pool's 4 rows are 4 pages in 4 banks, so 4 page reads (rev 3 read 4 KB
    // from the first row's address = rows p, p + 8, p + 16, p + 24 of the bank: the wrong rows); with a pool per
    // page ([max_seq / 4, 2048]) one read
    auto read_pool_tile = [&](const uint32_t* pools, uint32_t l1) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t s = 0; s < pmkatt::POOLS_PER_TILE; ++s) {
            const uint32_t dst = l1 + s * pmkatt::KPOOL * 1024;
            if (pools[s] == pmkatt::NONE_ID) {
                for (uint32_t off = 0; off < pmkatt::KPOOL * 1024; off += MEM_ZEROS_SIZE) {
                    noc_async_read(zeros, dst + off, MEM_ZEROS_SIZE);
                }
            } else if constexpr (RM_ROWS == 1) {
                for (uint32_t r = 0; r < pmkatt::KPOOL; ++r) {
                    noc_async_read_page(pools[s] * pmkatt::KPOOL + r, latrm, dst + r * 1024);
                }
            } else {
                noc_async_read_page(pools[s], latrm, dst);
            }
        }
    };
#endif

    for (uint32_t it = 0; it < n_items; ++it) {
        const uint32_t h = get_arg_val<uint32_t>(it0 + 2 * it), qt0 = get_arg_val<uint32_t>(it0 + 2 * it + 1);
        const uint32_t qg = CT - qt0 < QG ? CT - qt0 : QG;
        const uint32_t nchunk = qt0 + qg;  // chunk tiles 0 .. qt0 + qg - 1
        push_q(h, qt0, qg);
        const uint32_t w1_chunk_base = (qg << 16) | (qt0 << 24);  // w1: k0 | qg << 16 | qt0 << 24
        const uint32_t nb = (nchunk + NK - 1) / NK;               // chunk batches of the item (NK is a power of two)
        const uint32_t rot = pmkatt::batch_rot(cidx, it, nb);
        if (!sparse) {
            // context batches (dense: no masks), then the chunk batches in the rotated order
            for (uint32_t j0 = 0; j0 < ct0; j0 += NK) {
                const uint32_t nk = ct0 - j0 < NK ? ct0 - j0 : NK;
                RC_BEGIN();
                cb_reserve_back(CB_AK, NK * DT);
                RC_END(0);
                const uint32_t l1 = get_write_ptr(CB_AK);
                for (uint32_t i = 0; i < nk; ++i) {
                    read_ctx_tile(j0 + i, l1 + i * DT * tb);
                }
                RC_BEGIN();
                noc_async_read_barrier();
                RC_END(1);
                cb_push_back(CB_AK, NK * DT);
                plan(pmkatt::plan_w0(nk, pmkatt::KIND_CTX, false), w1_chunk_base, 0);
            }
            for (uint32_t bi = 0; bi < nb; ++bi) {
                uint32_t b = bi + rot;
                if (b >= nb) {
                    b -= nb;
                }
                const uint32_t k0 = b * NK;
                const uint32_t nk = nchunk - k0 < NK ? nchunk - k0 : NK;
                RC_BEGIN();
                cb_reserve_back(CB_AK, NK * DT);
                RC_END(0);
                const uint32_t l1 = get_write_ptr(CB_AK);
                read_l1_batch(pmkatt::SRC_CHUNK, k0, nk, l1);
                cb_push_back(CB_AK, NK * DT);
                plan(pmkatt::plan_w0(nk, pmkatt::KIND_CHUNK, false), k0 | w1_chunk_base, bi + 1 == nb ? 1 : 0);
            }
            if constexpr (DUAL) {
                go_page(pmkatt::ITEM_END, 0, 0, 0, 0);
            }
            continue;
        }
#if defined(PMK_SPARSE)
        // the item's selection rows + union: fetched one item ahead by the NCRISC (CB_ABM double buffer)
        RC_BEGIN();
        cb_wait_front(CB_ABM, 1);
        RC_END(4);
        invalidate_l1_cache();
        rows = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(get_read_ptr(CB_ABM));
        uni = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_ABM)) + 32 * qg * rw;
        const volatile tt_l1_ptr uint8_t* unib = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(uni);
        // the union's K tiles = the context groups some row selected into, ascending; batches of NK with QG masks.
        // CTX_L1: whole tiles from the context L1 store into CB_AK (KIND_CTX, no tilize; the batch's tiles need not
        // be consecutive: gidx carries each tile's index, so the dual-NoC page cannot be used -- the reads stay on
        // this NoC unless the batch IS consecutive, which the GO page then describes); else the rev 4..9 gather of
        // 8 pools per tile from the row-major DRAM store into CB_AKRM (the TRISC tilizes)
        uint32_t pools[NK][pmkatt::POOLS_PER_TILE];
        uint32_t gidx[NK];
        uint32_t nk = 0;
        auto flush_ctx = [&]() {
            if (nk == 0) {
                return;
            }
            const uint32_t kcb = CTX_L1 ? CB_AK : CB_AKRM;
            RC_BEGIN();
            cb_reserve_back(kcb, NK * DT);
            RC_END(0);
            const uint32_t l1 = get_write_ptr(kcb);
            bool consecutive = true;
            for (uint32_t i = 1; i < nk; ++i) {
                consecutive = consecutive && gidx[i] == gidx[0] + i;
            }
            if (CTX_L1 && consecutive) {
                read_l1_batch(pmkatt::SRC_CTX, gidx[0], nk, l1);  // (barrier + flag inside)
            } else {
                for (uint32_t i = 0; i < nk; ++i) {
                    if constexpr (CTX_L1) {
                        read_ctx_l1_tile(gidx[i], l1 + i * DT * tb);
                    } else {
                        read_pool_tile(pools[i], l1 + i * DT * tb);
                    }
                }
            }
            uint32_t mbits = 0;
            RC_BEGIN();
            for (uint32_t i = 0; i < nk; ++i) {
                for (uint32_t j = 0; j < qg; ++j) {
                    push_mask(gidx[i], j, false);
                    mbits |= 1u << (i * QG + j);
                }
            }
            pad_masks(nk * qg);
            RC_END(2);
            RC_BEGIN();
            noc_async_read_barrier();
            RC_END(1);
            cb_push_back(kcb, NK * DT);
            plan(pmkatt::plan_w0(nk, pmkatt::KIND_CTX, !CTX_L1), w1_chunk_base, mbits << 1);
            nk = 0;
        };
        for (uint32_t g = 0; g < ctx_groups; ++g) {
            if (unib[g] == 0) {
                continue;
            }
            for (uint32_t s = 0; s < pmkatt::POOLS_PER_TILE; ++s) {
                pools[nk][s] = g * pmkatt::POOLS_PER_TILE + s;
            }
            gidx[nk] = g;
            if (++nk == NK) {
                flush_ctx();
            }
        }
        flush_ctx();
        // the chunk's own tiles 0 .. qt0 + qg - 1 with membership masks per q tile (the diagonal: causal + tail), in
        // the rotated batch order
        for (uint32_t bi = 0; bi < nb; ++bi) {
            uint32_t b = bi + rot;
            if (b >= nb) {
                b -= nb;
            }
            const uint32_t k0 = b * NK;
            const uint32_t bk = nchunk - k0 < NK ? nchunk - k0 : NK;
            RC_BEGIN();
            cb_reserve_back(CB_AK, NK * DT);
            RC_END(0);
            const uint32_t l1 = get_write_ptr(CB_AK);
            uint32_t fl = 0;
            if constexpr (DUAL) {  // the odd tiles on the NCRISC's NoC (the page first: its reads overlap the masks)
                fl = slot_seq % KSLOTS;
                ++slot_seq;
                flags[fl] = 0;
                go_page(l1, k0, bk, fl, pmkatt::SRC_CHUNK);
            }
            for (uint32_t i = 0; i < bk; i += DUAL ? 2 : 1) {
                read_chunk_tile(k0 + i, l1 + i * DT * tb);
            }
            uint32_t mbits = 0;
            RC_BEGIN();
            for (uint32_t i = 0; i < bk; ++i) {
                const uint32_t k = k0 + i;
                for (uint32_t j = 0; j < qg; ++j) {
                    if (k > qt0 + j) {
                        continue;  // the compute skips (k, q tile) pairs past the diagonal
                    }
                    push_mask(ctx_groups + k, j, k == qt0 + j);
                    mbits |= 1u << (i * QG + j);
                }
            }
            if (mbits) {
                pad_masks(__builtin_popcount(mbits));
            }
            RC_END(2);
            RC_BEGIN();
            noc_async_read_barrier();
            RC_END(1);
            if constexpr (DUAL) {
                RC_BEGIN();
                invalidate_l1_cache();
                while (flags[fl] == 0) {
                    invalidate_l1_cache();
                }
                RC_END(4);
            }
            cb_push_back(CB_AK, NK * DT);
            plan(
                pmkatt::plan_w0(bk, pmkatt::KIND_CHUNK, false),
                k0 | w1_chunk_base,
                (bi + 1 == nb ? 1u : 0u) | (mbits << 1));
        }
        cb_pop_front(CB_ABM, 1);  // the item's bitmaps: the NCRISC may fetch item it + 2 into this page
        if constexpr (DUAL) {
            go_page(pmkatt::ITEM_END, 0, 0, 0, 0);
        }
#endif
    }
    if constexpr (DUAL) {
        go_page(pmkatt::GO_END, 0, 0, 0, 0);
    }
    plan(pmkatt::KIND_END, 0, 1);
    MK_MARK();
#if defined(MK_PROBE_ARG)
    // marks 2..6 = the reader's counters (slot waits, barriers, masks, Q, NCRISC waits); the flush scratch = the last
    // 64 B of CB_ASCR
    for (uint32_t k = 0; k < 5; ++k) {
        mkprobe::g_marks[mkprobe::g_n++] = g_rc[k];
    }
    MK_FLUSH(scr + get_tile_size(CB_ASCR) - 64);
#endif
}
