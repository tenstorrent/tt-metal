// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK attention phase (kernels/pmk/attn_common.hpp) — NCRISC (NoC1) of an attention core: per work item (h, qt0) of
// its item list (the same list as the BRISC's), the compute's o tiles (CB_AOUT, bf16, QG tile rows per item) -> o_lat
// [1, H_r, C, 512] TILE, page (h * CT + qt) * DT + d; after each item, SEM_OL +1 on every consumer core (§4b
// wait_min counters; v1: the W_uv readers wait for >= items).
// Rev 7, two more roles (README §9):
//   sparse: the item's selection BITMAPS (attn_common.hpp fetch_item_bitmaps: its 32 QG rows of sel, one read burst,
//           + the union bitmap by word ORs) into the CB_ABM page the BRISC consumes, PMK_BM_PAGES items ahead (2:
//           the next item's rows land while the BRISC streams this item's batches; 1: the fallback of the
//           large-slot geometries whose two pages do not fit L1 -- serial with the item);
//   dense (PMK_ATT_DUAL_NOC): the ODD chunk tiles of every batch, read on NoC1 into the CB_AK slot the BRISC
//           announces in a CB_AGO page (slot, k0, nk, flag); after the barrier the slot's flag word (CB_ASCR) is set.
// Compile-time args (PH_CT0 +): CB_AOUT, CT, SEM_OL, NCONS, SEM_LATB, CB_AK, CB_AGO, CB_ABM, CB_ASCR, FLAG_OFF,
//   then the TensorAccessorArgs of o_lat, the chunk latent store, the selection bitmaps (sel).
// Runtime args (ph_rt0 +): o_lat, chunk latent, sel addresses, chunk_start, n_items, max_items (the pairs' padded
//   width over the cores), max_items pairs (h, qt0; the first n_items valid), then NCONS consumer (x, y) pairs.
// Defines: PMK_QG, PMK_ATT_NK, PMK_SPARSE, PMK_LAT_L1, PMK_ATT_DUAL_NOC, PMK_BM_PAGES; PMK_PROBE = the probe CB index
//   (the compute's cycle counters are read from it, after its 0xC0DE word, and flushed with this RISC's marks:
//   MK_PROBE_ARG / MK_PROBE_CB as kernels/mk/probe.hpp), then this RISC's own counters (bitmap fetches, dual-NoC
//   reads).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/attn_common.hpp"

namespace {
constexpr uint32_t CB_AOUT = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CT = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t SEM_OL = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t NCONS = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t SEM_LATB = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t CB_AK = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t CB_AGO = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t CB_ABM = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t CB_ASCR = get_compile_time_arg_val(PH_CT0 + 8);
constexpr uint32_t FLAG_OFF = get_compile_time_arg_val(PH_CT0 + 9);
constexpr auto o_args = TensorAccessorArgs<PH_CT0 + 10>();
constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto i_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
constexpr auto c_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
constexpr uint32_t DT = pmkatt::DT, QG = pmkatt::QG, NK = pmkatt::NK;
#if defined(PMK_LAT_L1)
constexpr bool LAT_WHOLE = true;
#else
constexpr bool LAT_WHOLE = false;
#endif
#if defined(PMK_ATT_DUAL_NOC)
constexpr bool DUAL = true;  // rev 10: the sparse chunk / context-store batches too
#else
constexpr bool DUAL = false;
#endif
#if defined(PMK_SPARSE)
constexpr bool BUILD = true;
#else
constexpr bool BUILD = false;
#endif
#if defined(PMK_BM_PAGES)
constexpr uint32_t BMP = PMK_BM_PAGES;
#else
constexpr uint32_t BMP = 2;
#endif
static_assert(BMP == 1 || BMP == 2, "CB_ABM pages");

#if defined(MK_PROBE_ARG)
// this RISC's cycle counters: 0 = bitmap fetches + unions (sparse), 1 = dual-NoC reads (CB_AGO wait + the barrier)
uint32_t g_nc[2] = {0, 0};
FORCE_INLINE uint32_t now32() { return static_cast<uint32_t>(get_timestamp()); }
#define NC_DECL() uint32_t _nt0 = 0
#define NC_BEGIN() _nt0 = now32()
#define NC_END(k) g_nc[k] += now32() - _nt0
#else
#define NC_DECL() \
    do {          \
    } while (0)
#define NC_BEGIN() \
    do {           \
    } while (0)
#define NC_END(k) \
    do {          \
    } while (0)
#endif
}  // namespace

void kernel_main() {
    const auto olat = TensorAccessor(o_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto lat = TensorAccessor(l_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto sel = TensorAccessor(i_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto ctxst = TensorAccessor(c_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const uint32_t chunk_start = get_arg_val<uint32_t>(ph_rt0 + 4);
    const uint32_t n_items = get_arg_val<uint32_t>(ph_rt0 + 5);
    const uint32_t max_items = get_arg_val<uint32_t>(ph_rt0 + 6);
    const uint32_t it0 = ph_rt0 + 7;
    (void)ctxst;
    const uint32_t cons0 = it0 + 2 * max_items;
    const uint32_t tb = get_tile_size(CB_AOUT);
    NC_DECL();
    const uint32_t scr = get_write_ptr(CB_ASCR);
    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + FLAG_OFF);
    volatile tt_l1_ptr uint32_t* latb_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_LATB));
    const uint32_t ct0 = chunk_start / 32;
    const uint32_t ctx_pools = ct0 * pmkatt::POOLS_PER_TILE;
    (void)lat;
    (void)sel;
    (void)latb_sem;
    (void)ctx_pools;
    (void)tb;

    MK_MARK();
    auto item_qg = [&](uint32_t qt0) { return CT - qt0 < QG ? CT - qt0 : QG; };
    // the selection rows + union of item `it` (sparse) into the next CB_ABM page
    auto build = [&](uint32_t it) {
        if constexpr (BUILD) {
            if (it >= n_items) {
                return;
            }
            const uint32_t qt0 = get_arg_val<uint32_t>(it0 + 2 * it + 1);
            NC_BEGIN();
            cb_reserve_back(CB_ABM, 1);
            pmkatt::fetch_item_bitmaps(get_write_ptr(CB_ABM), sel, qt0, item_qg(qt0), ctx_pools);
            cb_push_back(CB_ABM, 1);
            NC_END(0);
        } else {
            (void)it;
        }
    };
    // one CB_AGO page (dual-NoC): the odd tiles of the batch on NoC1 (from the chunk store or the context store),
    // then the slot's flag; returns false on the item's ITEM_END page
    auto serve_go = [&]() -> bool {
        if constexpr (DUAL) {
            NC_BEGIN();
            cb_wait_front(CB_AGO, 1);
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_AGO));
            const uint32_t l1 = go[0], k0 = go[1], nk = go[2], fl = go[3], src = go[4];
            ASSERT(l1 != pmkatt::GO_END);
            if (l1 == pmkatt::ITEM_END) {
                cb_pop_front(CB_AGO, 1);
                NC_END(1);
                return false;
            }
            const uint32_t tk = get_tile_size(CB_AK);
            for (uint32_t i = 1; i < nk; i += 2) {
                const uint32_t k = k0 + i;
                if (src == pmkatt::SRC_CTX) {
                    noc_async_read(ctxst.get_noc_addr(k * DT), l1 + i * DT * tk, DT * tk);
                } else {
                    noc_semaphore_wait_min(latb_sem, k + 1);
                    if constexpr (LAT_WHOLE) {
                        noc_async_read(lat.get_noc_addr(k * DT), l1 + i * DT * tk, DT * tk);
                    } else {
                        for (uint32_t d = 0; d < DT; ++d) {
                            noc_async_read_page(k * DT + d, lat, l1 + (i * DT + d) * tk);
                        }
                    }
                }
            }
            noc_async_read_barrier();
            flags[fl] = 1;  // after the data landed (the BRISC cleared it before the page)
            cb_pop_front(CB_AGO, 1);
            NC_END(1);
        }
        return true;
    };
    // the item's o tiles -> o_lat, then the consumers' counter
    auto write_o = [&](uint32_t it) {
        const uint32_t h = get_arg_val<uint32_t>(it0 + 2 * it), qt0 = get_arg_val<uint32_t>(it0 + 2 * it + 1);
        const uint32_t qg = item_qg(qt0);
        for (uint32_t j = 0; j < qg; ++j) {
            for (uint32_t d = 0; d < DT; ++d) {
                cb_wait_front(CB_AOUT, d + 1);
                noc_async_write_page((h * CT + qt0 + j) * DT + d, olat, get_read_ptr(CB_AOUT) + d * tb);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_AOUT, DT);
        }
        if constexpr (NCONS > 0) {
            mk::signal_handoff(SEM_OL, cons0, NCONS);  // +1 per item on every W_uv consumer (monotonic)
        }
    };

    build(0);
    if constexpr (BMP >= 2) {
        build(1);
    }
    for (uint32_t it = 0; it < n_items; ++it) {
        if constexpr (DUAL) {  // the item's L1-store batches, up to its ITEM_END page
            while (serve_go()) {
            }
        }
        if constexpr (BMP == 1) {
            build(it + 1);  // blocks until the BRISC released item it's page; overlaps the compute's tail of item it
        }
        write_o(it);
        if constexpr (BMP >= 2) {
            build(it + 2);
        }
    }
    if constexpr (DUAL) {  // the terminator page (the CB must be empty at the phase boundary)
        cb_wait_front(CB_AGO, 1);
        invalidate_l1_cache();
        ASSERT(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_AGO))[0] == pmkatt::GO_END);
        cb_pop_front(CB_AGO, 1);
    }
    MK_MARK();
#if defined(PMK_PROBE) && defined(MK_PROBE_ARG)
    // the compute's cycle counters (PROBE CB words 0..3, ready at word 4 == 0xC0DE) -> marks 2..5 of this row, then
    // this RISC's two counters -> marks 6..7
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(PMK_PROBE));
    while (p[4] != 0xC0DEu) {
        invalidate_l1_cache();
    }
    for (uint32_t k = 0; k < 4; ++k) {
        mkprobe::g_marks[mkprobe::g_n++] = p[k];
    }
    for (uint32_t k = 0; k < 2; ++k) {
        mkprobe::g_marks[mkprobe::g_n++] = g_nc[k];
    }
    MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
}
