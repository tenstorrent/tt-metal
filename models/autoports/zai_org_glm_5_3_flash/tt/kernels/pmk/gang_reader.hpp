// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_xffn_gang_reader.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
// W8 unit 2 (Stage 14, coordinator 2026-10-25 16:48:26Z): the EP expert FFN on GANGS (GLM53_PF_MOE_XFFN_GANG,
// w8/combine_fusion_design.md) -- reader (BRISC). A gang of G cores runs each unit (local expert e, M group of <= MG
// tile rows) together; core s of the gang holds W slice s of the expert's full width (W_T tiles = W_full / G): gate|up
// shards s W_T .. s W_T + W_T - 1 (each = the (gate_j, up_j) tile pair of every K row, contiguous in one bank:
// XFFN_SHARD_BASES) and down rows s W_T .. s W_T + W_T - 1 of every column shard. The gang's partial y blocks are
// ring-reduce-scattered in L1 by the writer / compute (moe_xffn_gang_writer.cpp): the down weight chunks stream in the
// core's RING ORDER (step q: N chunk ((s - 1 - q / 2) mod G) + G (q mod 2)), so each hop's own partial is computed when
// the travelling block arrives. Units are dealt over the NG gangs (SNAKE by cost class, as moe_xffn_reader.cpp's
// XFFN_SNAKE over cores); every core of a gang computes the same list.
// Compile-time: as moe_xffn_reader.cpp (H_T, W_T = the SLICE, KC, NC, MG, E, EP, P, WTB, the accessors of the FULL
// expert tensors). Defines: XFFN_GANG = G, XFFN_GANG_NG = NG. Runtime: as moe_xffn_reader.cpp + gang id (8), slice s
// (9).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/prefill/moe_xffn_probe.hpp"

using namespace pmk_moe;
using namespace xffn_probe_slot;

#if !defined(XFFN_GANG) || !defined(XFFN_GANG_NG) || !defined(XFFN_SHARD_BASES) || !defined(XFFN_X_ON_NOC1)
#error "the gang reader needs XFFN_GANG, XFFN_GANG_NG, XFFN_SHARD_BASES and XFFN_X_ON_NOC1"
#endif

FORCE_INLINE void phase_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t W_T = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t KC = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t NC = get_compile_time_arg_val(CT0 + 3);
    constexpr uint32_t MG = get_compile_time_arg_val(CT0 + 4);
    constexpr uint32_t E = get_compile_time_arg_val(CT0 + 5);
    constexpr uint32_t EP = get_compile_time_arg_val(CT0 + 6);
    constexpr uint32_t WTB = get_compile_time_arg_val(CT0 + 8);
    constexpr auto gu_args = TensorAccessorArgs<CT0 + 9>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr uint32_t G = XFFN_GANG;
    constexpr uint32_t NG = XFFN_GANG_NG;
    constexpr uint32_t GU_T = 2 * W_T;  // gate|up tiles per K row of the slice
    constexpr uint32_t TBL_W = (2 * E + 15) / 16 * 16;
    static_assert(H_T % KC == 0 && H_T / NC == 2 * G && EP % 16 == 0, "gang chunking: H / NC = 2 G N chunks");

    const uint32_t cnt_addr = get_arg_val<uint32_t>(RT0 + 2);
    const uint32_t start_addr = get_arg_val<uint32_t>(RT0 + 3);
    const uint32_t tbl_addr = get_arg_val<uint32_t>(RT0 + 4);
    const uint32_t i = get_arg_val<uint32_t>(RT0 + 5);
    const uint32_t gang = get_arg_val<uint32_t>(RT0 + 8);
    const uint32_t s = get_arg_val<uint32_t>(RT0 + 9);
    XP_BEGIN();

    // tables: counts | starts | addresses | (the reader gathers no x: the writer does, XFFN_X_ON_NOC1)
    const uint32_t tab = get_write_ptr(CB_TAB);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    volatile tt_l1_ptr uint32_t* start = cnt + EP;
    volatile tt_l1_ptr uint32_t* tbl = start + EP;
    volatile tt_l1_ptr uint32_t* dl = tbl + TBL_W;
    uint64_t* gu_base = reinterpret_cast<uint64_t*>(reinterpret_cast<uint32_t>(dl + MG * 32 + 15) & ~63u);
    uint64_t* dn_base = gu_base + W_T;
    {
        XP_TIC();
        const InterleavedAddrGen<true> cnt_g{.bank_base_address = cnt_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> tbl_g{.bank_base_address = tbl_addr, .page_size = 2 * E * 4};
        noc_async_read(cnt_g.get_noc_addr(0), reinterpret_cast<uint32_t>(cnt), EP * 4);
        noc_async_read(start_g.get_noc_addr(0), reinterpret_cast<uint32_t>(start), EP * 4);
        noc_async_read(tbl_g.get_noc_addr(0), reinterpret_cast<uint32_t>(tbl), (2 * E * 4 + 63) / 64 * 64);
        noc_async_read_barrier();
        XP_TOC(R_TABLES);
    }

    auto unit = [&](uint32_t e, uint32_t gi) {
        const uint32_t m = cnt[e];
        const uint32_t mt_all = (m + 31) / 32;
        const uint32_t g0 = gi * MG;
        const uint32_t mt = mt_all - g0 < MG ? mt_all - g0 : MG;
        const uint32_t rows = m - g0 * 32 < mt * 32 ? m - g0 * 32 : mt * 32;
        const uint32_t d0 = start[e] + g0 * 32;
        const auto gu = TensorAccessor(gu_args, tbl[2 * e], WTB);
        const auto dn = TensorAccessor(dn_args, tbl[2 * e + 1], WTB);
        for (uint32_t j = 0; j < W_T; ++j) {  // this core's slice: shards s W_T + j of the full gate|up
            gu_base[j] = gu.get_noc_addr(2 * (s * W_T + j));
        }
        for (uint32_t sh = 0; sh < H_T / 2; ++sh) {
            dn_base[sh] = dn.get_noc_addr(2 * sh);
        }
        XP_ADD(R_UNITS, 1);
        cb_reserve_back(CB_CTL, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CTL))[0] = mt;
        cb_push_back(CB_CTL, 1);
        cb_reserve_back(CB_WCTL, 1);
        {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WCTL));
            w[0] = mt;
            w[1] = rows;
            w[2] = d0;
            w[3] = e;
        }
        cb_push_back(CB_WCTL, 1);

        for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
            XP_TIC();
            cb_reserve_back(CB_WGU, KC * GU_T);
            XP_TOC(R_GU_WAIT);
            XP_TIC();
            const uint32_t wp = get_write_ptr(CB_WGU);
            for (uint32_t k = 0; k < KC; ++k) {
                const uint32_t off = (kc * KC + k) * 2 * WTB;
                for (uint32_t j = 0; j < W_T; ++j) {
                    noc_async_read(gu_base[j] + off, wp + (k * GU_T + 2 * j) * WTB, 2 * WTB);
                }
            }
            noc_async_read_barrier();
            XP_TOC(R_GU_READ);
            XP_ADD(R_KB, KC * GU_T * WTB / 1024);
            cb_push_back(CB_WGU, KC * GU_T);
        }
        // down chunks in the ring order of slice s (moe_xffn_gang_writer.cpp)
        for (uint32_t q = 0; q < 2 * G; ++q) {
            const uint32_t nc = ((s + G - 1 - (q >> 1)) % G) + G * (q & 1);
            XP_TIC();
            cb_reserve_back(CB_WD, W_T * NC);
            XP_TOC(R_DN_WAIT);
            XP_TIC();
            const uint32_t wp = get_write_ptr(CB_WD);
            for (uint32_t k = 0; k < W_T; ++k) {
                const uint32_t row = (s * W_T + k) * 2 * WTB;  // the slice's down row in the full-height shard
                for (uint32_t jj = 0; jj < NC; jj += 2) {
                    noc_async_read(dn_base[(nc * NC + jj) / 2] + row, wp + (k * NC + jj) * WTB, 2 * WTB);
                }
            }
            noc_async_read_barrier();
            XP_TOC(R_DN_READ);
            XP_ADD(R_KB, W_T * NC * WTB / 1024);
            cb_push_back(CB_WD, W_T * NC);
        }
    };

    // the gang dealing: cost classes mt = MG (every full group), then MG - 1 .. 1 (each expert's last group); unit v
    // of the dealing order goes to gang v mod NG on even rounds (v / NG even), NG - 1 - v mod NG on odd ones
    uint32_t v = 0;
    for (uint32_t cls = MG; cls >= 1; --cls) {
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t m = cnt[e];
            if (m == 0) {
                continue;
            }
            const uint32_t mt_all = (m + 31) / 32;
            const uint32_t ng = (mt_all + MG - 1) / MG;
            const uint32_t last_mt = mt_all - (ng - 1) * MG;
            const uint32_t n_full = cls == MG ? ng - 1 : 0;
            const uint32_t n_cls = n_full + (last_mt == cls ? 1 : 0);
            for (uint32_t j = 0; j < n_cls; ++j, ++v) {
                const uint32_t to = ((v / NG) & 1) ? NG - 1 - v % NG : v % NG;
                if (to == gang) {
                    unit(e, j < n_full ? j : ng - 1);
                }
            }
        }
    }
    cb_reserve_back(CB_CTL, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CTL))[0] = 0;
    cb_push_back(CB_CTL, 1);
    cb_reserve_back(CB_WCTL, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WCTL))[0] = 0;
    cb_push_back(CB_WCTL, 1);
    XP_FLUSH((tab + 63) & ~63u, get_arg_val<uint32_t>(RT0 + 7), i);
}
