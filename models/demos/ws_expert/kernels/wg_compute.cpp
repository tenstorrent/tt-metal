// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert groups compute (TRISC). A group of GC cores runs whole experts (expert e -> group e % G); core j owns
// gate/up pair j (one 1 x 2 fp32 DST block a row) and down columns [DCOLS j, DCOLS (j + 1)) (blocks of 1 x DB). Its
// weights stream in: a 2-slot ring per part (CB 1 gate/up [KT][2], CB 3 down [KTD][DCOLS]), pushed by the NCRISC when
// the injector's slice of the expert has landed, popped after the expert's last gate/up row / last down row.
// Per row (all rows of the group's experts in order): tilize window (own x blocks within LOOK of the consumption
// point), gate/up over all KT (x ring CB 0, BK tiles a block), silu(g) * u -> one h tile (CB 16), then a DN op: the
// down of the oldest h row once it is DN_LAG rows old (h ring CB 2, all KTD tiles of the row) -> CB 17.
// x blocks of an expert: row r, K block b -> stream block 10 r + b of the expert; block l is core l % GC's.
// RT: 0 stamps, 1 j, 2 g, 3 nexp_g (this group's experts)
// CT: 0 KT, 1 BK, 2 KTD, 3 DCOLS, 4 DB, 5 NX, 6 LOOK, 7 counts L1 (64-aligned; word 0 ready, word 16 + e: M of expert e), 8 G,
//     9 GC
#ifndef DN_LAG
#define DN_LAG 1
#endif
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "glu_fused.hpp"
#ifndef WC_GLU
#define WC_GLU 3
#endif

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;

constexpr uint32_t kt = get_compile_time_arg_val(0);
constexpr uint32_t BK = get_compile_time_arg_val(1);
constexpr uint32_t ktd = get_compile_time_arg_val(2);
constexpr uint32_t dcols = get_compile_time_arg_val(3);
constexpr uint32_t DB = get_compile_time_arg_val(4);
constexpr uint32_t NX = get_compile_time_arg_val(5);
constexpr uint32_t LOOK = get_compile_time_arg_val(6);
constexpr uint32_t counts_l1 = get_compile_time_arg_val(7);
constexpr uint32_t G = get_compile_time_arg_val(8);
constexpr uint32_t GC = get_compile_time_arg_val(9);
constexpr uint32_t gw = 2;  // one gate/up pair
constexpr uint32_t BPR = kt / BK;  // x blocks a row
static_assert(dcols % DB == 0 && kt % BK == 0);
constexpr uint32_t WGU = kt * gw, WDN = ktd * dcols;  // weight tiles of a slot

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

uint32_t t_til = 0, t_xw = 0, t_ww = 0, t_hw = 0;
volatile uint32_t* counts = reinterpret_cast<volatile uint32_t*>(counts_l1);
uint32_t g_, nexp_g;
FORCE_INLINE uint32_t rows_of(uint32_t n) { return (counts[16 + g_ + n * G] + 31) / 32; }

FORCE_INLINE void tilize_batch(DataflowBuffer& xrm_dfb, DataflowBuffer& xst_dfb, uint32_t n) {
    compute_kernel_lib::tilize<BK, xrm_cb, xst_cb>(xrm_dfb, xst_dfb, n);
    reconfig_data_format(w_cb, xr_cb);
    pack_reconfig_data_format(h_cb);
    matmul_block_init(xr_cb, w_cb, false, gw, 1, kt);
}

// down consumption: rows in order; the down weights of expert wd_n are popped once its last row is consumed
uint32_t consumed = 0, wd_n = 0, wd_end = 0;  // (wd_end: rows of experts < wd_n + 1)
FORCE_INLINE void pop_finished_down(uint32_t upto_n) {  // pop experts < upto_n whose rows are all consumed
    while (wd_n < upto_n && consumed >= wd_end) {
        cb_wait_front(wd_cb, WDN);
        cb_pop_front(wd_cb, WDN);
        ++wd_n;
        if (wd_n < nexp_g) {
            wd_end += rows_of(wd_n);
        }
    }
}

FORCE_INLINE void down_row() {
    {
        const uint32_t a = wall();
        cb_wait_front(hr_cb, ktd);
        t_hw += wall() - a;
    }
    {
        const uint32_t a = wall();
        cb_wait_front(wd_cb, WDN);
        t_ww += wall() - a;
    }
    cb_reserve_back(y_cb, dcols);
    for (uint32_t b = 0; b < dcols / DB; ++b) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < ktd; ++k) {
            matmul_block(hr_cb, wd_cb, k, k * dcols + b * DB, 0, false, DB, 1, ktd);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < DB; ++t) {
            pack_tile<true>(t, y_cb, b * DB + t);
        }
        tile_regs_release();
    }
    cb_pop_front(hr_cb, ktd);
    cb_push_back(y_cb, dcols);
    ++consumed;
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    const uint32_t J = get_arg_val<uint32_t>(1);
    g_ = get_arg_val<uint32_t>(2);
    nexp_g = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(xr_cb, w_cb, h_cb);
#if WC_GLU == 0
    expert_activation::init<expert_activation::Kind::Silu>();
#else
    ws_glu::init<WC_GLU - 1>();
#endif
    while (counts[0] == 0) {
    }
    UNPACK(stamps[0] = wall());
    DataflowBuffer xrm_dfb(xrm_cb), xst_dfb(xst_cb);
    uint32_t til_n = 0, til_l = J, til_base = 0, cons = 0;  // next own x block: expert, block, stream base
    auto maybe_tilize = [&]() {
        uint32_t n = til_n, l = til_l, base = til_base, cnt = 0, first = 0;  // (til_l: always one of this core's)
        while (n < nexp_g) {
            const uint32_t nb = rows_of(n) * BPR;
            if (l >= nb) {
                base += nb;
                ++n;
                l = J;
                continue;
            }
            const uint32_t s = base + l;
            if (s >= cons + LOOK) {
                break;
            }
            if (cnt == 0) {
                first = s;
            }
            ++cnt;
            l += GC;
        }
        if (cnt >= 2 || (cnt == 1 && first < cons + NX)) {
            const uint32_t a = wall();
            tilize_batch(xrm_dfb, xst_dfb, cnt);
            t_til += wall() - a;
#ifdef WG_TRACE0
            if (first >= WG_TRACE0 && first < WG_TRACE0 + 40) {
                PACK(stamps[256 + first - WG_TRACE0] = wall());
            }
#endif
            til_n = n;
            til_l = l;
            til_base = base;
        }
    };
    matmul_block_init(xr_cb, w_cb, false, gw, 1, kt);
    uint32_t produced = 0;
    wd_end = nexp_g ? rows_of(0) : 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        if (n == 1) {
            t_til = t_xw = t_ww = t_hw = 0;
        }
        const uint32_t R = rows_of(n);
        {
            const uint32_t a = wall();
            cb_wait_front(w_cb, WGU);
            t_ww += wall() - a;
        }
        const uint32_t e_start = produced;
        for (uint32_t r = 0; r < R; ++r) {
            maybe_tilize();
            tile_regs_acquire();
            for (uint32_t b = 0; b < BPR; ++b) {
                const uint32_t a = wall();
                cb_wait_front(xr_cb, BK);
                const uint32_t bw = wall();
                t_xw += bw - a;
#ifdef WG_TRACE0
                if (cons >= WG_TRACE0 && cons < WG_TRACE0 + 40) {
                    UNPACK(stamps[64 + cons - WG_TRACE0] = bw);
                }
#endif
                for (uint32_t i = 0; i < BK; ++i) {
                    matmul_block(xr_cb, w_cb, i, (b * BK + i) * gw, 0, false, gw, 1, kt);
                }
                cb_pop_front(xr_cb, BK);
                ++cons;
            }
            tile_regs_commit();
            cb_reserve_back(h_cb, 1);
            tile_regs_wait();
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#if WC_GLU == 0
            expert_activation::tile_pair<expert_activation::Kind::Silu>(0);
#else
            ws_glu::tile_pair<WC_GLU - 1>(0);
#endif
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile(0, h_cb);
            tile_regs_release();
            cb_push_back(h_cb, 1);
            ++produced;
            if (consumed < produced && (consumed < e_start || consumed + DN_LAG < produced)) {
                down_row();
                pop_finished_down(n);
            }
        }
        cb_pop_front(w_cb, WGU);  // this expert's gate/up weights are done
        pop_finished_down(n);
        PACK(stamps[1 + n] = wall());
    }
    while (consumed < produced) {
        down_row();
        pop_finished_down(nexp_g);
    }
    pop_finished_down(nexp_g);
    PACK(stamps[63] = wall());
    UNPACK(stamps[40] = t_til);
    UNPACK(stamps[41] = t_xw);
    UNPACK(stamps[42] = t_ww);
    UNPACK(stamps[43] = t_hw);
}
