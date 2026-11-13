// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert dynamic-M compute (TRISC): wx_compute.cpp with the per-expert program read from the plan ring the BRISC
// fills (ws_plan.hpp: computed on device from each expert's M), and posts its done word (expert + 1) after each expert
// so the slot can be reused. x tilize: before every gate/up piece it tilizes this core's own x blocks whose stream
// index is within LOOK blocks of its consumption point (in this expert and the next), as one
// batch when there are >= 2 or one is within NX (its ring slot may already be free): staging stays bounded for any M,
// and every block is tilized >= NX - (a piece's blocks) ahead of its use.
// RT: 0 stamps. CT: 0 KT, 1 NPAIR, 2 BK, 3 NEXP, 4 KTD, 5 DCOLS, 6 plan buffer addr, 7 NX, 8 LOOK
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "glu_fused.hpp"
#include "ws_plan.hpp"
#ifndef WC_GLU
#define WC_GLU 3
#endif

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t pout_cb = tt::CBIndex::c_4;
constexpr uint32_t pinl_cb = tt::CBIndex::c_5;
constexpr uint32_t pinr_cb = tt::CBIndex::c_6;
constexpr uint32_t xrm_cb = tt::CBIndex::c_7;  // row-major x block (from DRAM)
constexpr uint32_t xst_cb = tt::CBIndex::c_8;  // tilized x block (to the column multicast)
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;

constexpr uint32_t kt = get_compile_time_arg_val(0);
constexpr uint32_t npair = get_compile_time_arg_val(1);
constexpr uint32_t BK = get_compile_time_arg_val(2);
constexpr uint32_t nexp = get_compile_time_arg_val(3);
constexpr uint32_t ktd = get_compile_time_arg_val(4);
constexpr uint32_t dcols = get_compile_time_arg_val(5);
constexpr uint32_t planbuf = get_compile_time_arg_val(6);
constexpr uint32_t NX = get_compile_time_arg_val(7);
constexpr uint32_t LOOK = get_compile_time_arg_val(8);
constexpr uint32_t gw = 2 * npair;
constexpr uint32_t DB = 4;
static_assert(gw == DB && dcols % DB == 0);
constexpr uint32_t F_PRE = 3, F_SEND = 12, F_DN = 16;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

bool h_fmt = true;  // packer set for bf16 h / y (else the fp32 partial)
uint32_t t_til = 0, t_xw = 0, t_dn = 0;

// tilize n own blocks (the BRISC's row-major blocks in CB 7 -> CB 8), then back to the matmul configuration
FORCE_INLINE void tilize_batch(DataflowBuffer& xrm_dfb, DataflowBuffer& xst_dfb, uint32_t n) {
    compute_kernel_lib::tilize<BK, xrm_cb, xst_cb>(xrm_dfb, xst_dfb, n);
    reconfig_data_format(w_cb, xr_cb);  // srcA = in1 (weights), srcB = in0 (x / h)
    pack_reconfig_data_format(h_cb);
    h_fmt = true;
    matmul_block_init(xr_cb, w_cb, false, gw, 1, kt);
}

// gate/up's in0 is x (CB 0), down's is h (CB 2): when their formats differ (bfp8 x), SrcB switches with them
#ifdef WX_X_FMT_DIFFERS
#define SRCB_TO_H() reconfig_data_format_srcb(xr_cb, hr_cb)
#define SRCB_TO_X() reconfig_data_format_srcb(hr_cb, xr_cb)
#else
#define SRCB_TO_H()
#define SRCB_TO_X()
#endif

FORCE_INLINE void down_row() {
    SRCB_TO_H();
    cb_wait_front(hr_cb, ktd);
    cb_reserve_back(y_cb, dcols);
    for (uint32_t b = 0; b < dcols / DB; ++b) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < ktd; ++k) {
            matmul_block(hr_cb, wd_cb, k, k * dcols + b * DB, 0, false, DB, 1, ktd);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (!h_fmt) {
            pack_reconfig_data_format(pout_cb, y_cb);
            h_fmt = true;
        }
        for (uint32_t t = 0; t < DB; ++t) {
            pack_tile<true>(t, y_cb, b * DB + t);
        }
        tile_regs_release();
    }
    cb_pop_front(hr_cb, ktd);
    cb_push_back(y_cb, dcols);
    SRCB_TO_X();
}

FORCE_INLINE volatile uint32_t* plan_peek(uint32_t e) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(planbuf) + (e % ws_plan::NSLOT) * ws_plan::SLOT_WORDS;
    return p[ws_plan::H_READY] == e + 1 ? p : nullptr;
}

FORCE_INLINE volatile uint32_t* plan_of(uint32_t e) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(planbuf) + (e % ws_plan::NSLOT) * ws_plan::SLOT_WORDS;
    while (p[ws_plan::H_READY] != e + 1) {
    }
    return p;
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    volatile uint32_t* ctrl = reinterpret_cast<volatile uint32_t*>(planbuf) + ws_plan::CTRL;
    compute_kernel_hw_startup<SrcOrder::Reverse>(xr_cb, w_cb, h_cb);
#if WC_GLU == 0
    expert_activation::init<expert_activation::Kind::Silu>();
#else
    ws_glu::init<WC_GLU - 1>();
#endif
    cb_wait_front(w_cb, kt * gw);
    cb_wait_front(wd_cb, ktd * dcols);
    UNPACK(stamps[0] = wall());
    DataflowBuffer xrm_dfb(xrm_cb), xst_dfb(xst_cb);
    // x tilize window: next own block to tilize (expert, own index, cluster blocks before that expert); x consumed
    uint32_t til_e = 0, til_i = 0, til_base = 0, cons = 0;
    auto maybe_tilize = [&](uint32_t cur_e) {
        uint32_t e = til_e, i = til_i, base = til_base, n = 0, first = 0;
        // (blocking reads only: the three TRISCs must take the same decisions - a non-blocking peek at a plan the BRISC
        // is still writing let them diverge (UNPACK in a tilize batch, MATH in a matmul: hang). Plan e + 1 always
        // arrives: the plan ring only waits for expert e - 1, which this core finished.)
        while (e < nexp && e <= cur_e + 1) {
            volatile uint32_t* p = plan_of(e);
            if (i >= p[ws_plan::H_NMY]) {
                base += p[ws_plan::H_NBE];
                ++e;
                i = 0;
                continue;
            }
            const uint32_t sidx = base + p[ws_plan::OFF_MY + 4 * i];
            if (sidx >= cons + LOOK) {
                break;
            }
            if (n == 0) {
                first = sidx;
            }
            ++n;
            ++i;
        }
        if (n >= 2 || (n == 1 && first < cons + NX)) {
            const uint32_t a = wall();
            tilize_batch(xrm_dfb, xst_dfb, n);
            t_til += wall() - a;
            til_e = e;
            til_i = i;
            til_base = base;
        }
    };
    matmul_block_init(xr_cb, w_cb, false, gw, 1, kt);
    uint32_t produced = 0, consumed = 0;
    for (uint32_t e = 0; e < nexp; ++e) {
        if (e == 2) {
            t_til = t_xw = t_dn = 0;
        }
        volatile uint32_t* pl = plan_of(e);
        const uint32_t e_start = produced;
        const uint32_t nop = pl[ws_plan::H_NOP];
        for (uint32_t p = 0; p < nop; ++p) {
            const uint32_t k0 = pl[ws_plan::OFF_OP + 3 * p];
            const uint32_t k1 = pl[ws_plan::OFF_OP + 3 * p + 1];
            const uint32_t flags = pl[ws_plan::OFF_OP + 3 * p + 2];
            if (flags & F_DN) {
                if (consumed < produced && (!DN_LAG || consumed < e_start || consumed + DN_LAG < produced)) {
                    const uint32_t a = wall();
                    down_row();
                    t_dn += wall() - a;
                    ++consumed;
                }
                continue;
            }
            maybe_tilize(e);
            tile_regs_acquire();
            if (flags & F_PRE) {
                const uint32_t pin_cb = (flags & 1) ? pinl_cb : pinr_cb;
                cb_wait_front(pin_cb, gw);
                reconfig_data_format_srca(w_cb, pin_cb);
                copy_init(pin_cb);
                for (uint32_t t = 0; t < gw; ++t) {
                    copy_tile(pin_cb, t, t);
                }
                cb_pop_front(pin_cb, gw);
                reconfig_data_format_srca(pin_cb, w_cb);
                matmul_block_init(xr_cb, w_cb, false, gw, 1, kt);
            }
            for (uint32_t k = k0; k < k1; k += BK) {  // one x ring block at a time
                const uint32_t n = k1 - k < BK ? k1 - k : BK;
                const uint32_t a = wall();
                cb_wait_front(xr_cb, BK);
                t_xw += wall() - a;
                for (uint32_t i = 0; i < n; ++i) {
                    matmul_block(xr_cb, w_cb, i, (k + i) * gw, 0, false, gw, 1, kt);
                }
                cb_pop_front(xr_cb, BK);
                ++cons;
            }
            tile_regs_commit();
            if (flags & F_SEND) {
                cb_reserve_back(pout_cb, gw);
                tile_regs_wait();
                if (h_fmt) {
                    pack_reconfig_data_format(h_cb, pout_cb);
                    h_fmt = false;
                }
                for (uint32_t t = 0; t < gw; ++t) {
                    pack_tile(t, pout_cb);
                }
                tile_regs_release();
                cb_push_back(pout_cb, gw);
            } else {
                cb_reserve_back(h_cb, npair);
                tile_regs_wait();
                if (!h_fmt) {
                    pack_reconfig_data_format(pout_cb, h_cb);
                    h_fmt = true;
                }
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t t = 0; t < gw; t += 2) {
#if WC_GLU == 0
                    expert_activation::tile_pair<expert_activation::Kind::Silu>(t);
#else
                    ws_glu::tile_pair<WC_GLU - 1>(t);
#endif
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                for (uint32_t t = 0; t < gw; t += 2) {
                    pack_tile(t, h_cb);
                }
                tile_regs_release();
                cb_push_back(h_cb, npair);
                ++produced;
            }
        }
        PACK(stamps[1 + e] = wall());
        // move the tilize pointer past this expert before its plan slot can be reused (its blocks are all tilized:
        // the expert consumed them all); else a later peek would wait for a plan that never comes back
        while (til_e <= e) {
            volatile uint32_t* tp = plan_of(til_e);
            if (til_i < tp[ws_plan::H_NMY]) {
                break;
            }
            til_base += tp[ws_plan::H_NBE];
            ++til_e;
            til_i = 0;
        }
        UNPACK(ctrl[ws_plan::DONE_T0] = e + 1);
        MATH(ctrl[ws_plan::DONE_T1] = e + 1);
        PACK(ctrl[ws_plan::DONE_T2] = e + 1);
    }
    while (consumed < produced) {
        down_row();
        ++consumed;
    }
    PACK(stamps[1 + nexp] = wall());
    UNPACK(stamps[40] = t_til);
    UNPACK(stamps[41] = t_xw);
    UNPACK(stamps[42] = t_dn);
    MATH(stamps[43] = t_til);
    MATH(stamps[44] = t_xw);
}
