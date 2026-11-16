// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert shared expert, compute (TRISC): one cluster core of the shared expert (ws_shared.py). The core owns NPAIR
// gate/up pairs and DC down columns, weights resident (CB 1, CB 3). Per row tile of its cluster: gate/up over the
// KT / BK x ring blocks (CB 0, the hub's multicasts), silu(g) * u -> NPAIR h tiles (CB 16: the NCRISC sends them to
// the hub, which multicasts whole h rows into every cluster core's h ring), then the down of the previous row over
// the whole h row (h ring CB 2, KTD tiles) -> pack-untilized into CB 17 (32 rows x DC * 32 columns, row after row)
// for the NCRISC's row writes. The down lags one row (the h ring holds NS rows).
// RT: 0 stamps, 1 rows. CT: 0 KT, 1 NPAIR, 2 BK, 3 KTD, 4 DC.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
#include "glu_fused.hpp"
#ifndef WC_GLU
#define WC_GLU 3
#endif

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;

constexpr uint32_t kt = get_compile_time_arg_val(0);
constexpr uint32_t npair = get_compile_time_arg_val(1);
constexpr uint32_t BK = get_compile_time_arg_val(2);
constexpr uint32_t ktd = get_compile_time_arg_val(3);
constexpr uint32_t DC = get_compile_time_arg_val(4);
constexpr uint32_t gw = 2 * npair;
constexpr uint32_t NB = kt / BK;  // x blocks a row
static_assert(kt % BK == 0 && gw <= 4 && DC <= 4);

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

uint32_t t_xw = 0, t_hw = 0;

FORCE_INLINE void gu_init() { matmul_block_init(xr_cb, w_cb, false, gw, 1, kt); }

#ifdef WX_X_FMT_DIFFERS
#define SRCB_TO_H() reconfig_data_format_srcb(xr_cb, hr_cb)
#define SRCB_TO_X() reconfig_data_format_srcb(hr_cb, xr_cb)
#else
#define SRCB_TO_H()
#define SRCB_TO_X()
#endif

FORCE_INLINE void down_row() {
    SRCB_TO_H();
    {
        const uint32_t a = wall();
        cb_wait_front(hr_cb, ktd);
        t_hw += wall() - a;
    }
    cb_reserve_back(y_cb, DC);
    matmul_block_init(hr_cb, wd_cb, false, DC, 1, ktd);
    pack_untilize_dest_init<DC, DC, false, TILE_C_DIM, false, false /*configure_remap*/>(y_cb);
    tile_regs_acquire();
    for (uint32_t k = 0; k < ktd; ++k) {
        matmul_block(hr_cb, wd_cb, k, k * DC, 0, false, DC, 1, ktd);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_untilize_dest<DC, DC>(y_cb, 1, 0);
    tile_regs_release();
    pack_untilize_uninit(y_cb);
    pack_reconfig_data_format(h_cb);  // (the uninit leaves the packer at y's format; h may differ)
    cb_pop_front(hr_cb, ktd);
    cb_push_back(y_cb, DC);
    SRCB_TO_X();
    gu_init();
}

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    const uint32_t rows = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(xr_cb, w_cb, h_cb);
    // DEST's untilize read remap on for the whole kernel: the down's pack-untilize needs it, the other packs work
    // with it on, and toggling it syncs the Tensix pipeline
    MATH((llk_math_reconfig_remap(true)));
    ws_glu::init<WC_GLU - 1>();
    cb_wait_front(w_cb, kt * gw);
    cb_wait_front(wd_cb, ktd * DC);
    UNPACK(stamps[0] = wall());
    gu_init();
    for (uint32_t i = 0; i < rows; ++i) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t a = wall();
            cb_wait_front(xr_cb, BK);
            t_xw += wall() - a;
            for (uint32_t k = 0; k < BK; ++k) {
                matmul_block(xr_cb, w_cb, k, (b * BK + k) * gw, 0, false, gw, 1, kt);
            }
            cb_pop_front(xr_cb, BK);
        }
        tile_regs_commit();
        cb_reserve_back(h_cb, npair);
        tile_regs_wait();
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t t = 0; t < gw; t += 2) {
            ws_glu::tile_pair<WC_GLU - 1>(t);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        for (uint32_t t = 0; t < gw; t += 2) {
            pack_tile(t, h_cb);
        }
        tile_regs_release();
        cb_push_back(h_cb, npair);
        if (i > 0) {
            down_row();
        }
    }
    if (rows > 0) {
        down_row();
    }
    PACK(stamps[63] = wall());
    UNPACK(stamps[41] = t_xw);
    UNPACK(stamps[42] = t_hw);
}
