// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert down compute (TRISC). The core owns CW output columns of every row. Per row (in the h ring's order):
// DST[1 x CW] (fp32) = h[row, 0:KTD] (CB 2, the h ring) @ Wd[0:KTD, columns] (CB 1, resident) -> CB 16.
// RT: 0 stamps addr. CT: 0 KTD, 1 CW, 2 R, 3 NEXP
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"

constexpr uint32_t h_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_1;
constexpr uint32_t out_cb = tt::CBIndex::c_16;
constexpr uint32_t ktd = get_compile_time_arg_val(0);
constexpr uint32_t cw = get_compile_time_arg_val(1);
constexpr uint32_t R = get_compile_time_arg_val(2);
constexpr uint32_t nexp = get_compile_time_arg_val(3);
static_assert(cw <= 4, "fp32 half-sync DST holds 4 tiles");

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    compute_kernel_hw_startup<SrcOrder::Reverse>(h_cb, wd_cb, out_cb);
    matmul_block_init(h_cb, wd_cb, false, cw, 1, ktd);
    cb_wait_front(wd_cb, ktd * cw);
    PACK(stamps[0] = wall());
    for (uint32_t e = 0; e < nexp; ++e) {
        for (uint32_t r = 0; r < R; ++r) {
            cb_wait_front(h_cb, ktd);
            tile_regs_acquire();
            for (uint32_t k = 0; k < ktd; ++k) {
                matmul_block(h_cb, wd_cb, k, k * cw, 0, false, cw, 1, ktd);
            }
            tile_regs_commit();
            cb_pop_front(h_cb, ktd);
            cb_reserve_back(out_cb, cw);
            tile_regs_wait();
            for (uint32_t t = 0; t < cw; ++t) {
                pack_tile(t, out_cb);
            }
            tile_regs_release();
            cb_push_back(out_cb, cw);
        }
        PACK(stamps[1 + e] = wall());
    }
}
