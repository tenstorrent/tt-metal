// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert end pass, compute: per token, y = sum over its S rows of row k * scalar k (the gate weights, 1.0 for the
// shared row), each row as 4 tiles of contiguous elements. The op is elementwise, so the tile layout the 1024
// contiguous elements land in does not matter, and packing puts them back in the same order. Each row is a
// SCALAR-broadcast ELWMUL into the token's 4 fp32 DEST tiles: ELWMUL adds its product to DEST (the standard binary
// path never clears it, and DEST starts each token cleared by the previous pack), so the S products accumulate there.
// RT: 0 tokens, 1 stamps (0: compute start, 63: end, 1: cycles waiting for a token's inputs, 2: for output space).
// CT: 0 S. WE_ROWS_CB / WE_W_CB / WE_OUT_CB: the CBs (default 0, 1, 16; the fold puts them beside the shared expert's).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
#ifndef WE_ROWS_CB
#define WE_ROWS_CB 0
#define WE_W_CB 1
#define WE_OUT_CB 16
#endif
    constexpr uint32_t rows_cb = WE_ROWS_CB, w_cb = WE_W_CB, out_cb = WE_OUT_CB;
    constexpr uint32_t TPR = 4;
    const uint32_t nt = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(1));
    UNPACK(stamps[0] = wall());
    compute_kernel_hw_startup(rows_cb, w_cb, out_cb);
    mul_bcast_scalar_init(rows_cb, w_cb);
    uint32_t t_in = 0, t_out = 0;
    for (uint32_t t = 0; t < nt; ++t) {
        uint32_t a = wall();
        cb_wait_front(rows_cb, S * TPR);
        cb_wait_front(w_cb, S);
        t_in += wall() - a;
        tile_regs_acquire();
        for (uint32_t k = 0; k < S; ++k) {
            for (uint32_t i = 0; i < TPR; ++i) {
                mul_tiles_bcast_scalar(rows_cb, w_cb, k * TPR + i, k, i);
            }
        }
        tile_regs_commit();
        cb_pop_front(rows_cb, S * TPR);
        cb_pop_front(w_cb, S);
        a = wall();
        cb_reserve_back(out_cb, TPR);
        t_out += wall() - a;
        tile_regs_wait();
        for (uint32_t i = 0; i < TPR; ++i) {
            pack_tile(i, out_cb, i);
        }
        tile_regs_release();
        cb_push_back(out_cb, TPR);
    }
    UNPACK(stamps[1] = t_in);
    PACK(stamps[2] = t_out);
    PACK(stamps[63] = wall());
}
