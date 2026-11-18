// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EP-4 pre-sum (ep4_layer.build_combine), adder compute: per run of the core's range with count c > 0 (the run table in
// L1, valid once TAB_FLAG is set), partial = sum over j < c of w_j * O row base + j, in j order (fp32 DEST: each row is
// a SCALAR-broadcast ELWMUL into the run's 4 DEST tiles, which accumulates - as we_compute.cpp), packed bf16 into
// CB 16. At the end one page into CB 17 (the writer clears TAB_FLAG once it sees it). CT: 0 tab (L1), 1 tab flag (L1
// word). RT: 0 nt, 1 stamps (0 start, 63 end, 1 input wait, 2 output wait cycles).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t tab = get_compile_time_arg_val(0);
    constexpr uint32_t tab_flag = get_compile_time_arg_val(1);
    constexpr uint32_t rows_cb = 0, w_cb = 1, out_cb = 16, done_cb = 17, TPR = 4;
    const uint32_t nt = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(1));
    volatile uint32_t* tb = reinterpret_cast<volatile uint32_t*>(tab);
    volatile uint32_t* flag = reinterpret_cast<volatile uint32_t*>(tab_flag);
    UNPACK(stamps[0] = wall());
    compute_kernel_hw_startup(rows_cb, w_cb, out_cb);
    mul_bcast_scalar_init(rows_cb, w_cb);
    while (*flag == 0) {
    }
    uint32_t t_in = 0, t_out = 0;
    for (uint32_t i = 0; i < nt; ++i) {
        const uint32_t c = tb[i * 32 + 1];
        if (c == 0) {
            continue;
        }
        tile_regs_acquire();
        for (uint32_t j = 0; j < c; ++j) {
            const uint32_t a = wall();
            cb_wait_front(rows_cb, TPR);
            cb_wait_front(w_cb, 1);
            t_in += wall() - a;
            for (uint32_t q = 0; q < TPR; ++q) {
                mul_tiles_bcast_scalar(rows_cb, w_cb, q, 0, q);
            }
            cb_pop_front(rows_cb, TPR);
            cb_pop_front(w_cb, 1);
        }
        tile_regs_commit();
        const uint32_t a = wall();
        cb_reserve_back(out_cb, TPR);
        t_out += wall() - a;
        tile_regs_wait();
        for (uint32_t q = 0; q < TPR; ++q) {
            pack_tile(q, out_cb, q);
        }
        tile_regs_release();
        cb_push_back(out_cb, TPR);
    }
    cb_reserve_back(done_cb, 1);
    cb_push_back(done_cb, 1);
    UNPACK(stamps[1] = t_in);
    PACK(stamps[2] = t_out);
    PACK(stamps[63] = wall());
}
