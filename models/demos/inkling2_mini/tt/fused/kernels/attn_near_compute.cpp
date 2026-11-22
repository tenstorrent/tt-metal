// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused near partial, compute. Core (g, i): slot tiles j of kv group g (3 query heads in rows 0..2, rest zero):
//   S_j = q K_j^T + mask_j                     fp32 (q pre-scaled, mask = rel bias + near mask)
//   e_j = exp(S_j - c)                         SFPU fp32, packed bf16. c = this core's row max (of its scores and,
//                                              on the sink core, the sink block), computed by the writer RISC
//   z = sum_j e_j @ ones (+ e_s @ ones, has_sink)   row sums of those bf16 values, in every column
//   O = sum_j e_j V_j                          unnormalized; O, z and c go to the combine program
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_mask = 3, cb_sink = 4, cb_ones = 6, cb_c = 8;
constexpr uint32_t cb_s = 16, cb_e = 19, cb_es = 20, cb_z = 22, cb_o = 23;
constexpr uint32_t dt = 4;

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    constexpr uint32_t has_sink = get_named_compile_time_arg_val("has_sink");
    compute_kernel_hw_startup(cb_q, cb_k, cb_s);
    cb_wait_front(cb_q, dt);
    cb_wait_front(cb_sink, 1);
    cb_wait_front(cb_mask, nj);
    cb_wait_front(cb_ones, 1);

    // S_j
    for (uint32_t j = 0; j < nj; ++j) {
        cb_wait_front(cb_k, dt);
        tile_regs_acquire();
        reconfig_data_format(cb_k, cb_q);  // matmul: srcA = in1
        matmul_init(cb_q, cb_k, 1);
        for (uint32_t t = 0; t < dt; ++t) {
            matmul_tiles(cb_q, cb_k, t, t, 0);
        }
        load(cb_mask, j, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_s);
        tile_regs_release();
        cb_pop_front(cb_k, dt);
    }

    // e (the writer scans S for the row max and pushes c)
    cb_wait_front(cb_s, nj);
    cb_wait_front(cb_c, 1);
    constexpr uint32_t ne = nj + has_sink;
    for (uint32_t j = 0; j < ne; ++j) {
        tile_regs_acquire();
        load(j < nj ? cb_s : cb_sink, j < nj ? j : 0, 0);
        load(cb_c, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, j < nj ? cb_e : cb_es);
        tile_regs_release();
    }
    cb_pop_front(cb_s, nj);

    // z (every column)
    cb_wait_front(cb_e, nj);
    if constexpr (has_sink) {
        cb_wait_front(cb_es, 1);
    }
    tile_regs_acquire();
    reconfig_data_format(cb_ones, cb_e);
    matmul_init(cb_e, cb_ones, 0);
    for (uint32_t j = 0; j < nj; ++j) {
        matmul_tiles(cb_e, cb_ones, j, 0, 0);
    }
    if constexpr (has_sink) {
        matmul_tiles(cb_es, cb_ones, 0, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    put(0, cb_z);
    tile_regs_release();

    // O = sum_j e_j V_j
    tile_regs_acquire();
    reconfig_data_format(cb_v, cb_e);
    matmul_init(cb_e, cb_v, 0);
    for (uint32_t j = 0; j < nj; ++j) {
        cb_wait_front(cb_v, dt);
        for (uint32_t t = 0; t < dt; ++t) {
            matmul_tiles(cb_e, cb_v, j, t, t);
        }
        cb_pop_front(cb_v, dt);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_o, dt);
    pack_reconfig_data_format(cb_o);
    for (uint32_t t = 0; t < dt; ++t) {
        pack_tile(t, cb_o);
    }
    cb_push_back(cb_o, dt);
    tile_regs_release();
}
