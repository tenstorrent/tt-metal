// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Attention o-projection + post-attention short conv on this core's columns, compute (tt/fused/attn_wo.py).
// Matmul part: bank_linear_compute.cpp unchanged. conv: the op sequence of conv_compute.cpp on one dense page
// (slot k = output column tile k of this core, see attn_wo_writer.cpp): elementwise fp32 SFPU, so each element is
// bit-equal to the conv program's (SFPU on the faces holding the used slots only):
//   y = a*tap0 + ring1*tap1 + ring2*tap2 + ring3*tap3; yr = bf16(y); out = yr (+ resid), fp32; xf = fp32(a).
// Runtime args: rot (unused), number of passes, then ncq per pass.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_out = 16;
constexpr uint32_t cb_a = 2, cb_ring = 3, cb_taps = 4, cb_res = 5, cb_xf = 17, cb_y = 18, cb_yr = 19;
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t nblk = get_named_compile_time_arg_val("nblk");
constexpr uint32_t max_ncq = get_named_compile_time_arg_val("max_ncq");
constexpr uint32_t has_resid = get_named_compile_time_arg_val("res_es") != 0;
constexpr uint32_t slots = get_named_compile_time_arg_val("slots");  // used elements = 32 * slots
constexpr VectorMode vm = slots <= 8 ? VectorMode::None : slots <= 16 ? VectorMode::R : VectorMode::RC;
// face 0 only (slots <= 8): its first 2 slots rows, IT 4-row groups (2 per 4 rows); the writer reads 32 slots
// elements. wo_it 0 = the whole face.
constexpr int IT = get_named_compile_time_arg_val("wo_it") && slots <= 8 ? (2 * slots + 3) / 4 * 2 : 8;
constexpr uint32_t SLOT = kb * max_ncq;
constexpr uint32_t DST_TILES = 8;

namespace {
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
    uint32_t a = 0;
    a++;  // rot: the writer lays x out in walk order
    const uint32_t npass = get_arg_val<uint32_t>(a++);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t ncq = get_arg_val<uint32_t>(a++);
        matmul_block_init(cb_x, cb_w, false, ncq, 1, 1);
        tile_regs_acquire();
        for (uint32_t p = 0; p < nblk; ++p) {
            if (q == 0) {  // x arrives in walk order: block p (= K block (p + rot) % nblk) at position p
                cb_wait_front(cb_x, (p + 1) * kb);
            }
            cb_wait_front(cb_w, SLOT);
            for (uint32_t r = 0; r < kb; ++r) {
                matmul_block(cb_x, cb_w, p * kb + r, r * ncq, 0, false, ncq, 1, 1);
            }
            cb_pop_front(cb_w, SLOT);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, DST_TILES);
        tile_regs_wait();
        for (uint32_t t = 0; t < ncq; ++t) {
            pack_tile<true>(t, cb_out, t);
        }
        tile_regs_release();
        cb_push_back(cb_out, DST_TILES);
    }
    cb_pop_front(cb_x, kt);

    {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_taps, 4);
        cb_wait_front(cb_ring, 3);
        tile_regs_acquire();
        load(cb_a, 0, 0);
        load(cb_taps, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile<DST_ACCUM_MODE, IT>(0, 1, 1, vm);
        for (uint32_t k = 1; k < 4; ++k) {
            load(cb_ring, k - 1, 2);
            load(cb_taps, k, 3);
            mul_binary_tile_init();
            mul_binary_tile<DST_ACCUM_MODE, IT>(2, 3, 2, vm);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::Default, DST_ACCUM_MODE, IT>(1, 2, 1, vm);
        }
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_xf);
        put(1, cb_yr);
        tile_regs_release();
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_taps, 4);
        cb_pop_front(cb_ring, 3);

        cb_wait_front(cb_yr, 1);
        if constexpr (has_resid) {
            cb_wait_front(cb_res, 1);
        }
        tile_regs_acquire();
        load(cb_yr, 0, 0);
        if constexpr (has_resid) {
            load(cb_res, 0, 1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::Default, DST_ACCUM_MODE, IT>(0, 1, 0, vm);
        }
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_y);
        tile_regs_release();
        cb_pop_front(cb_yr, 1);
        if constexpr (has_resid) {
            cb_pop_front(cb_res, 1);
        }
    }
}
