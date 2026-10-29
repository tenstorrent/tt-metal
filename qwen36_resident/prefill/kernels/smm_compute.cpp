// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Small-M streaming matmul, compute: this core's [Mt, nt] output tiles, accumulated over the K blocks in
// L1 by the packer (cb_acc is never pushed during the walk, so every block packs to the same tiles), then
// copied to cb_out, through SiLU when SILU is set.
//
// Compile-time args: 0 Mt, 1 kb, 2 nt, 3 blocks, 4 sh, 5 sw, 6 SILU.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

namespace {
constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_acc = 24, cb_out = 16;
}

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t kb = get_compile_time_arg_val(1);
    constexpr uint32_t nt = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t sh = get_compile_time_arg_val(4);
    constexpr uint32_t sw = get_compile_time_arg_val(5);
    constexpr bool silu = get_compile_time_arg_val(6);
    static_assert(Mt % sh == 0 && nt % sw == 0, "output block splits into subblocks");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_acc);
    matmul_block_init(cb_in0, cb_in1, 0, sw, sh, kb);
    cb_reserve_back(cb_acc, Mt * nt);
    for (uint32_t b = 0; b < blocks; b++) {
        cb_wait_front(cb_in0, Mt * kb);
        cb_wait_front(cb_in1, kb * nt);
        pack_reconfig_l1_acc(b > 0 ? 1 : 0);
        for (uint32_t i0 = 0; i0 < Mt; i0 += sh) {
            for (uint32_t j0 = 0; j0 < nt; j0 += sw) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < kb; k++) {
                    matmul_block(cb_in0, cb_in1, i0 * kb + k, k * nt + j0, 0, false, sw, sh, kb);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < sh; i++) {
                    for (uint32_t j = 0; j < sw; j++) {
                        pack_tile<true>(i * sw + j, cb_acc, (i0 + i) * nt + j0 + j);
                    }
                }
                tile_regs_release();
            }
        }
        cb_pop_front(cb_in0, Mt * kb);
        cb_pop_front(cb_in1, kb * nt);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, Mt * nt);

    cb_wait_front(cb_acc, Mt * nt);
    cb_reserve_back(cb_out, Mt * nt);
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    copy_tile_to_dst_init_short(cb_acc);
    if constexpr (silu) {
        silu_tile_init();
    }
    for (uint32_t t = 0; t < Mt * nt; t++) {
        tile_regs_acquire();
        copy_tile(cb_acc, t, 0);
        if constexpr (silu) {
            silu_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, Mt * nt);
    cb_pop_front(cb_acc, Mt * nt);
}
