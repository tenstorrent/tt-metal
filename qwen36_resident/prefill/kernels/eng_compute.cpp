// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill engine compute: one core's [mt, nt] output block of a distributed matmul, accumulated over the
// K blocks in L1 by the packer (out subblocks of sh x sw tiles in DEST; the first K block overwrites, the
// rest accumulate). The accumulator cb_acc is never pushed during the walk, so every K block packs to the
// same tiles; it is pushed once at the end for the next stage.
//
// Compile-time args: 0 mt, 1 kb, 2 nt, 3 blocks, 4 sh, 5 sw.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

namespace {
constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_acc = 16;
}

void kernel_main() {
    constexpr uint32_t mt = get_compile_time_arg_val(0);
    constexpr uint32_t kb = get_compile_time_arg_val(1);
    constexpr uint32_t nt = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t sh = get_compile_time_arg_val(4);
    constexpr uint32_t sw = get_compile_time_arg_val(5);
    static_assert(mt % sh == 0 && nt % sw == 0, "output block must split into subblocks");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_acc);
    matmul_block_init(cb_in0, cb_in1, 0, sw, sh, kb);
    cb_reserve_back(cb_acc, mt * nt);
    for (uint32_t b = 0; b < blocks; b++) {
        cb_wait_front(cb_in0, mt * kb);
        cb_wait_front(cb_in1, kb * nt);
        if (b == 1) {
            pack_reconfig_l1_acc(1);
        }
        for (uint32_t i0 = 0; i0 < mt; i0 += sh) {
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
        cb_pop_front(cb_in0, mt * kb);
        cb_pop_front(cb_in1, kb * nt);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, mt * nt);
}
