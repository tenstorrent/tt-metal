// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router scores of the Hy4 decode MoE program (TRISC) (fused/moe/program.py): this core owns one tile column of the
// router (32 experts). logits = x . W[:, column] over the 192 K tiles (x: [ROWS, 32] BF16 row tiles; W: BF16 tiles
// streamed in blocks of kb K tiles; the standard matmul at the kernel's math fidelity, HiFi4, FP32 accumulation),
// scores = sigmoid(logits), packed as one FP32 [ROWS, 32] tile.
//
// compile-time args: [cb_x, cb_w, cb_out, x_tiles, kb]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
#ifdef HY4_LAUNCH_ONLY  // measurement only: what launching this program costs
    return;
#endif
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t kb = get_compile_time_arg_val(4);

    // (a matmul's operands go to the source registers in reverse order: in0 to SrcB, in1 to SrcA; with the regular
    // order the weights were unpacked with x's [8, 32] geometry and the scores came out wrong, log mk5_board_7)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_block_init(cb_x, cb_w, false, 1, 1, kb);

    uint32_t go = 0;
    UNPACK(go = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(go = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(go = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    if (go == 0) {
        return;
    }

    cb_wait_front(cb_x, x_tiles);
    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < x_tiles; k0 += kb) {
        cb_wait_front(cb_w, kb);
        for (uint32_t k = 0; k < kb; ++k) {
            matmul_block(cb_x, cb_w, k0 + k, k, 0, false, 1, 1, kb);
        }
        cb_pop_front(cb_w, kb);
    }
    sigmoid_tile_init();
    sigmoid_tile<VectorMode::R, false, DST_ACCUM_MODE, ROWS <= 4 ? 2 : ROWS / 2>(0);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
