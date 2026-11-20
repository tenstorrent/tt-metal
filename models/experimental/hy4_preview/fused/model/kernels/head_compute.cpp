// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// LM head unit of the fused decode step (TRISC) (fused/model/ends_fused.py EndsHead; PLAN.md section 46): this core
// owns `cols` tile columns (32 token ids each) of the chip's vocabulary shard. Per column:
// logits = x . W[:, column] over the x_tiles K tiles (x: [ROWS, 32] BF16 row tiles, resident; W: BF16 tiles streamed
// in blocks of kb K tiles; the standard matmul at the kernel's math fidelity, HiFi4, FP32 accumulation), packed as
// one FP32 [ROWS, 32] tile. One tile in DEST at a time.
//
// fused/moe/kernels/router_compute.cpp without the sigmoid, looped over the core's columns. `cols` comes from the
// core's data-movement kernel (kernels/head_unit.cpp) through the mailbox: 0 = the hub never released this core,
// nothing is computed and nothing is waited for.
//
// compile-time args: [cb_x, cb_w, cb_out, x_tiles, kb]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t kb = get_compile_time_arg_val(4);

    // (a matmul's operands go to the source registers in reverse order: in0 to SrcB, in1 to SrcA; see
    // router_compute.cpp)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_block_init(cb_x, cb_w, false, 1, 1, kb);

    uint32_t cols = 0;
    UNPACK(cols = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(cols = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(cols = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    if (cols == 0) {
        return;
    }

    cb_wait_front(cb_x, x_tiles);
    for (uint32_t c = 0; c < cols; ++c) {
        tile_regs_acquire();
        for (uint32_t k0 = 0; k0 < x_tiles; k0 += kb) {
            cb_wait_front(cb_w, kb);
            for (uint32_t k = 0; k < kb; ++k) {
                matmul_block(cb_x, cb_w, k0 + k, k, 0, false, 1, 1, kb);
            }
            cb_pop_front(cb_w, kb);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
