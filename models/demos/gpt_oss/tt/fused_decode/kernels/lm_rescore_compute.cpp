// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-stage DFlash LM head, stage 2 with tile de-duplication, compute (TRISC) (dflash/lm_rescore.py): if
// the writer says this core owns its column tile (mailbox 1), out = x . W[:, tile] over kt K tiles ([rows, 32]
// activation tiles, custom_mm, LoFi, FP32 accumulation: kernels/stream_matmul_compute.cpp's arithmetic).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    uint32_t go = 0;
    UNPACK(go = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(go = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(go = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    if (!go) {
        return;
    }
    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, dest_layout>(cb_in0, cb_w, cb_out);
    cb_wait_front(cb_in0, kt);
    tile_regs_acquire();
    cb_wait_front(cb_w, kt);
    custom_mm_block<dest_layout>(cb_in0, cb_w, 0, 0, 0, kt);
    cb_pop_front(cb_w, kt);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in0, kt);
    custom_mm_block_uninit<dest_layout>(cb_out);
}
