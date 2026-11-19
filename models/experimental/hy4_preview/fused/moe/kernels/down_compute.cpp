// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down projection of the Hy4 decode MoE program (TRISC) (fused/moe/program.py). Derived from
// models/demos/gpt_oss/tt/fused_decode/kernels/stream_down_union_compute.cpp.
//
// This core owns `cols` output tile columns. One DST acquire holds them for the whole union: for union slot i the
// slot's P activation tiles ([ROWS, 32] BF16, tiles i * P .. of the activation buffer) times this core's columns of
// the expert's down weights (one block, column-major) accumulate into DST 0 .. cols - 1 (custom_mm, LoFi, FP32
// accumulation; split accumulators merged with the last slot).
//
// compile-time args: [cb_act, cb_w, cb_out, P, cols]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

#ifndef PLAIN_TERMS
#define PLAIN_TERMS 1
#endif

void kernel_main() {
    constexpr uint32_t cb_act = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t P = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);

    constexpr bool transpose = false;
    constexpr bool read_transposed = true;  // column-major weights
#if DOWN_DENSE
    constexpr auto dest_layout = CustomMmDestLayout::DenseSplitAcc;
    static_assert(cols <= 8, "the output tiles of a core stay in DST (8 FP32 tiles, 32 rows apart)");
#else
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    static_assert(cols <= 4, "the output tiles of a core stay in DST (4 FP32 tiles)");
#endif
    custom_mm_block_init<transpose, dest_layout>(cb_act, cb_w, cb_out, cols);

    uint32_t U = 0;
    UNPACK(U = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(U = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(U = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    [[maybe_unused]] uint32_t first_plain = 0;  // slots from here on have PLAIN_TERMS terms
    UNPACK(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    if (U > 0) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < U; ++i) {
            cb_wait_front(cb_act, (i + 1) * P);  // filled in slot order by the finishers, never popped
#if PLAIN_TERMS > 1
            const uint32_t terms = i >= first_plain ? PLAIN_TERMS : 1;
#else
            constexpr uint32_t terms = 1;
#endif
            for (uint32_t term = 0; term < terms; ++term) {  // a plain slot's weight is the sum of its terms
                cb_wait_front(cb_w, cols * P);
#ifndef HY4_NO_MATH
                if (i + 1 == U && term + 1 == terms) {
                    custom_mm_block<dest_layout, true, read_transposed>(cb_act, cb_w, i * P, 0, 0, P, cols);
                } else {
                    custom_mm_block<dest_layout, false, read_transposed>(cb_act, cb_w, i * P, 0, 0, P, cols);
                }
#endif
                cb_pop_front(cb_w, cols * P);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, cols);
        tile_regs_wait();
        for (uint32_t c = 0; c < cols; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols);
    }
    custom_mm_block_uninit<dest_layout>(cb_out);
}
