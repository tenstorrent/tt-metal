// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Microbench: matmul_block throughput by output block shape (rt x ct tiles), as the flat expert's gate/up / down
// computes run it: in0 (rt x K tiles, e.g. bfp8 x / h) times in1 (K x ct tiles, bfp4 weights), fp32 DEST. The inputs
// sit in their CBs (pushed once by the reader, never popped): only the unpack + math path is timed, a zone over
// `reps` blocks of K steps (MATH thread). Compile-time args: rt, ct, k (K steps a block), reps, k_call (1: one
// matmul_block_k a block, 0: a matmul_block a step)
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t k = get_compile_time_arg_val(2);
    constexpr uint32_t reps = get_compile_time_arg_val(3);
    constexpr bool k_call = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t in0 = tt::CBIndex::c_0, in1 = tt::CBIndex::c_1, out = tt::CBIndex::c_16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_block_init(in0, in1, false, ct, rt, k);
    cb_wait_front(in0, rt * k);
    cb_wait_front(in1, k * ct);
    {
        DeviceZoneScopedN("MM-SHAPE");
        for (uint32_t r = 0; r < reps; ++r) {
            tile_regs_acquire();
            if constexpr (k_call) {
                matmul_block_k(in0, in1, 0, 0, 0, false, ct, rt, k, k, ct);
            } else {
                for (uint32_t kk = 0; kk < k; ++kk) {
                    matmul_block(in0, in1, kk, kk * ct, 0, false, ct, rt, k);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
        }
    }
}
