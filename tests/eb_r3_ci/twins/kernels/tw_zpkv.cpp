// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of zero_padded_kv_cache's compute loop
// (ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/compute/zero_padded_kv_cache.cpp:20-46):
// the same init and loop, Wt from common arg 7 as there, run TWIN_ITERS (compile arg 3) times.
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mask_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(3);

    const uint32_t Wt = get_common_arg_val<uint32_t>(7);

    CircularBuffer src(src_cb);
    CircularBuffer mask(mask_cb);
    CircularBuffer out(out_cb);

    compute_kernel_hw_startup(src_cb, mask_cb, out_cb);
    mul_init(src_cb, mask_cb);

    for (uint32_t it = 0; it < twin_iters; ++it) {
        mask.wait_front(1);
        src.wait_front(Wt);
        out.reserve_back(Wt);
        for (uint32_t i = 0; i < Wt; ++i) {
            tile_regs_acquire();
            mul_tiles(src_cb, mask_cb, i, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();
        }
        out.push_back(Wt);
        src.pop_front(Wt);
        mask.pop_front(1);
    }
}
