// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 router projection, matmul core reader (NCRISC) (fused_decode/router.py: RouterB1).
//
// Core (n, kb) computes row 0 of logits tile n restricted to K tiles [kb * kt, kb * kt + kt): it reads its `kt` weight
// tiles (BF16, pages [core * kt, core * kt + kt) of the per-core weight layout) and builds `kt` FP32 activation tiles
// whose row 0 is the flat FP32 router input x[32 k .. 32 k + 31] (face 0 row 0 / face 1 row 0); rows 1..31 are never
// written (they only reach rows 1..31 of the product, which nobody reads).
//
// runtime args: [w_addr, x_addr, x_noc_x, x_noc_y, core, kb]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);  // flat FP32 x on the boundary core
    const uint32_t x_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t x_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);
    const uint32_t kb = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr auto w_args = TensorAccessorArgs<3>();
    const auto s_w = TensorAccessor(w_args, w_addr, 2048);

    cb_reserve_back(cb_w, kt);
    const uint32_t w_l1 = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < kt; ++t) {
        noc_async_read(s_w.get_noc_addr(core * kt + t), w_l1 + 2048 * t, 2048);
    }
    cb_reserve_back(cb_x, kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    // the core's x slice (kt x 32 FP32 values) in one read into the last activation tile's rows 16..31 (faces 2 / 3,
    // never read back: they feed only rows 16..31 of the product), then row 0 of each tile filled locally
    const uint32_t stage = x_l1 + 4096 * (kt - 1) + 2048;
    static_assert(kt * 128 <= 2048, "the x slice fits faces 2 / 3 of one tile");
    noc_async_read(get_noc_addr(x_noc_x, x_noc_y, x_addr + 128 * kb * kt), stage, kt * 128);
    noc_async_read_barrier();
    for (uint32_t t = 0; t < kt; ++t) {
        noc_async_read(get_noc_addr(stage + 128 * t), x_l1 + 4096 * t, 64);
        noc_async_read(get_noc_addr(stage + 128 * t + 64), x_l1 + 4096 * t + 1024, 64);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);
    cb_push_back(cb_w, kt);
}
