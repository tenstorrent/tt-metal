// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash verify QKV heads split + RoPE, compute (qkv_rope_reader.cpp): per QKV tile (head h, column tile d)
// of a rotated head (h < ROPE_HEADS): out = x cos_d + (x T) sin_d (T: the rotation matrix tile, as
// rotary_embedding_llama's prefill mode); the other heads (V) copied. FP32 in DST.
// compile-time args: [cb_x, cb_cos, cb_sin, cb_t, cb_out, HPC, ROPE_HEADS]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(2);
    constexpr uint32_t cb_t = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t HPC = get_compile_time_arg_val(5);
    constexpr uint32_t ROPE_HEADS = get_compile_time_arg_val(6);
    const uint32_t hc = get_absolute_logical_x();

    compute_kernel_hw_startup(cb_x, cb_t, cb_out);
    cb_wait_front(cb_x, 2 * HPC);
    cb_wait_front(cb_cos, 2);
    cb_wait_front(cb_sin, 2);
    cb_wait_front(cb_t, 1);
    pack_reconfig_data_format(cb_out);
    for (uint32_t i = 0; i < 2 * HPC; ++i) {
        const uint32_t h = hc * HPC + i / 2, d = i % 2;
        tile_regs_acquire();
        if (h < ROPE_HEADS) {
            // DST 0 = (x T) sin
            reconfig_data_format(cb_t, cb_x);  // (matmul: SrcA <- in1 (T), SrcB <- in0 (x))
            matmul_init(cb_x, cb_t, 0);
            matmul_tiles(cb_x, cb_t, i, 0, 0);
            reconfig_data_format(cb_sin, cb_sin);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_sin);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_sin, d, 0);
            // DST 1 = x cos
            reconfig_data_format(cb_x, cb_cos);
            mul_init(cb_x, cb_cos);
            mul_tiles(cb_x, cb_cos, i, d, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        } else {
            reconfig_data_format_srca(cb_x);
            copy_init(cb_x);
            copy_tile(cb_x, i, 0);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_x, 2 * HPC);
    cb_pop_front(cb_cos, 2);
    cb_pop_front(cb_sin, 2);
    cb_pop_front(cb_t, 1);
}
