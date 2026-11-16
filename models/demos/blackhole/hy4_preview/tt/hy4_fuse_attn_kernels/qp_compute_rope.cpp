// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// Hy4 B17 fused q-prep, rope cores: out_t = x_t * cos_t + (x_t @ tm) * sin_t, t = 0, 1 (adjacent-pair
// rotary_embedding_llama, one 32-column tile each).  CT: 0 cb_x, 1 cb_c, 2 cb_s, 3 cb_t, 4 cb_out
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_c = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_t = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    compute_kernel_hw_startup(cb_x, cb_t, cb_out);
    CircularBuffer x(cb_x), c(cb_c), s(cb_s), t(cb_t), o(cb_out);
    x.wait_front(2);
    c.wait_front(2);
    s.wait_front(2);
    t.wait_front(1);
    o.reserve_back(2);
    for (uint32_t i = 0; i < 2; ++i) {
        reconfig_data_format(cb_x, cb_t);
        matmul_init(cb_x, cb_t);
        tile_regs_acquire();
        matmul_tiles(cb_x, cb_t, i, 0, 0);  // dst0 = x @ tm
        copy_init(cb_s);
        copy_tile(cb_s, i, 1);  // dst1 = sin
        copy_init(cb_x);
        copy_tile(cb_x, i, 2);  // dst2 = x
        copy_init(cb_c);
        copy_tile(cb_c, i, 3);  // dst3 = cos
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(2, 3, 2);
        // the stock rotary_embedding_llama keeps both products in bf16 circular buffers: round them before the add
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(2);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, i);
        tile_regs_release();
    }
    o.push_back(2);
    x.pop_front(2);
    c.pop_front(2);
    s.pop_front(2);
    t.pop_front(1);
}
