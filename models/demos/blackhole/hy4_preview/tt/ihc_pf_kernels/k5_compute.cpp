// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K5 compute: out_{s,j} = x_{s,j} + post_s * y_j (exact fp32 SFPU; y bf16 -> fp32 exact, post_s
// replicated over the columns). CT: 0 cb_x, 1 cb_y, 2 cb_s, 3 cb_o, 4 CT. cb_x and cb_s are unpacked straight to dest
// (fp32 exact).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_o = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    compute_kernel_hw_startup(cb_x, cb_s, cb_o);
    CircularBuffer x(cb_x), y(cb_y), s(cb_s), o(cb_o);
    s.wait_front(4);
    for (uint32_t j = 0; j < CT; ++j) {
        if ((j & 1) == 0) {
            x.wait_front(8);
            y.wait_front(2);
        }
        const uint32_t xo = (j & 1) * 4, yo = (j & 1);
        for (uint32_t half = 0; half < 2; ++half) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_y);
            copy_tile_to_dst_init_short(cb_y);
            copy_tile(cb_y, yo, 0);
            reconfig_data_format_srca(cb_s);
            copy_tile_to_dst_init_short(cb_s);
            copy_tile(cb_s, 2 * half, 1);
            copy_tile(cb_s, 2 * half + 1, 2);
            reconfig_data_format_srca(cb_x);
            copy_tile_to_dst_init_short(cb_x);
            copy_tile(cb_x, xo + 2 * half, 3);
            copy_tile(cb_x, xo + 2 * half + 1, 4);
            mul_binary_tile_init();
            mul_binary_tile(1, 0, 1);
            mul_binary_tile(2, 0, 2);
            add_binary_tile_init();
            add_binary_tile(1, 3, 1);
            add_binary_tile(2, 4, 2);
            tile_regs_commit();
            tile_regs_wait();
            o.reserve_back(2);
            pack_reconfig_data_format(cb_o);
            pack_tile(1, cb_o);
            pack_tile(2, cb_o);
            o.push_back(2);
            tile_regs_release();
        }
        if ((j & 1) == 1) {
            x.pop_front(8);
            y.pop_front(2);
        }
    }
}
