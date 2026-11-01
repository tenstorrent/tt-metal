// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused prefill attention glue, compute.  Per block (one tile row x one head, 16 tiles = 512 dims):
//   [IN_RM: tilize the 32 row-major rows -> 16 tiles]; tiles 0..13 are copied; tiles 14, 15 (dims 448..511) get the
//   RoPE rotation with the same bf16 stages as the unfused ops (pf_tune.rope_pe): t1 = x*c, p = x@P, t3 = p*s, y = t1 +
//   t3 (each rounded to bf16); [OUT_RM: untilize the 16 tiles -> 32 row-major rows].
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    constexpr uint32_t cb_p = get_compile_time_arg_val(4);
    constexpr uint32_t cb_t1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_t2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_t3 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_y = get_compile_time_arg_val(8);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(9);
    constexpr uint32_t IN_RM = get_compile_time_arg_val(10);
    constexpr uint32_t OUT_RM = get_compile_time_arg_val(11);
    constexpr uint32_t ROPE = get_compile_time_arg_val(12);  // 0: pass-through (layout conversion only)
    const uint32_t nb = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_c, cb_y);
    CircularBuffer x(cb_x), c(cb_c), s(cb_s), p(cb_p), t1(cb_t1), t2(cb_t2), t3(cb_t3), y(cb_y);
    p.wait_front(1);

    for (uint32_t b = 0; b < nb; ++b) {
        if constexpr (IN_RM) {
            compute_kernel_lib::tilize<16, cb_in, cb_x>(1);
        }
        x.wait_front(16);
        c.wait_front(2);
        s.wait_front(2);

        // tiles 0..13 pass through
        reconfig_data_format(cb_x, cb_x);
        pack_reconfig_data_format(cb_y);
        copy_init(cb_x);
        constexpr uint32_t NCOPY = ROPE ? 14 : 16;
        for (uint32_t i = 0; i < NCOPY; ++i) {
            tile_regs_acquire();
            copy_tile(cb_x, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            y.reserve_back(1);
            pack_tile(0, cb_y);
            y.push_back(1);
            tile_regs_release();
        }
        if constexpr (ROPE) {
            for (uint32_t j = 0; j < 2; ++j) {
                // t1 = x * c
                reconfig_data_format(cb_x, cb_c);
                pack_reconfig_data_format(cb_t1);
                mul_init(cb_x, cb_c);
                tile_regs_acquire();
                mul_tiles(cb_x, cb_c, 14 + j, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                t1.reserve_back(1);
                pack_tile(0, cb_t1);
                t1.push_back(1);
                tile_regs_release();
                // p = x @ P
                reconfig_data_format(cb_x, cb_p);
                pack_reconfig_data_format(cb_t2);
                matmul_init(cb_x, cb_p);
                tile_regs_acquire();
                matmul_tiles(cb_x, cb_p, 14 + j, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                t2.reserve_back(1);
                pack_tile(0, cb_t2);
                t2.push_back(1);
                tile_regs_release();
                // t3 = p * s
                t2.wait_front(1);
                reconfig_data_format(cb_t2, cb_s);
                pack_reconfig_data_format(cb_t3);
                mul_init(cb_t2, cb_s);
                tile_regs_acquire();
                mul_tiles(cb_t2, cb_s, 0, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                t3.reserve_back(1);
                pack_tile(0, cb_t3);
                t3.push_back(1);
                tile_regs_release();
                t2.pop_front(1);
                // y = t1 + t3
                t1.wait_front(1);
                t3.wait_front(1);
                reconfig_data_format(cb_t1, cb_t3);
                pack_reconfig_data_format(cb_y);
                add_init(cb_t1, cb_t3);
                tile_regs_acquire();
                add_tiles(cb_t1, cb_t3, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                y.reserve_back(1);
                pack_tile(0, cb_y);
                y.push_back(1);
                tile_regs_release();
                t1.pop_front(1);
                t3.pop_front(1);
            }
        }
        x.pop_front(16);
        c.pop_front(2);
        s.pop_front(2);
        if constexpr (OUT_RM) {
            compute_kernel_lib::untilize<16, cb_y, cb_rm>(1);
        }
    }
}
