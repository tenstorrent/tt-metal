// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused prefill attention glue, compute.  Per block (one tile row x one head, NT tiles = 32 * NT dims):
//   [IN_RM: tilize the 32 row-major rows -> NT tiles]; tiles 0..NT-3 are copied; the last two tiles (the 64 RoPE dims)
//   are rotated: ROPE_MODE 0 (FPU stages):  t1 = x*c (mul_tiles), p = x@P, t3 = p*s, y = t1 + t3 (add_tiles), every
//   stage rounded to bf16 ROPE_MODE 1 (SFPU stages): the same with mul_binary_tile / add_binary_tile (the arithmetic of
//   ttnn.multiply / ttnn.add on bf16) ROPE_MODE 2 (addcmul):     y = addcmul(x*c (SFPU), x@P, s)  (the arithmetic of
//   the indexer rope ttnn.addcmul(ttnn.multiply(x, c), x @ P, s)) [OUT_RM: untilize the NT tiles -> 32 row-major rows].
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
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
    constexpr uint32_t NT = get_compile_time_arg_val(13);    // tiles per head block (16: 512 dims, 4: indexer 128 dims)
    constexpr uint32_t MODE = get_compile_time_arg_val(14);
    const uint32_t nb = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_c, cb_y);
    CircularBuffer x(cb_x), c(cb_c), s(cb_s), p(cb_p), t1(cb_t1), t2(cb_t2), t3(cb_t3), y(cb_y);
    p.wait_front(1);

    for (uint32_t b = 0; b < nb; ++b) {
        if constexpr (IN_RM) {
            compute_kernel_lib::tilize<NT, cb_in, cb_x>(1);
        }
        x.wait_front(NT);
        c.wait_front(2);
        s.wait_front(2);

        // the non-rotated tiles pass through
        reconfig_data_format(cb_x, cb_x);
        pack_reconfig_data_format(cb_y);
        copy_init(cb_x);
        constexpr uint32_t NCOPY = ROPE ? NT - 2 : NT;
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
                const uint32_t xi = NT - 2 + j;
                if constexpr (MODE == 0) {
                    // t1 = x * c
                    reconfig_data_format(cb_x, cb_c);
                    pack_reconfig_data_format(cb_t1);
                    mul_init(cb_x, cb_c);
                    tile_regs_acquire();
                    mul_tiles(cb_x, cb_c, xi, j, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    t1.reserve_back(1);
                    pack_tile(0, cb_t1);
                    t1.push_back(1);
                    tile_regs_release();
                } else {
                    // t1 = x * c on the SFPU (MODE 1), or m = x * c kept in DEST for the addcmul (MODE 2)
                    reconfig_data_format(cb_x, cb_x);
                    pack_reconfig_data_format(cb_t1);
                    if constexpr (MODE == 1) {
                        tile_regs_acquire();
                        copy_init(cb_x);
                        copy_tile(cb_x, xi, 0);
                        reconfig_data_format_srca(cb_x, cb_c);
                        copy_init(cb_c);
                        copy_tile(cb_c, j, 1);
                        mul_binary_tile_init();
                        mul_binary_tile(0, 1, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        t1.reserve_back(1);
                        pack_tile(0, cb_t1);
                        t1.push_back(1);
                        tile_regs_release();
                    }
                }
                // p = x @ P
                reconfig_data_format(cb_x, cb_p);
                pack_reconfig_data_format(cb_t2);
                matmul_init(cb_x, cb_p);
                tile_regs_acquire();
                matmul_tiles(cb_x, cb_p, xi, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                t2.reserve_back(1);
                pack_tile(0, cb_t2);
                t2.push_back(1);
                tile_regs_release();
                t2.wait_front(1);
                if constexpr (MODE == 2) {
                    // y = addcmul(x * c, p, s)
                    reconfig_data_format(cb_x, cb_x);
                    pack_reconfig_data_format(cb_y);
                    tile_regs_acquire();
                    copy_init(cb_x);
                    copy_tile(cb_x, xi, 0);
                    reconfig_data_format_srca(cb_x, cb_c);
                    copy_init(cb_c);
                    copy_tile(cb_c, j, 1);
                    mul_binary_tile_init();
                    mul_binary_tile(0, 1, 0);
                    reconfig_data_format_srca(cb_c, cb_t2);
                    copy_init(cb_t2);
                    copy_tile(cb_t2, 0, 1);
                    reconfig_data_format_srca(cb_t2, cb_s);
                    copy_init(cb_s);
                    copy_tile(cb_s, j, 2);
                    addcmul_tile_init();
                    addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, 0x3F800000u);
                    tile_regs_commit();
                    tile_regs_wait();
                    y.reserve_back(1);
                    pack_tile(0, cb_y);
                    y.push_back(1);
                    tile_regs_release();
                    t2.pop_front(1);
                } else {
                    // t3 = p * s
                    pack_reconfig_data_format(cb_t3);
                    if constexpr (MODE == 0) {
                        reconfig_data_format(cb_t2, cb_s);
                        mul_init(cb_t2, cb_s);
                        tile_regs_acquire();
                        mul_tiles(cb_t2, cb_s, 0, j, 0);
                    } else {
                        reconfig_data_format(cb_t2, cb_t2);
                        tile_regs_acquire();
                        copy_init(cb_t2);
                        copy_tile(cb_t2, 0, 0);
                        reconfig_data_format_srca(cb_t2, cb_s);
                        copy_init(cb_s);
                        copy_tile(cb_s, j, 1);
                        mul_binary_tile_init();
                        mul_binary_tile(0, 1, 0);
                    }
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
                    pack_reconfig_data_format(cb_y);
                    if constexpr (MODE == 0) {
                        reconfig_data_format(cb_t1, cb_t3);
                        add_init(cb_t1, cb_t3);
                        tile_regs_acquire();
                        add_tiles(cb_t1, cb_t3, 0, 0, 0);
                    } else {
                        reconfig_data_format(cb_t1, cb_t1);
                        tile_regs_acquire();
                        copy_init(cb_t1);
                        copy_tile(cb_t1, 0, 0);
                        reconfig_data_format_srca(cb_t1, cb_t3);
                        copy_init(cb_t3);
                        copy_tile(cb_t3, 0, 1);
                        add_binary_tile_init();
                        add_binary_tile(0, 1, 0);
                    }
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
        }
        x.pop_front(NT);
        c.pop_front(2);
        s.pop_front(2);
        if constexpr (OUT_RM) {
            compute_kernel_lib::untilize<NT, cb_y, cb_rm>(1);
        }
    }
}
