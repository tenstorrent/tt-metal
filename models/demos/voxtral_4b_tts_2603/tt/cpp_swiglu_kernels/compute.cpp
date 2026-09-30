// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the fused C++ SwiGLU (see tt/cpp_swiglu.py): silu(x @ w1) * (x @ w3).
//
// The in1 block is interleaved gate/up, so an RT x 2 subblock holds, in DEST, g(r) u(r) for RT output
// rows of one column pair. K-partials spill to an fp32 CB between K blocks; after the last block silu
// and the product run on DEST and only the gate slots are packed. RT rows per subblock because the
// bf8_b weight tile is the unpack-bound operand: each one unpacked feeds RT products, not one.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t PN = get_compile_time_arg_val(3);
    constexpr uint32_t RT = get_compile_time_arg_val(4);
    constexpr uint32_t W = 2 * PN;
    constexpr uint32_t SUB = 2 * RT;

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_w = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;
    constexpr auto cb_p = tt::CBIndex::c_24;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_p);
    matmul_block_init(cb_x, cb_w, 0, 2, RT, KB);

    for (uint32_t b = 0; b < NB; ++b) {
        const bool last = b == NB - 1;
        cb_wait_front(cb_x, MT * KB);
        cb_wait_front(cb_w, KB * W);
        for (uint32_t rs = 0; rs < MT / RT; ++rs) {
            for (uint32_t p = 0; p < PN; ++p) {
                tile_regs_acquire();
                if (b > 0) {
                    reconfig_data_format_srca(cb_w, cb_p);
                    copy_init(cb_p);
                    cb_wait_front(cb_p, SUB);
                    for (uint32_t i = 0; i < SUB; ++i) {
                        copy_tile(cb_p, i, i);
                    }
                    cb_pop_front(cb_p, SUB);
                    reconfig_data_format_srca(cb_p, cb_w);
                    matmul_block_init(cb_x, cb_w, 0, 2, RT, KB);
                }
                uint32_t i0 = rs * RT * KB;
                uint32_t i1 = 2 * p;
                for (uint32_t k = 0; k < KB; ++k) {
                    matmul_block(cb_x, cb_w, i0, i1, 0, 0, 2, RT, KB);
                    ++i0;
                    i1 += W;
                }
                if (last) {
                    silu_tile_init();
                    for (uint32_t j = 0; j < RT; ++j) {
                        silu_tile(2 * j);
                    }
                    mul_binary_tile_init();
                    for (uint32_t j = 0; j < RT; ++j) {
                        mul_binary_tile(2 * j, 2 * j + 1, 2 * j);
                    }
                    tile_regs_commit();
                    cb_reserve_back(cb_y, RT);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_y);
                    for (uint32_t j = 0; j < RT; ++j) {
                        pack_tile(2 * j, cb_y);
                    }
                    tile_regs_release();
                    cb_push_back(cb_y, RT);
                    matmul_block_init(cb_x, cb_w, 0, 2, RT, KB);
                } else {
                    tile_regs_commit();
                    cb_reserve_back(cb_p, SUB);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_p);
                    for (uint32_t i = 0; i < SUB; ++i) {
                        pack_tile(i, cb_p);
                    }
                    tile_regs_release();
                    cb_push_back(cb_p, SUB);
                }
            }
        }
        cb_pop_front(cb_x, MT * KB);
        cb_pop_front(cb_w, KB * W);
    }
}
