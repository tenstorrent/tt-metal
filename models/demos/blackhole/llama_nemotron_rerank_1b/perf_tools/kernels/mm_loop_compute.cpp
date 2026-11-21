// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// probe_mm_loop.py: minimal_matmul's compute loop (matmul_blocks) on tiles already in L1, with no
// reader / writer and no CB handshakes, REPS times over one (M_block x K_block) x (K_block x N_block)
// block: the math thread's own ceiling. PACK_EVERY_K: pack each subblock after every K block with L1
// accumulation (minimal_matmul); otherwise the K loop of REPS blocks stays in DST (pack at the end).
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t REPS = get_compile_time_arg_val(0);
    constexpr uint32_t M_block = get_compile_time_arg_val(1);
    constexpr uint32_t K_block = get_compile_time_arg_val(2);
    constexpr uint32_t N_block = get_compile_time_arg_val(3);
    constexpr uint32_t sh = get_compile_time_arg_val(4);
    constexpr uint32_t sw = get_compile_time_arg_val(5);
    constexpr uint32_t in0_cb = 0, in1_cb = 1, out_cb = 3;
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, out_cb);
    matmul_init(in0_cb, in1_cb);
    matmul_block_init(in0_cb, in1_cb, false, sw, sh, K_block);
#ifdef PACK_EVERY_K
    for (uint32_t r = 0; r < REPS; ++r) {
        for (uint32_t m0 = 0; m0 < M_block; m0 += sh) {
            for (uint32_t n0 = 0; n0 < N_block; n0 += sw) {
                tile_regs_acquire();
                uint32_t in0_index = m0 * K_block, in1_index = n0;
                for (uint32_t k = 0; k < K_block; ++k) {
                    matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, sw, sh, K_block);
                    in0_index++;
                    in1_index += N_block;
                }
                tile_regs_commit();
                tile_regs_wait();
                uint32_t d = 0;
                for (uint32_t h = 0; h < sh; ++h) {
                    for (uint32_t w = 0; w < sw; ++w) {
                        pack_tile<true>(d++, out_cb, (m0 + h) * N_block + n0 + w);
                    }
                }
                tile_regs_release();
            }
        }
        if (r == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
    }
#else
    for (uint32_t m0 = 0; m0 < M_block; m0 += sh) {
        for (uint32_t n0 = 0; n0 < N_block; n0 += sw) {
            tile_regs_acquire();
            for (uint32_t r = 0; r < REPS; ++r) {
                uint32_t in0_index = m0 * K_block, in1_index = n0;
                for (uint32_t k = 0; k < K_block; ++k) {
                    matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, sw, sh, K_block);
                    in0_index++;
                    in1_index += N_block;
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t d = 0;
            for (uint32_t h = 0; h < sh; ++h) {
                for (uint32_t w = 0; w < sw; ++w) {
                    pack_tile<true>(d++, out_cb, (m0 + h) * N_block + n0 + w);
                }
            }
            tile_regs_release();
        }
    }
#endif
}
