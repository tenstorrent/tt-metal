// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
using namespace ckernel;

template <uint32_t A, uint32_t B, uint32_t Out, uint32_t Partial, uint32_t KBlock, uint32_t N, uint32_t K, uint32_t Subblock = 4>
void projection() {
    // Keep the baseline's BF16 partials and packer-L1 accumulation, including
    // its final reload before the last K block. No FP32 accumulation change.
    constexpr uint32_t blocks = K / KBlock;
    matmul_block_init(A, B, 0, Subblock, 1, KBlock);
    pack_reconfig_data_format(Partial);
    pack_reconfig_l1_acc(0);
    for (uint32_t block = 0; block < blocks; ++block) {
        cb_wait_front(A, KBlock);
        cb_wait_front(B, KBlock * N);
        const bool last = block == blocks - 1;
#if PROJECTION_HOIST_PACK
        // These settings persist across subblocks. Each setter drains the
        // packer, so change them only when accumulation/output mode changes.
        if (block == 1 || last) { pack_reconfig_l1_acc(!last); }
        if (last) { pack_reconfig_data_format(Out); }
#endif
        for (uint32_t n = 0; n < N; n += Subblock) {
            tile_regs_acquire();
            if (last) {
                reconfig_data_format_srca(B, Partial);
                copy_init(Partial);
                cb_wait_front(Partial, Subblock);
                copy_block(Partial, 0, 0, Subblock);
                cb_pop_front(Partial, Subblock);
                reconfig_data_format_srca(Partial, B);
                matmul_block_init(A, B, 0, Subblock, 1, KBlock);
            }
            for (uint32_t k = 0; k < KBlock; ++k) {
                matmul_block(A, B, k, k * N + n, 0, false, Subblock, 1, KBlock);
            }
            tile_regs_commit();
            const uint32_t destination = last ? Out : Partial;
            cb_reserve_back(destination, Subblock);
            tile_regs_wait();
#if !PROJECTION_HOIST_PACK
            pack_reconfig_data_format(destination);
            pack_reconfig_l1_acc(!last && block > 0);
#endif
            pack_block(0, destination, Subblock);
            tile_regs_release();
            cb_push_back(destination, Subblock);
        }
        if (block < blocks - 2) {
            for (uint32_t n = 0; n < N; n += Subblock) {
                cb_wait_front(Partial, Subblock);
                cb_pop_front(Partial, Subblock);
            }
        }
        cb_pop_front(A, KBlock);
        cb_pop_front(B, KBlock * N);
    }
    pack_reconfig_l1_acc(0);
}

