// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// post_attention compute (fp32 dest): the bf16-dest SFPU sigmoid of each gate tile packed bf16 (matches unary
// sigmoid), then the SFPU product attention x sigmoid with the bf16-dest rounding and zero rule (matches binary_ng
// multiply), packed bf16.  CBs: 0 gate, 1 attention, 2 sigmoid, 16 out (bf16, 8 each).

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_gate = 0, cb_att = 1, cb_sig = 2, cb_out = 16, head_tiles = 8, group_tiles = 4;
    compute_kernel_hw_startup(cb_gate, cb_att, cb_sig);
    CircularBuffer(cb_gate).wait_front(head_tiles);
    CircularBuffer(cb_att).wait_front(head_tiles);

    {
        reconfig_data_format(cb_gate, cb_gate);
        pack_reconfig_data_format(cb_sig);
        sigmoid_tile_init<false>();
        for (uint32_t b = 0; b < head_tiles; b += group_tiles) {
            CircularBuffer(cb_sig).reserve_back(group_tiles);
            tile_regs_acquire();
            copy_init(cb_gate);
            for (uint32_t i = 0; i < group_tiles; ++i) {
                copy_tile(cb_gate, b + i, i);
                sigmoid_tile<VectorMode::RC, false, false>(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < group_tiles; ++i) {
                pack_tile(i, cb_sig);
            }
            tile_regs_release();
            CircularBuffer(cb_sig).push_back(group_tiles);
        }
        CircularBuffer(cb_gate).pop_front(head_tiles);
    }

    {
        CircularBuffer(cb_sig).wait_front(head_tiles);
        pack_reconfig_data_format(cb_out);
        mul_binary_tile_init();
        for (uint32_t b = 0; b < head_tiles; b += 2) {
            CircularBuffer(cb_out).reserve_back(2);
            tile_regs_acquire();
            copy_init(cb_att);
            copy_tile(cb_att, b, 0);
            copy_tile(cb_att, b + 1, 2);
            copy_init(cb_sig);
            copy_tile(cb_sig, b, 1);
            mul_binary_tile<false>(0, 1, 0);
            copy_tile(cb_sig, b + 1, 3);
            mul_binary_tile<false>(2, 3, 2);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            pack_tile(2, cb_out);
            tile_regs_release();
            CircularBuffer(cb_out).push_back(2);
        }
        CircularBuffer(cb_sig).pop_front(head_tiles);
        CircularBuffer(cb_att).pop_front(head_tiles);
    }
}
