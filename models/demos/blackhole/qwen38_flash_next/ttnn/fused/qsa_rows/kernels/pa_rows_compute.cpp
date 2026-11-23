// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_rows program 5 compute (fp32 dest), one tile per core: the bf16-dest SFPU sigmoid of the gate tile packed bf16
// (matches unary sigmoid), then the SFPU product attention x sigmoid with the bf16-dest rounding and zero rule
// (matches binary_ng multiply), packed bf16.
// CBs: 0 gate, 1 attention, 2 sigmoid, 16 out (bf16, 1 each).
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_gate = 0, cb_att = 1, cb_sig = 2, cb_out = 16;
    compute_kernel_hw_startup(cb_gate, cb_att, cb_sig);
    CircularBuffer(cb_gate).wait_front(1);
    CircularBuffer(cb_att).wait_front(1);
    {
        reconfig_data_format(cb_gate, cb_gate);
        pack_reconfig_data_format(cb_sig);
        sigmoid_tile_init<false>();
        CircularBuffer(cb_sig).reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_gate);
        copy_tile(cb_gate, 0, 0);
        sigmoid_tile<VectorMode::RC, false, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sig);
        tile_regs_release();
        CircularBuffer(cb_sig).push_back(1);
        CircularBuffer(cb_gate).pop_front(1);
    }
    {
        CircularBuffer(cb_sig).wait_front(1);
        pack_reconfig_data_format(cb_out);
        mul_binary_tile_init();
        CircularBuffer(cb_out).reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_att);
        copy_tile(cb_att, 0, 0);
        copy_init(cb_sig);
        copy_tile(cb_sig, 0, 1);
        mul_binary_tile<false>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        CircularBuffer(cb_out).push_back(1);
        CircularBuffer(cb_sig).pop_front(1);
        CircularBuffer(cb_att).pop_front(1);
    }
}
