// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MLP_GU_FUSED split-SwiGLU (tt/mlp_swiglu.py), compute: out = silu(gate) * up per output tile.
//
// CB_IN holds (gate, up) tile pairs in order (swiglu_pair_reader.cpp); each pair becomes one CB_OUT tile. Both inputs are
// unpacked into DST, SiLU runs on the gate register and the product is an SFPU DST x DST multiply, so with
// fp32_dest_acc_en (the program's compute config) the whole silu(g) * u is fp32 and rounds to the output format once, at
// pack. This is the minimal_matmul FUSE_SWIGLU output stage (minimal_matmul/device/kernels/compute.cpp swiglu_block) run
// on a materialised [gate|up] matmul output.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    const uint32_t count = get_arg_val<uint32_t>(0);

    constexpr uint32_t GATE_DST = 0;
    constexpr uint32_t UP_DST = 1;
    CircularBuffer cb_in(CB_IN);
    CircularBuffer cb_out(CB_OUT);
    compute_kernel_hw_startup(CB_IN, CB_OUT);
    for (uint32_t i = 0; i < count; ++i) {
        cb_in.wait_front(2);
        cb_out.reserve_back(1);
        tile_regs_acquire();
        copy_init(CB_IN);
        copy_tile(CB_IN, 0, GATE_DST);
        copy_tile(CB_IN, 1, UP_DST);
        silu_tile_init();
        silu_tile(GATE_DST);
        mul_binary_tile_init();
        mul_binary_tile(GATE_DST, UP_DST, GATE_DST);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(GATE_DST, CB_OUT);
        tile_regs_release();
        cb_out.push_back(1);
        cb_in.pop_front(2);
    }
}
