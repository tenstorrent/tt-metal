// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared expert eltwise program, compute (16-bit dest, as the replaced bf16 ops run): the instruction sequences of
//   unary eltwise_sfpu.cpp  silu:            copy_tile, silu_tile_init, silu_tile, pack            -> cb_silu
//   binary_ng eltwise_binary_sfpu_no_bcast:  copy_tile x2, mul_binary_tile (silu(gate) x up), pack -> cb_inter
// and on the scalar core
//   unary eltwise_sfpu.cpp  sigmoid:         copy_tile, sigmoid_tile_init<0>, sigmoid_tile<RC, 0>, pack -> cb_sig
//   binary_ng eltwise_binary_sfpu_col_bcast, first step: unary_bcast<COL> of the sigmoid tile, pack -> cb_sig_bcast
// (the multiply of the shared partial by that broadcast tile runs in the MoE post program).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    FUSED_ZONE("fz_se_c_main");
    const uint32_t has_scalar = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t cb_up = get_named_compile_time_arg_val("cb_up");
    constexpr uint32_t cb_silu = get_named_compile_time_arg_val("cb_silu");
    constexpr uint32_t cb_inter = get_named_compile_time_arg_val("cb_inter");
    constexpr uint32_t cb_scalar = get_named_compile_time_arg_val("cb_scalar");
    constexpr uint32_t cb_sig = get_named_compile_time_arg_val("cb_sig");
    constexpr uint32_t cb_sig_bcast = get_named_compile_time_arg_val("cb_sig_bcast");

    DataflowBuffer gate(cb_gate);
    DataflowBuffer up(cb_up);
    DataflowBuffer silu(cb_silu);
    DataflowBuffer inter(cb_inter);
    DataflowBuffer scalar(cb_scalar);
    DataflowBuffer sig(cb_sig);
    DataflowBuffer sig_bcast(cb_sig_bcast);

    // ---- eltwise_sfpu.cpp, SFPU_OP_CHAIN_0 = silu ----
    compute_kernel_hw_startup(cb_gate, cb_silu);
    copy_init(cb_gate);
    gate.wait_front(1);
    silu.reserve_back(1);
    tile_regs_acquire();
    copy_tile(cb_gate, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_silu);
    tile_regs_release();
    gate.pop_front(1);
    silu.push_back(1);

    // ---- eltwise_binary_sfpu_no_bcast.cpp, MUL ----
    silu.wait_front(1);
    up.wait_front(1);
    inter.reserve_back(1);
    mul_binary_tile_init();
    tile_regs_acquire();
    reconfig_data_format_srca(cb_up, cb_silu);
    copy_init(cb_silu);
    copy_tile(cb_silu, 0, 0);
    reconfig_data_format_srca(cb_silu, cb_up);
    copy_init(cb_up);
    copy_tile(cb_up, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_inter);
    pack_tile(0, cb_inter);
    tile_regs_release();
    silu.pop_front(1);
    up.pop_front(1);
    inter.push_back(1);

    if (has_scalar) {
        FUSED_ZONE("fz_se_c_sigmoid");
        // ---- eltwise_sfpu.cpp, SFPU_OP_CHAIN_0 = sigmoid (vector mode RC, accurate) ----
        scalar.wait_front(1);
        sig.reserve_back(1);
        reconfig_data_format_srca(cb_up, cb_scalar);
        copy_init(cb_scalar);
        tile_regs_acquire();
        copy_tile(cb_scalar, 0, 0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sig);
        pack_tile(0, cb_sig);
        tile_regs_release();
        scalar.pop_front(1);
        sig.push_back(1);

        // ---- eltwise_binary_sfpu_col_bcast.cpp: the broadcast operand through unary_bcast<COL> ----
        sig.wait_front(1);
        sig_bcast.reserve_back(1);
        pack_reconfig_data_format(cb_sig_bcast);
        compute_kernel_hw_startup(cb_sig, cb_sig_bcast);
        unary_bcast_init<BroadcastType::COL>(cb_sig);
        tile_regs_acquire();
        unary_bcast<BroadcastType::COL>(cb_sig, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sig_bcast);
        tile_regs_release();
        sig.pop_front(1);
        sig_bcast.push_back(1);
    }
}
