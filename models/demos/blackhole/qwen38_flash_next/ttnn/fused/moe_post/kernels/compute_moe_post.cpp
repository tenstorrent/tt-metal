// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE post program, compute (one output column tile): the replaced ops' instruction sequences on CBs of their data
// formats, so the values are the composed chain's:
//   deepseek_moe_fast_reduce_nc_fused_compute.cpp: FPU ELWMUL COL-broadcast MAC into the 32-bit dest, acc_to_dest,
//     slot order 0..top_k-1 (the unowned slots multiply the zero tile by a zero score exactly as the chain does),
//     one bf16 pack                                                                        -> cb_routed
//   binary_ng eltwise_binary_sfpu_scalar_bcast / _col_bcast (has_sig): copy_tile x2, mul_binary_tile in the 16-bit
//     dest rounding = the shared partial times its column-broadcast sigmoid                 -> cb_gated
//   binary_ng eltwise_binary_no_bcast (ttnn.add defaults fast_and_approximate_mode=True): FPU add_tiles of the
//     routed and the shared tile = routed + shared                                          -> cb_out
// The kernel runs with the 32-bit dest the MAC needs.  The SFPU multiply takes the 16-bit-dest template argument, so
// its software rounding to bf16 is the chain's and the pack of the bf16-valued fp32 is exact.  The chain's FPU add
// rounds its sum to bf16 on the 16-bit dest write; here the exact sum of two bf16 values (at most 24 significant
// bits) lands in the 32-bit dest and the packer rounds it to bf16 with the same rule.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t cb_act = get_named_compile_time_arg_val("cb_act");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_routed = get_named_compile_time_arg_val("cb_routed");
    constexpr uint32_t cb_shared = get_named_compile_time_arg_val("cb_shared");
    constexpr uint32_t cb_sig = get_named_compile_time_arg_val("cb_sig");
    constexpr uint32_t cb_gated = get_named_compile_time_arg_val("cb_gated");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t has_sig = get_named_compile_time_arg_val("has_sig");
    constexpr uint32_t cb_rhs = has_sig ? cb_gated : cb_shared;  // the add's right operand

    DataflowBuffer act(cb_act);
    DataflowBuffer scores(cb_scores);
    DataflowBuffer routed(cb_routed);
    DataflowBuffer shared(cb_shared);
    DataflowBuffer sig(cb_sig);
    DataflowBuffer gated(cb_gated);
    DataflowBuffer out(cb_out);

    // ---- deepseek_moe_fast_reduce_nc_fused_compute.cpp ----
    compute_kernel_hw_startup(cb_act, cb_scores, cb_routed);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_act, cb_scores);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_act, cb_scores, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_act, cb_scores);
    {
        FUSED_ZONE("fz_mp_c_wait");
        scores.wait_front(top_k);
        act.wait_front(top_k);
    }
    {
        FUSED_ZONE("fz_mp_c_mac");
        tile_regs_acquire();
        for (uint32_t k = 0; k < top_k; ++k) {
            mul_tiles_bcast_cols(cb_act, cb_scores, k, k, 0);
        }
        tile_regs_commit();
        routed.reserve_back(1);
        pack_reconfig_data_format(cb_routed);
        tile_regs_wait();
        pack_tile(0, cb_routed);
        tile_regs_release();
        routed.push_back(1);
        act.pop_front(top_k);
        scores.pop_front(top_k);
    }

    {
        FUSED_ZONE("fz_mp_c_sig");
        // ---- binary_ng SFPU multiply of the ungated shared partial by its broadcast sigmoid (has_sig) ----
        shared.wait_front(1);
        if constexpr (has_sig) {
            sig.wait_front(1);
            gated.reserve_back(1);
            mul_binary_tile_init();
            tile_regs_acquire();
            reconfig_data_format_srca(cb_act, cb_shared);
            copy_init(cb_shared);
            copy_tile(cb_shared, 0, 0);
            reconfig_data_format_srca(cb_shared, cb_sig);
            copy_init(cb_sig);
            copy_tile(cb_sig, 0, 1);
            mul_binary_tile<false>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_gated);
            pack_tile(0, cb_gated);
            tile_regs_release();
            gated.push_back(1);
            shared.pop_front(1);
            sig.pop_front(1);
            gated.wait_front(1);
        }
    }

    {
        FUSED_ZONE("fz_mp_c_add");
        // ---- binary_ng eltwise_binary_no_bcast.cpp: FPU ELWADD of the routed and the shared tile ----
        routed.wait_front(1);
        out.reserve_back(1);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_routed, cb_rhs);
        reconfig_data_format(cb_routed, cb_rhs);
        tile_regs_acquire();
        add_tiles(cb_routed, cb_rhs, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        routed.pop_front(1);
        if constexpr (has_sig) {
            gated.pop_front(1);
        } else {
            shared.pop_front(1);
        }
    }
}
