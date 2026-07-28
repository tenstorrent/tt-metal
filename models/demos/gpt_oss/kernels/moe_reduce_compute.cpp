// SPDX-License-Identifier: Apache-2.0
//
// gpt-oss custom fused MoE expert-reduce compute kernel.
//
// Computes, per output tile:
//     dst0 = sum_e  score_col[e] * act_tile[e]        (hardware MAC, col-broadcast)
//
// This is the gpt-oss decode expert tail  sum_e w[e] * down[e]  (the bias term
// bias*sum_e w[e] is added on the host side / by a cheap follow-up op).
//
// Adapted from
//   ttnn/.../deepseek_moe_fast_reduce_nc_fused/device/kernels/..._compute.cpp
// The reduce math (init_bcast ELWMUL/COL + acc_to_dest MAC) is IDENTICAL and was
// PCC-validated (TEST A: 0.99995 vs sum_e down[e]). The ONLY difference from the
// DeepSeek kernel is upstream: our reader packs score tiles with a DIRECT
// expert->column-0 mapping (expert e == score column e), with no all_to_all
// dispatch / expert_mapping gather. That removes the a2a byte-layout coupling
// that made the stock op mis-score on a single 1x1 device.

#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
constexpr uint32_t reduction_dim_size = get_compile_time_arg_val(1);  // num experts (E)
constexpr uint32_t input_granularity = get_compile_time_arg_val(2);
constexpr uint32_t compute_input_cb_id_0 = get_compile_time_arg_val(3);  // activations
constexpr uint32_t compute_input_cb_id_1 = get_compile_time_arg_val(4);  // score tiles
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(5);

void kernel_main() {
    CircularBuffer cb_in0(compute_input_cb_id_0);
    CircularBuffer cb_in1(compute_input_cb_id_1);
    CircularBuffer cb_out(compute_output_cb_id);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t num_input_tiles_iter = reduction_dim_size / input_granularity;

    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
        compute_input_cb_id_0, compute_input_cb_id_1, compute_output_cb_id);

    // acc_to_dest=1 => each mul_tiles_bcast_cols does dst0 += act * score (MAC)
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        compute_input_cb_id_0, compute_input_cb_id_1, 1 /*acc_to_dest*/)));

    reconfig_data_format(compute_input_cb_id_0, compute_input_cb_id_1);

    // Score tiles are pre-loaded once by the reader and stay resident.
    cb_in1.wait_front(reduction_dim_size);
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_in0.wait_front(input_granularity);
            for (uint32_t k = 0; k < input_granularity; ++k) {
                const uint32_t expert_tile = j * input_granularity + k;
                mul_tiles_bcast_cols(compute_input_cb_id_0, compute_input_cb_id_1, k, expert_tile, dst0);
            }
            cb_in0.pop_front(input_granularity);
        }
        tile_regs_commit();
        cb_out.reserve_back(one_tile);
        pack_reconfig_data_format(compute_output_cb_id);
        tile_regs_wait();
        pack_tile(dst0, compute_output_cb_id);
        tile_regs_release();
        cb_out.push_back(one_tile);
    }
    cb_in1.pop_front(reduction_dim_size);
}
