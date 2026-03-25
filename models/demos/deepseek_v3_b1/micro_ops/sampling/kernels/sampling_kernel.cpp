// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Top-K sampling unified kernel (k>1 path) wrapper.
// All logic lives in unified_kernels/sampling.hpp.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/sampling.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("sampling_is_active_core") == 1;
    static constexpr bool is_final_core = get_named_compile_time_arg_val("sampling_is_final_core") == 1;
    static constexpr bool is_mesh_sender_core = get_named_compile_time_arg_val("sampling_mesh_sender_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using SamplingReaderCTArgs = deepseek_b1_ops::TopKSampling::ReaderCTArgs<
        get_named_compile_time_arg_val("sampling_num_values"),
        get_named_compile_time_arg_val("sampling_topk_k"),
        get_named_compile_time_arg_val("sampling_winner_page_bytes"),
        get_named_compile_time_arg_val("sampling_num_senders"),
        get_named_compile_time_arg_val("sampling_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_receiver_semaphore_id"),
        get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage1_sender"),
        get_named_compile_time_arg_val("sampling_stage1_receiver"),
        get_named_compile_time_arg_val("sampling_stage2_sender"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_stage1_slot_base_offset"),
        get_named_compile_time_arg_val("sampling_stage1_num_slots"),
        get_named_compile_time_arg_val("sampling_stage1_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_stage1_local_slot_offset"),
        get_named_compile_time_arg_val("sampling_stage2_slot_base_offset"),
        get_named_compile_time_arg_val("sampling_stage2_num_slots"),
        get_named_compile_time_arg_val("sampling_stage2_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_stage2_local_slot_offset"),
        get_named_compile_time_arg_val("sampling_mesh_local_send_slot_offset"),
        get_named_compile_time_arg_val("sampling_sender_idx"),
        0,
        0,
        0,
        0xFFFFFFFF,
        0,
        get_named_compile_time_arg_val("sampling_gather_cb"),
        get_named_compile_time_arg_val("sampling_winner_cb"),
        get_named_compile_time_arg_val("sampling_softmax_in_cb"),
        get_named_compile_time_arg_val("sampling_softmax_out_cb"),
        get_named_compile_time_arg_val("sampling_softmax_exp_cb"),
        get_named_compile_time_arg_val("sampling_scaler_cb"),
        get_named_compile_time_arg_val("sampling_temp_cb"),
        get_named_compile_time_arg_val("sampling_inv_temp_bf16"),
        get_named_compile_time_arg_val("sampling_topk_in_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_in_indices_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_indices_cb")>;

    deepseek_b1_ops::TopKSampling::ReaderArgs args{
        .scores_addr = get_common_arg_val<uint32_t>(0),
        .indices_addr = get_common_arg_val<uint32_t>(1),
        .output_addr = get_common_arg_val<uint32_t>(2),
        .final_noc_x = get_common_arg_val<uint32_t>(3),
        .final_noc_y = get_common_arg_val<uint32_t>(4),
        .scratch_addr = get_common_arg_val<uint32_t>(5),
        .global_sem_addr = get_common_arg_val<uint32_t>(6),
        .global_stage2_sem_addr = get_common_arg_val<uint32_t>(7),
        .gather_addr = 0,
    };

    deepseek_b1_ops::TopKSampling::
        Op<SamplingReaderCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;
    sampling_op(args);

#elif defined(COMPILE_FOR_BRISC)
    using SamplingWriterCTArgs = deepseek_b1_ops::TopKSampling::WriterCTArgs<
        get_named_compile_time_arg_val("sampling_winner_page_bytes"),
        get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"),
        0,
        0,
        0,
        get_named_compile_time_arg_val("sampling_topk_k"),
        get_named_compile_time_arg_val("sampling_softmax_out_cb"),
        get_named_compile_time_arg_val("sampling_rand_cb"),
        get_named_compile_time_arg_val("sampling_winner_cb"),
        get_named_compile_time_arg_val("sampling_p_bf16"),
        get_named_compile_time_arg_val("sampling_topk_scores_stride"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage2_receiver")>;

    deepseek_b1_ops::TopKSampling::WriterArgs args{
        .final_noc_x = get_common_arg_val<uint32_t>(0),
        .final_noc_y = get_common_arg_val<uint32_t>(1),
        .scratch_addr = get_common_arg_val<uint32_t>(2),
        .output_addr = get_common_arg_val<uint32_t>(3),
        .rand_output_addr = get_common_arg_val<uint32_t>(4),
    };

    deepseek_b1_ops::TopKSampling::
        Op<SamplingWriterCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;
    sampling_op(args);

#elif defined(COMPILE_FOR_TRISC)
    using SamplingComputeCTArgs = deepseek_b1_ops::TopKSampling::ComputeCTArgs<
        get_named_compile_time_arg_val("sampling_softmax_in_cb"),
        get_named_compile_time_arg_val("sampling_softmax_out_cb"),
        get_named_compile_time_arg_val("sampling_softmax_exp_cb"),
        get_named_compile_time_arg_val("sampling_softmax_sub_cb"),
        get_named_compile_time_arg_val("sampling_max_cb"),
        get_named_compile_time_arg_val("sampling_sum_cb"),
        get_named_compile_time_arg_val("sampling_scaler_cb"),
        get_named_compile_time_arg_val("sampling_temp_cb"),
        get_named_compile_time_arg_val("sampling_rand_cb"),
        get_named_compile_time_arg_val("sampling_seed"),
        get_named_compile_time_arg_val("sampling_topk_k"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_num_values"),
        get_named_compile_time_arg_val("sampling_topk_in_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_in_indices_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_indices_cb")>;
    deepseek_b1_ops::TopKSampling::ComputeArgs args{};
    deepseek_b1_ops::TopKSampling::
        Op<SamplingComputeCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;

    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(ckernel::SFPU_FPU, 0, 1));
    if constexpr (SamplingComputeCTArgs::topk_k == 32) {
        deepseek_compute_kernel_hw_startup<true>(
            SamplingComputeCTArgs::topk_in_scores_cb,
            SamplingComputeCTArgs::topk_in_indices_cb,
            SamplingComputeCTArgs::topk_out_scores_cb);
    } else {
        deepseek_compute_kernel_hw_startup<true>(0, 0, 0);
    }

    sampling_op(args);
#endif
}
