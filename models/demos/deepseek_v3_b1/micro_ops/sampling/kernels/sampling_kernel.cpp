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

#if defined(COMPILE_FOR_NCRISC)
FORCE_INLINE uint64_t get_safe_multicast_noc_addr(
    uint32_t noc_x_start,
    uint32_t noc_y_start,
    uint32_t noc_x_end,
    uint32_t noc_y_end,
    uint32_t addr,
    uint8_t noc = noc_index) {
    if (noc == 0) {
        return get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    } else {
        // For NOC 1, swap start and end coordinates
        return get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr, noc);
    }
}
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    uint32_t ncrisc_rt_arg_idx = 0;
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
        0xFFFFFFFF,
        0,
        0xFFFFFFFF,
        0,
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
        get_named_compile_time_arg_val("sampling_topk_out_indices_cb"),
        get_named_compile_time_arg_val("sampling_phase2_scores_byte_offset"),
        get_named_compile_time_arg_val("sampling_phase2_indices_byte_offset"),
        get_named_compile_time_arg_val("sampling_mesh_stage_scores_cb"),
        get_named_compile_time_arg_val("sampling_mesh_stage_indices_cb"),
        get_named_compile_time_arg_val("sampling_scores_scratch_stage2_offset"),
        get_named_compile_time_arg_val("sampling_indices_scratch_stage2_offset"),
        get_named_compile_time_arg_val("sampling_scores_scratch_addr"),
        get_named_compile_time_arg_val("sampling_indices_scratch_addr")>;

    deepseek_b1_ops::TopKSampling::ReaderArgs args{
        .scores_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .indices_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .output_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .final_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .final_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .global_sem_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .global_stage2_sem_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
    };

    deepseek_b1_ops::TopKSampling::
        Op<SamplingReaderCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;

#elif defined(COMPILE_FOR_BRISC)
    uint32_t brisc_rt_arg_idx = 0;
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
        get_named_compile_time_arg_val("sampling_topk_scores_slot_bytes"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_output_addr"),
        get_named_compile_time_arg_val("sampling_rand_output_addr"),
        get_named_compile_time_arg_val("sampling_inv_temp_bf16"),
        get_named_compile_time_arg_val("sampling_softmax_in_cb"),
        get_named_compile_time_arg_val("sampling_temp_cb"),
        0,
        get_named_compile_time_arg_val("sampling_enable_metadata"),
        get_named_compile_time_arg_val("sampling_copy_probabilities"),
        get_named_compile_time_arg_val("sampling_metadata_address")>;

    deepseek_b1_ops::TopKSampling::WriterArgs args{
        .final_noc_x = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .final_noc_y = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
    };

    deepseek_b1_ops::TopKSampling::
        Op<SamplingWriterCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;

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
        get_named_compile_time_arg_val("sampling_stage1_receiver"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_num_values"),
        get_named_compile_time_arg_val("sampling_num_senders"),
        get_named_compile_time_arg_val("sampling_topk_in_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_in_indices_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_indices_cb"),
        get_named_compile_time_arg_val("sampling_mesh_stage_scores_cb"),
        get_named_compile_time_arg_val("sampling_mesh_stage_indices_cb"),
        get_named_compile_time_arg_val("sampling_stage1_row_elements"),
        get_named_compile_time_arg_val("sampling_stage1_num_input_tiles"),
        get_named_compile_time_arg_val("sampling_stage2_row_elements"),
        get_named_compile_time_arg_val("sampling_stage2_num_input_tiles")>;
    deepseek_b1_ops::TopKSampling::ComputeArgs args{};
    deepseek_b1_ops::TopKSampling::
        Op<SamplingComputeCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;

    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(ckernel::SFPU_FPU, 0, 1));
    if constexpr (SamplingComputeCTArgs::topk_k == 32) {
        deepseek_compute_kernel_hw_startup<true>(
            SamplingComputeCTArgs::topk_in_scores_cb,
            SamplingComputeCTArgs::topk_in_scores_cb,
            SamplingComputeCTArgs::topk_out_scores_cb);
    } else {
        deepseek_compute_kernel_hw_startup<true>(
            SamplingComputeCTArgs::softmax_in_cb,
            SamplingComputeCTArgs::softmax_in_cb,
            SamplingComputeCTArgs::softmax_out_cb);
    }

    sampling_op.set_seed(get_named_compile_time_arg_val("sampling_seed"));
#endif

    constexpr uint32_t num_internal_iterations = get_named_compile_time_arg_val("sampling_num_internal_iterations");
    for (uint32_t i = 0; i < num_internal_iterations; ++i) {
        sampling_op(args);
        if (num_internal_iterations == 1) {
            break;
        }
#if defined(COMPILE_FOR_NCRISC)
        // Single-device loop barrier: final core releases non-final cores for next iteration.
        // This prevents receiver semaphore increments from later iterations racing ahead.
        if constexpr (!SamplingReaderCTArgs::mesh_mode) {
            const uint32_t local_ready_sem_addr =
                get_semaphore(get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"));
            auto local_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_ready_sem_addr);
            if constexpr (Core::is_final_core) {
                constexpr uint32_t num_dests = get_named_compile_time_arg_val("sampling_loop_num_dests");
                if constexpr (num_dests > 0) {
                    constexpr uint32_t mcast_start_x = get_named_compile_time_arg_val("sampling_loop_mcast_start_x");
                    constexpr uint32_t mcast_start_y = get_named_compile_time_arg_val("sampling_loop_mcast_start_y");
                    constexpr uint32_t mcast_end_x = get_named_compile_time_arg_val("sampling_loop_mcast_end_x");
                    constexpr uint32_t mcast_end_y = get_named_compile_time_arg_val("sampling_loop_mcast_end_y");
                    const uint64_t mcast_noc_addr =
                        get_safe_multicast_noc_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 0);
                    const uint64_t mcast_sem_addr = mcast_noc_addr | local_ready_sem_addr;

                    noc_semaphore_set(local_ready_sem_ptr, 1);
                    noc_semaphore_set_multicast(local_ready_sem_addr, mcast_sem_addr, num_dests);
                    noc_async_write_barrier();
                    noc_semaphore_set(local_ready_sem_ptr, 0);
                }
            } else {
                noc_semaphore_wait(local_ready_sem_ptr, 1);
                noc_semaphore_set(local_ready_sem_ptr, 0);
            }
        }
#endif
    }
}
