// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sampling unified kernel (k=1 argmax fast path) wrapper.
// Logic is implemented in unified_kernels/sampling.hpp so fused kernels can
// invoke the same operation in a composable style.

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
    using SamplingCTArgs = deepseek_b1_ops::Sampling::CTArgs<
        get_named_compile_time_arg_val("sampling_num_values"),
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
        get_named_compile_time_arg_val("sampling_sender_idx")>;

    constexpr uint32_t gather_cb = get_named_compile_time_arg_val("sampling_gather_cb");
    deepseek_b1_ops::Sampling::Args args{
        get_common_arg_val<uint32_t>(0),
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        get_common_arg_val<uint32_t>(6),
        get_common_arg_val<uint32_t>(7),
        get_write_ptr(gather_cb),
    };

    deepseek_b1_ops::Sampling::Op<SamplingCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
        sampling_op;
    sampling_op(args);
#elif defined(COMPILE_FOR_BRISC)
    using SamplingCTArgs = deepseek_b1_ops::Sampling::CTArgs<
        0,
        get_named_compile_time_arg_val("sampling_winner_page_bytes"),
        0,
        0,
        0,
        get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0>;

    deepseek_b1_ops::Sampling::Args args{
        0,
        0,
        0,
        get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        0,
        0,
        0,
    };

    deepseek_b1_ops::Sampling::Op<SamplingCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
        sampling_op;
    sampling_op(args);
#elif defined(COMPILE_FOR_TRISC)
    // No-op for k=1 argmax fast path.
#endif
}
