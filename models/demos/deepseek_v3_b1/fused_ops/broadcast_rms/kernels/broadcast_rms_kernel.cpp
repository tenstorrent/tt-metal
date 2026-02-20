// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused Broadcast + RMSNorm unified kernel
// - NCRISC: Broadcast reader + RMSNorm reader
// - BRISC: Broadcast writer
// - TRISC: RMSNorm compute

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#if !defined(SKIP_CCL)
#include "../../../unified_kernels/broadcast.hpp"
#endif
#include "../../../unified_kernels/rmsnorm.hpp"

#if defined(COMPILE_FOR_NRISC)
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#endif

void kernel_main() {
    constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;

// -----------------------
// NCRISC: Broadcast reader + RMSNorm reader
// -----------------------
#if defined(COMPILE_FOR_NCRISC)

#if !defined(SKIP_CCL)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("cb0_id"),
        get_named_compile_time_arg_val("packet_size_in_pages"),
        get_named_compile_time_arg_val("tensor0_page_size"),
        get_named_compile_time_arg_val("is_sender"),
        get_named_compile_time_arg_val("core_noc_x"),
        get_named_compile_time_arg_val("core_noc_y"),
        get_named_compile_time_arg_val("is_secondary_sender")>;

    // Only read broadcast runtime args if CCL is enabled
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{};

    bcast_args = deepseek_b1_ops::Broadcast::ReaderArgs{
        get_common_arg_val<uint32_t>(0),  // tensor_address0
        get_common_arg_val<uint32_t>(1),  // tile_id_start
        get_common_arg_val<uint32_t>(2),  // tile_id_end
    };
#endif

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;

    // RMSNorm reader runtime args
    deepseek_b1_ops::RMSNorm::ReaderArgs rms_args{};

// -----------------------
// BRISC: Broadcast writer
// -----------------------
#elif defined(COMPILE_FOR_BRISC)
#if !defined(SKIP_CCL)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("cb0_id"),
        get_named_compile_time_arg_val("packet_size_in_pages"),
        get_named_compile_time_arg_val("tensor0_page_size"),
        get_named_compile_time_arg_val("num_targets_forward_direction"),
        get_named_compile_time_arg_val("num_targets_backward_direction"),
        get_named_compile_time_arg_val("is_sender"),
        get_named_compile_time_arg_val("core_noc_x"),
        get_named_compile_time_arg_val("core_noc_y"),
        get_named_compile_time_arg_val("is_secondary_sender"),
        get_named_compile_time_arg_val("has_secondary_target"),
        get_named_compile_time_arg_val("start_distance_in_hops_forward"),
        get_named_compile_time_arg_val("range_hops_forward"),
        get_named_compile_time_arg_val("start_distance_in_hops_backward"),
        get_named_compile_time_arg_val("range_hops_backward")>;

    deepseek_b1_ops::Broadcast::WriterArgs bcast_args{};

    bcast_args = deepseek_b1_ops::Broadcast::WriterArgs{
        get_common_arg_val<uint32_t>(0),   // tensor_address0
        get_common_arg_val<uint32_t>(1),   // out_ready_sem_bank_addr
        get_common_arg_val<uint32_t>(2),   // tile_id_start
        get_common_arg_val<uint32_t>(3),   // tile_id_end
        get_common_arg_val<uint32_t>(4),   // wait_output_semaphore
        get_common_arg_val<uint32_t>(5),   // reset_global_semaphore
        get_common_arg_val<uint32_t>(6),   // out_ready_sem_noc0_x
        get_common_arg_val<uint32_t>(7),   // out_ready_sem_noc0_y
        get_common_arg_val<uint32_t>(8),   // out_ready_sem_wait_value
        get_common_arg_val<uint32_t>(9),   // barrier_sem
        get_common_arg_val<uint32_t>(10),  // barrier_sem_noc0_x
        get_common_arg_val<uint32_t>(11),  // barrier_sem_noc0_y
        get_common_arg_val<uint32_t>(12),  // ring_index
        get_common_arg_val<uint32_t>(13),  // secondary_sync_sem
        get_common_arg_val<uint32_t>(14),  // num_connections
    };
#endif

        using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
        deepseek_b1_ops::RMSNorm::WriterArgs rms_args{};

// -----------------------
// TRISC: RMSNorm compute
// -----------------------
#elif defined(COMPILE_FOR_TRISC)

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb")>;

    deepseek_b1_ops::RMSNorm::ComputeArgs rms_args{
        get_common_arg_val<uint32_t>(0),  // epsilon (common runtime arg 0)
        get_common_arg_val<float>(1),     // scalar (1/N)
    };

#if !defined(SKIP_CCL)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ComputeCTArgs;
    deepseek_b1_ops::Broadcast::ComputeArgs bcast_args{};
#endif

    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);

#endif

    // CCL Broadcast
#if !defined(SKIP_CCL)
    deepseek_b1_ops::Broadcast::Op<BcastCTArgs, true> bcast;
    bcast(bcast_args);
#endif

#if defined(COMPILE_FOR_NCRISC)
    if constexpr (skip_ccl) {
        // Single-device: setup sharded buffer for input
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
    }
#endif

#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t intermediate_cb = get_named_compile_time_arg_val("intermediate_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

    if constexpr (skip_ccl) {
        // Single-device: only setup gamma buffer
        cb_reserve_back(gamma_cb, num_tiles);
        cb_push_back(gamma_cb, num_tiles);
    } else {
        // Multi-device: setup intermediate (broadcast output) and gamma buffers
        cb_reserve_back(intermediate_cb, num_tiles);
        cb_push_back(intermediate_cb, num_tiles);
        cb_reserve_back(gamma_cb, num_tiles);
        cb_push_back(gamma_cb, num_tiles);
    }
#endif

    // RMSNorm op
    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, true, true> rms;
    rms(rms_args);
}
