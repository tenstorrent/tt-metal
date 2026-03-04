// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused Broadcast + RMSNorm unified kernel
// - NCRISC: Broadcast writer + RMSNorm reader
// - BRISC: Broadcast reader
// - TRISC: RMSNorm compute

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#if !defined(SKIP_CCL) || defined(ENABLE_SOCKET_READER)
#include "../../../unified_kernels/broadcast.hpp"
#endif
#include "../../../unified_kernels/rmsnorm.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#endif

void kernel_main() {
    constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;

// -----------------------
// NCRISC: Broadcast writer
// -----------------------
#if defined(COMPILE_FOR_NCRISC)
#if !defined(SKIP_CCL)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_tensor0_page_size"),
        get_named_compile_time_arg_val("bcast_num_neighbors"),
        get_named_compile_time_arg_val("bcast_num_links"),
        get_named_compile_time_arg_val("bcast_is_root"),
        get_named_compile_time_arg_val("bcast_chunk_size_bytes"),
        get_named_compile_time_arg_val("bcast_last_chunk_size_bytes"),
        get_named_compile_time_arg_val("bcast_num_chunks")>;

    deepseek_b1_ops::Broadcast::WriterArgs bcast_args{
        get_common_arg_val<uint32_t>(0),    // tensor_address0
        get_common_arg_val<uint32_t>(1),    // my_noc_x
        get_common_arg_val<uint32_t>(2),    // my_noc_y
        {get_common_arg_val<uint32_t>(3),   // sem_bank_addrs[0]
         get_common_arg_val<uint32_t>(4)},  // sem_bank_addrs[1]
    };
#endif
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;

    // RMSNorm reader runtime args
    deepseek_b1_ops::RMSNorm::ReaderArgs rms_args{};

#endif

// -----------------------
// BRISC: Broadcast reader
// -----------------------
#if defined(COMPILE_FOR_BRISC)

#if !defined(SKIP_CCL) || defined(ENABLE_SOCKET_READER)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender"),
        get_named_compile_time_arg_val("bcast_use_socket")>;

    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{
        get_common_arg_val<uint32_t>(0),  // socket_config_addr
        get_common_arg_val<uint32_t>(1),  // socket_page_size
        get_common_arg_val<uint32_t>(2),  // socket_num_pages
    };
#endif

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    deepseek_b1_ops::RMSNorm::WriterArgs rms_args{};

#endif

// -----------------------
// TRISC: RMSNorm compute
// -----------------------
#if defined(COMPILE_FOR_TRISC)

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

    deepseek_compute_kernel_init();

#endif

    // CCL Broadcast: runs on all cores in normal mode.
    // In socket-reader + skip_ccl mode, only BRISC executes this path (socket recv via broadcast reader).
#if !defined(SKIP_CCL) || (defined(ENABLE_SOCKET_READER) && defined(COMPILE_FOR_BRISC))
    deepseek_b1_ops::Broadcast::Op<BcastCTArgs, true> bcast;
    bcast(bcast_args);
#endif

#if defined(COMPILE_FOR_NCRISC)
    constexpr bool use_socket = get_named_compile_time_arg_val("use_socket") == 1;
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");

    if constexpr (skip_ccl) {
        // In socket mode BRISC signals rmsnorm_input_cb readiness via cb_push_back after socket recv;
        // NCRISC must not call setup_sharded_buffer on it or it will double-signal the CB.
        if constexpr (!use_socket) {
            constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
            unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        }
        unified_kernels::setup_sharded_buffer(gamma_cb, rmsnorm_num_tiles);
    } else {
        constexpr uint32_t intermediate_cb = get_named_compile_time_arg_val("intermediate_cb");
        unified_kernels::setup_sharded_buffer(intermediate_cb, rmsnorm_num_tiles);
        unified_kernels::setup_sharded_buffer(gamma_cb, rmsnorm_num_tiles);
    }
#endif

    // RMSNorm op
    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, true, true> rms;
    rms(rms_args);
}
