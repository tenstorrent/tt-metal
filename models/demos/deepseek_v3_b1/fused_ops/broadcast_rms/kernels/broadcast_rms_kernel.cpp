// filepath:
// /localdev/nardo/generic/tt-metal/models/demos/deepseek_v3_b1/fused_ops/broadcast_rms/kernels/broadcast_rms_kernel.cpp

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused Broadcast + RMSNorm unified kernel
// - NCRISC: Broadcast reader only
// - BRISC: Broadcast writer + RMSNorm reader
// - TRISC: RMSNorm compute

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/broadcast.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#endif

// Note: avoid declaring namespace-scope constexpr flags that call
// get_named_compile_time_arg_val("...") here. Doing so can trigger
// constexpr lookup of CTArgs in compilation units that don't provide
// the named compile-time args and leads to __builtin_unreachable
// errors. Instead, query named CTArgs directly inside the RISC
// branches where they are required.

void kernel_main() {
// -----------------------
// NCRISC: Broadcast reader
// Expected Named CTArgs (examples):
//   - cb0_id, packet_size_in_pages, tensor0_page_size
//   - is_sender, core_noc_x, core_noc_y, is_secondary_sender, is_active_broadcaster
// Runtime args (per-core):
//   1: tensor_address0
//   2: tile_id_start
//   3: tile_id_end
// -----------------------
#if defined(COMPILE_FOR_NCRISC)
    // Instantiate ReaderCTArgs with named compile-time args (must match broadcast.hpp template params)
    using ReaderCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("cb0_id"),
        get_named_compile_time_arg_val("packet_size_in_pages"),
        get_named_compile_time_arg_val("tensor0_page_size"),
        get_named_compile_time_arg_val("is_sender"),
        get_named_compile_time_arg_val("core_noc_x"),
        get_named_compile_time_arg_val("core_noc_y"),
        get_named_compile_time_arg_val("is_secondary_sender"),
        get_named_compile_time_arg_val("is_active_broadcaster")>;

    // For fused op on NCRISC, layout runtime args so: [tensor_address0, tile_id_start, tile_id_end]
    // Note: RMSNorm reader will run on BRISC; NCRISC only performs the broadcast read into the CB.
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_reader_args{
        get_arg_val<uint32_t>(1),  // tensor_address0
        get_arg_val<uint32_t>(2),  // tile_id_start
        get_arg_val<uint32_t>(3),  // tile_id_end
    };

    {
        DeviceZoneScopedN("BROADCAST_READER");
        // Perform broadcast reader to load data into CB
        deepseek_b1_ops::Broadcast::Op<ReaderCTArgs, true> bcast;
        bcast(bcast_reader_args);
    }

// -----------------------
// BRISC: Broadcast writer + RMSNorm reader
// Expected Named CTArgs (examples):
//   - writer CTArgs: cb0_id, packet_size_in_pages, tensor0_page_size, num_targets_*, core_noc_x/y,
//   has_secondary_target, etc.
// Runtime args writer (per-core):
//   0: tensor_address0
//   1: out_ready_sem_bank_addr
//   2: tile_id_start
//   3: tile_id_end
//   ... additional semaphore / routing args appended by op.py via setup_routing_plane_connection
//   15: rmsnorm scalar (or semaphore index) <-- BRISC must be provided this at runtime
// -----------------------
#elif defined(COMPILE_FOR_BRISC)
    // Instantiate WriterCTArgs with named compile-time args
    using WriterCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
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
        get_named_compile_time_arg_val("has_reverse_secondary_connection"),
        get_named_compile_time_arg_val("start_distance_in_hops_forward"),
        get_named_compile_time_arg_val("range_hops_forward"),
        get_named_compile_time_arg_val("start_distance_in_hops_backward"),
        get_named_compile_time_arg_val("range_hops_backward"),
        get_named_compile_time_arg_val("using_persistent_buffers")>;

    // Writer runtime args correspond to WriterArgs struct in broadcast.hpp (all uint32_t)
    deepseek_b1_ops::Broadcast::WriterArgs bcast_writer_args{
        get_arg_val<uint32_t>(0),   // tensor_address0
        get_arg_val<uint32_t>(1),   // out_ready_sem_bank_addr
        get_arg_val<uint32_t>(2),   // tile_id_start
        get_arg_val<uint32_t>(3),   // tile_id_end
        get_arg_val<uint32_t>(4),   // wait_output_semaphore
        get_arg_val<uint32_t>(5),   // reset_global_semaphore
        get_arg_val<uint32_t>(6),   // out_ready_sem_noc0_x
        get_arg_val<uint32_t>(7),   // out_ready_sem_noc0_y
        get_arg_val<uint32_t>(8),   // out_ready_sem_wait_value
        get_arg_val<uint32_t>(9),   // barrier_sem
        get_arg_val<uint32_t>(10),  // barrier_sem_noc0_x
        get_arg_val<uint32_t>(11),  // barrier_sem_noc0_y
        get_arg_val<uint32_t>(12),  // ring_index
        get_arg_val<uint32_t>(13),  // secondary_sync_sem
        get_arg_val<uint32_t>(14),  // num_connections
    };

    {
        DeviceZoneScopedN("BROADCAST_WRITER");
        // Broadcast writer: send data out
        deepseek_b1_ops::Broadcast::Op<WriterCTArgs, true> bcast_writer;
        bcast_writer(bcast_writer_args);
    }

    // Immediately after the writer completes, run the RMSNorm reader on BRISC
    // so the writer path both sends data and prepares RMSNorm scalars.
    // Host must provide the named CTArgs and runtime args used below (see op.py).
    {
        // Query CTArgs locally
        constexpr uint32_t rmsnorm_faces = get_named_compile_time_arg_val("rmsnorm_num_faces");
        constexpr uint32_t scalars_cb = get_named_compile_time_arg_val("rmsnorm_scalars_cb");

        // Runtime scalar packed is expected at RTArg index 15 for BRISC
        uint32_t scalar_packed = get_arg_val<uint32_t>(15);

        // Call the reduction scalar generator directly on BRISC
        // generate_reduce_scaler is a device helper that writes the packed
        // scalar into the scalars CB. Use the compile-time faces parameter.
        // Use the writer-side helper (no template 'faces' param) which is
        // safe and available for BRISC builds. Use needs_zeroing=true for
        // correctness in generic cases.
        wh_generate_reduce_scaler<true>(scalars_cb, scalar_packed);
    }

// -----------------------
// TRISC: RMSNorm compute
// Expected Named CTArgs: rmsnorm_fp32_acc, rmsnorm_num_tiles, rmsnorm_rsqrt_fast_approx
// Runtime args: (passed as compile-time for many of these; compute args typically use CTArgs for CB indices)
// -----------------------
#elif defined(COMPILE_FOR_TRISC)
    using RMSNormComputeCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1>;

    deepseek_b1_ops::RMSNorm::ComputeArgs rms_compute_args{
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_named_compile_time_arg_val("rmsnorm_interm_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_arg_val<uint32_t>(0),  // epsilon (runtime arg 0)
    };

    // Note: avoid calling unified_kernels::setup_sharded_buffer here; the
    // helper may not be available in this build context. The necessary
    // buffers should be created by the host (Python) side via CB
    // descriptors. Instantiate the RMSNorm compute op, querying the
    // 'is_active_core' named CTArg directly in the template parameter.
    {
        DeviceZoneScopedN("RMSNORM_COMPUTE");
        deepseek_b1_ops::RMSNorm::
            Op<RMSNormComputeCTArgs, (get_named_compile_time_arg_val("is_active_core") == 1), true>
                rms_compute;
        rms_compute(rms_compute_args);
    }
#endif
}
