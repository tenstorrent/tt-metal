// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Unified Reduce-to-Root Kernel for 4x2 mesh 3-level reduction tree.
 *
 * Uses the unified ReduceToOneB1 op from unified_kernels/reduce_to_one_b1.hpp
 */

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/reduce_to_one_b1.hpp"

void kernel_main() {
    using ReduceToOne = deepseek_b1_ops::ReduceToOneB1;

#if defined(COMPILE_FOR_NCRISC)
    // Reader CTArgs
    using CTArgs = ReduceToOne::ReaderCTArgs<
        get_named_compile_time_arg_val("device_role"),
        get_named_compile_time_arg_val("num_tiles"),
        get_named_compile_time_arg_val("local_cb"),
        get_named_compile_time_arg_val("received_cb_r1"),
        get_named_compile_time_arg_val("received_cb_r2"),
        get_named_compile_time_arg_val("received_cb_r3"),
        get_named_compile_time_arg_val("is_fabric_core")>;

    // Reader runtime args (from common args)
    ReduceToOne::ReaderArgs rt_args{
        get_common_arg_val<uint32_t>(0),  // recv_sem_round1
        get_common_arg_val<uint32_t>(1),  // recv_sem_round2
        get_common_arg_val<uint32_t>(2),  // recv_sem_round3
    };

#elif defined(COMPILE_FOR_BRISC)
    // Writer CTArgs
    using CTArgs = ReduceToOne::WriterCTArgs<
        get_named_compile_time_arg_val("device_role"),
        get_named_compile_time_arg_val("num_tiles"),
        get_named_compile_time_arg_val("payload_size_bytes"),
        get_named_compile_time_arg_val("local_cb"),
        get_named_compile_time_arg_val("scratch_cb"),
        get_named_compile_time_arg_val("packet_cb"),
        get_named_compile_time_arg_val("packet_header_cb"),
        get_named_compile_time_arg_val("num_hops"),
        get_named_compile_time_arg_val("dst_fabric_node_chip_id"),
        get_named_compile_time_arg_val("dst_fabric_node_mesh_id"),
        get_named_compile_time_arg_val("output_core_noc_x"),
        get_named_compile_time_arg_val("output_core_noc_y"),
        get_named_compile_time_arg_val("num_workers"),
        get_named_compile_time_arg_val("slot_size_bytes"),
        get_named_compile_time_arg_val("is_fabric_core")>;

    // Writer runtime args for worker cores only (from per-core args)
    // Fabric cores have different args (sem IDs + fabric connection) read inside the op
    ReduceToOne::WorkerWriterArgs rt_args{};
    if constexpr (CTArgs::is_fabric_core == 0) {
        rt_args = {
            get_arg_val<uint32_t>(0),  // fabric_core_noc_x
            get_arg_val<uint32_t>(1),  // fabric_core_noc_y
            get_arg_val<uint32_t>(2),  // my_slot_idx
            get_arg_val<uint32_t>(3),  // worker_sem_id
            get_arg_val<uint32_t>(4),  // dst_l1_addr
            get_arg_val<uint32_t>(5),  // dst_sem_addr
            get_arg_val<uint32_t>(6),  // output_base_addr
            get_arg_val<uint32_t>(7),  // shard_idx
            get_arg_val<uint32_t>(8),  // socket_config_addr
        };
    }

#elif defined(COMPILE_FOR_TRISC)
    // Compute CTArgs
    using CTArgs = ReduceToOne::ComputeCTArgs<
        get_named_compile_time_arg_val("device_role"),
        get_named_compile_time_arg_val("num_tiles"),
        get_named_compile_time_arg_val("local_cb"),
        get_named_compile_time_arg_val("received_cb_r1"),
        get_named_compile_time_arg_val("received_cb_r2"),
        get_named_compile_time_arg_val("received_cb_r3"),
        get_named_compile_time_arg_val("output_cb"),
        get_named_compile_time_arg_val("scratch_cb"),
        get_named_compile_time_arg_val("is_fabric_core")>;

    // Compute has no runtime args
    ReduceToOne::ComputeArgs rt_args{};
    deepseek_compute_kernel_init();
#endif

    // Execute the op (looped for testing iteration correctness)
    // IsWorkerCore = true (compile-time) since fabric core logic is handled inside the op
    constexpr uint32_t num_loop_iters = get_named_compile_time_arg_val("num_loop_iters");
    ReduceToOne::Op<CTArgs, true> op;
    for (uint32_t iter = 0; iter < num_loop_iters; ++iter) {
        op(rt_args);
    }
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
