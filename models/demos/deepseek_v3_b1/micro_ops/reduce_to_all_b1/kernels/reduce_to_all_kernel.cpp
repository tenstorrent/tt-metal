// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Unified Reduce-to-All Kernel — Ring + Cross-Column algorithm: similar to reduce to one b1 kernel but all 8 devices
 * have the final output. Phase 1: FC BRISC=FWD, FC NCRISC=BWD (matching sdpa_reduce_to_all). Phase 2: FC BRISC closes
 * FWD, opens cross-column conn, forwards R3.
 */

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/reduce_to_all_b1.hpp"

void kernel_main() {
    using ReduceToAll = deepseek_b1_ops::ReduceToAllB1;

#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = ReduceToAll::ReaderCTArgs<
        get_named_compile_time_arg_val("num_tiles"),
        get_named_compile_time_arg_val("local_cb"),
        get_named_compile_time_arg_val("received_cb"),
        get_named_compile_time_arg_val("is_fabric_core"),
        get_named_compile_time_arg_val("slots_per_direction"),
        get_named_compile_time_arg_val("slot_size_bytes"),
        get_named_compile_time_arg_val("packet_cb"),
        get_named_compile_time_arg_val("payload_size_bytes"),
        get_named_compile_time_arg_val("r2_buffer_offset"),
        get_named_compile_time_arg_val("ncrisc_buffer_offset")>;

    ReduceToAll::ReaderArgs rt_args{};
    if constexpr (CTArgs::is_fabric_core == 0) {
        rt_args = {
            get_common_arg_val<uint32_t>(0),  // recv_sem_round1
            get_common_arg_val<uint32_t>(1),  // recv_sem_round2
            get_common_arg_val<uint32_t>(2),  // recv_sem_round3
        };
    }

#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = ReduceToAll::WriterCTArgs<
        get_named_compile_time_arg_val("num_tiles"),
        get_named_compile_time_arg_val("payload_size_bytes"),
        get_named_compile_time_arg_val("local_cb"),
        get_named_compile_time_arg_val("scratch_cb"),
        get_named_compile_time_arg_val("packet_cb"),
        get_named_compile_time_arg_val("slot_size_bytes"),
        get_named_compile_time_arg_val("is_fabric_core"),
        get_named_compile_time_arg_val("fwd_dst_chip_id"),
        get_named_compile_time_arg_val("fwd_dst_mesh_id"),
        get_named_compile_time_arg_val("bwd_dst_chip_id"),
        get_named_compile_time_arg_val("bwd_dst_mesh_id"),
        get_named_compile_time_arg_val("r3_dst_chip_id"),
        get_named_compile_time_arg_val("r3_dst_mesh_id"),
        get_named_compile_time_arg_val("reload_cb"),
        get_named_compile_time_arg_val("compute_tile_size"),
        get_named_compile_time_arg_val("slots_per_direction"),
        get_named_compile_time_arg_val("r2_buffer_offset"),
        get_named_compile_time_arg_val("ncrisc_buffer_offset"),
        get_named_compile_time_arg_val("r3_buffer_offset")>;

    ReduceToAll::WorkerWriterArgs rt_args{};
    if constexpr (CTArgs::is_fabric_core == 0) {
        rt_args = {
            get_arg_val<uint32_t>(0),                  // fc_noc_x
            get_arg_val<uint32_t>(1),                  // fc_noc_y
            get_arg_val<uint32_t>(2),                  // is_type_a
            get_arg_val<uint32_t>(3),                  // r1_slot_offset
            get_arg_val<uint32_t>(4),                  // r1_slot_bit
            get_semaphore(get_arg_val<uint32_t>(5)),   // r1_sem_addr (from sem ID)
            get_arg_val<uint32_t>(6),                  // r2_slot_offset
            get_arg_val<uint32_t>(7),                  // r2_slot_bit
            get_semaphore(get_arg_val<uint32_t>(8)),   // r2_sem_addr (from sem ID)
            get_arg_val<uint32_t>(9),                  // r1_dst_l1_addr
            get_arg_val<uint32_t>(10),                 // r1_dst_sem_addr
            get_arg_val<uint32_t>(11),                 // r2_dst_l1_addr
            get_arg_val<uint32_t>(12),                 // r2_dst_sem_addr
            get_arg_val<uint32_t>(13),                 // r3_dst_l1_addr
            get_arg_val<uint32_t>(14),                 // r3_dst_sem_addr
            get_arg_val<uint32_t>(15),                 // output_base_addr
            get_arg_val<uint32_t>(16),                 // r3_slot_offset
            get_arg_val<uint32_t>(17),                 // r3_slot_bit
            get_semaphore(get_arg_val<uint32_t>(18)),  // r3_sem_addr (from sem ID)
        };
    }

#elif defined(COMPILE_FOR_TRISC)
    using CTArgs = ReduceToAll::ComputeCTArgs<
        get_named_compile_time_arg_val("num_tiles"),
        get_named_compile_time_arg_val("local_cb"),
        get_named_compile_time_arg_val("received_cb"),
        get_named_compile_time_arg_val("scratch_cb"),
        get_named_compile_time_arg_val("reload_cb"),
        get_named_compile_time_arg_val("is_fabric_core")>;

    ReduceToAll::ComputeArgs rt_args{};
    deepseek_compute_kernel_init();
#endif

    constexpr uint32_t num_loop_iters = get_named_compile_time_arg_val("num_loop_iters");
    ReduceToAll::Op<CTArgs, true> op;
    for (uint32_t iter = 0; iter < num_loop_iters; ++iter) {
        op(rt_args);
    }
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
