// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Distributed Create Q Heads unified kernel
// Single kernel file, compiles correctly for BRISC, NCRISC, and TRISC
//
// Splits tilize work across three core sets:
//   - Original SDPA input cores: NOPE phase 1 [8, 256] -> output tiles 0..7
//   - NOPE helper cores: NOPE phase 2 [8, 256] -> output tiles 8..15
//   - ROPE helper cores: ROPE [8, 64] -> output tiles 16..17
//
// Helper cores write their tilized output directly into the original cores'
// output CB backing storage. Original cores remain the only owners of the
// destination CB metadata and publish the full 18-tile Q shard.
//
// Standalone op pattern:
//   - NCRISC: All data movement (sender / original-receiver / helper-receiver)
//   - BRISC: Idle
//   - TRISC: Tilization on original/helper cores

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/distributed_create_q_heads.hpp"

struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_original_receiver_core = get_named_compile_time_arg_val("is_original_receiver_core") == 1;
    static constexpr bool is_nope_helper_core = get_named_compile_time_arg_val("is_nope_helper_core") == 1;
    static constexpr bool is_rope_helper_core = get_named_compile_time_arg_val("is_rope_helper_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    uint32_t receiver_data_addr = get_common_arg_val<uint32_t>(0);
    using DistributedCreateQHeadsCTArgs = deepseek_b1_ops::DistributedCreateQHeads::SenderCTArgs<
        get_named_compile_time_arg_val("qnope_data_size_bytes"),
        get_named_compile_time_arg_val("qrope_head_size_bytes")>;
    deepseek_b1_ops::DistributedCreateQHeads::SenderArgs sender_args{
        get_named_compile_time_arg_val("sender_grid_start_x"),
        get_named_compile_time_arg_val("sender_grid_start_y"),
        get_named_compile_time_arg_val("qnope_cols"),
        get_named_compile_time_arg_val("qnope_cb"),
        get_named_compile_time_arg_val("qrope_cb"),
        get_named_compile_time_arg_val("src_num_pages"),
        get_semaphore(get_named_compile_time_arg_val("nope_phase1_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("nope_phase2_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("rope_semaphore_id")),
        {
            get_named_compile_time_arg_val("original_noc_coords_row0"),
            get_named_compile_time_arg_val("original_noc_coords_row1"),
            get_named_compile_time_arg_val("original_noc_coords_row2"),
            get_named_compile_time_arg_val("original_noc_coords_row3"),
            get_named_compile_time_arg_val("original_noc_coords_row4"),
            get_named_compile_time_arg_val("original_noc_coords_row5"),
            get_named_compile_time_arg_val("original_noc_coords_row6"),
            get_named_compile_time_arg_val("original_noc_coords_row7"),
        },
        {
            get_named_compile_time_arg_val("nope_helper_noc_coords_row0"),
            get_named_compile_time_arg_val("nope_helper_noc_coords_row1"),
            get_named_compile_time_arg_val("nope_helper_noc_coords_row2"),
            get_named_compile_time_arg_val("nope_helper_noc_coords_row3"),
            get_named_compile_time_arg_val("nope_helper_noc_coords_row4"),
            get_named_compile_time_arg_val("nope_helper_noc_coords_row5"),
            get_named_compile_time_arg_val("nope_helper_noc_coords_row6"),
            get_named_compile_time_arg_val("nope_helper_noc_coords_row7"),
        },
        {
            get_named_compile_time_arg_val("rope_helper_noc_coords_row0"),
            get_named_compile_time_arg_val("rope_helper_noc_coords_row1"),
            get_named_compile_time_arg_val("rope_helper_noc_coords_row2"),
            get_named_compile_time_arg_val("rope_helper_noc_coords_row3"),
            get_named_compile_time_arg_val("rope_helper_noc_coords_row4"),
            get_named_compile_time_arg_val("rope_helper_noc_coords_row5"),
            get_named_compile_time_arg_val("rope_helper_noc_coords_row6"),
            get_named_compile_time_arg_val("rope_helper_noc_coords_row7"),
        },
        receiver_data_addr,
    };
    deepseek_b1_ops::DistributedCreateQHeads::ReceiverArgs receiver_args{
        get_semaphore(get_named_compile_time_arg_val("nope_phase1_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("nope_phase2_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("rope_semaphore_id")),
        {
            get_named_compile_time_arg_val("original_noc_coords_row0"),
            get_named_compile_time_arg_val("original_noc_coords_row1"),
            get_named_compile_time_arg_val("original_noc_coords_row2"),
            get_named_compile_time_arg_val("original_noc_coords_row3"),
            get_named_compile_time_arg_val("original_noc_coords_row4"),
            get_named_compile_time_arg_val("original_noc_coords_row5"),
            get_named_compile_time_arg_val("original_noc_coords_row6"),
            get_named_compile_time_arg_val("original_noc_coords_row7"),
        },
        get_named_compile_time_arg_val("receiver_in_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("nope_tiles"),
        get_named_compile_time_arg_val("rope_tiles"),
        get_named_compile_time_arg_val("num_nope_senders"),
        get_named_compile_time_arg_val("num_rope_senders"),
    };

#elif defined(COMPILE_FOR_TRISC)
    deepseek_b1_ops::DistributedCreateQHeads::ComputeArgs compute_args{
        get_named_compile_time_arg_val("receiver_in_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("nope_tiles"),
        get_named_compile_time_arg_val("rope_tiles"),
    };
    deepseek_compute_kernel_init();
#endif

#if defined(COMPILE_FOR_NCRISC)
    deepseek_b1_ops::DistributedCreateQHeads::Op<
        DistributedCreateQHeadsCTArgs,
        Core::is_sender_core,
        Core::is_original_receiver_core,
        Core::is_nope_helper_core,
        Core::is_rope_helper_core,
        true,
        true,
        true>
        distributed_create_q_heads_op;
    distributed_create_q_heads_op(sender_args);
    distributed_create_q_heads_op(receiver_args);
#endif
#if defined(COMPILE_FOR_TRISC)
    deepseek_b1_ops::DistributedCreateQHeads::Op<
        deepseek_b1_ops::DistributedCreateQHeads::ComputeCTArgs,
        false,
        Core::is_original_receiver_core,
        Core::is_nope_helper_core,
        Core::is_rope_helper_core,
        true,
        true,
        true>
        distributed_create_q_heads_compute_op;
    distributed_create_q_heads_compute_op(compute_args);
#endif
}
