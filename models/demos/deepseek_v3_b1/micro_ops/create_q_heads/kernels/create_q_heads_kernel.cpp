// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Create Q Heads unified kernel
// Single kernel file, compiles correctly for BRISC, NCRISC, and TRISC
//
// Gathers data from 12x8 sender cores to 4x2 receiver cores, then tilizes.
// Each sender row maps to a different receiver core.
//
// Memory layout for tilization (max tilize dim = 256):
//   Phase 1: First 8 halves of QNOPE - shape [8, 256] → 8 tiles
//   Phase 2: Second 8 halves of QNOPE - shape [8, 256] → 8 tiles
//   Phase 3: QROPE - shape [8, 64] → 2 tiles
//
// Standalone op pattern:
//   - NCRISC: All data movement (sender on qnope/qrope cores, receiver on sdpa input cores)
//   - BRISC: Idle
//   - TRISC: Tilization on receiver cores

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/create_q_heads.hpp"

struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_receiver_core = get_named_compile_time_arg_val("is_receiver_core") == 1;
};

// Standalone op configuration:
// - pop_src: true (pop source CB after sending)

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // NCRISC: All data movement — sender on qnope/qrope cores, receiver on sdpa input cores
    uint32_t receiver_data_addr = get_common_arg_val<uint32_t>(0);
    using CreateQHeadsSenderCTArgs = deepseek_b1_ops::CreateQHeads::SenderCTArgs<
        get_named_compile_time_arg_val("qnope_data_size_bytes"),
        get_named_compile_time_arg_val("qrope_head_size_bytes")>;
    deepseek_b1_ops::CreateQHeads::SenderArgs sender_args{
        get_named_compile_time_arg_val("sender_grid_start_x"),
        get_named_compile_time_arg_val("sender_grid_start_y"),
        get_named_compile_time_arg_val("head_stride_bytes"),
        get_named_compile_time_arg_val("qnope_cols"),
        get_named_compile_time_arg_val("qnope_cb"),
        get_named_compile_time_arg_val("qrope_cb"),
        get_named_compile_time_arg_val("src_num_pages"),
        get_semaphore(get_named_compile_time_arg_val("nope_phase1_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("nope_phase2_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("rope_semaphore_id")),
        {
            get_named_compile_time_arg_val("target_noc_coords_row0"),
            get_named_compile_time_arg_val("target_noc_coords_row1"),
            get_named_compile_time_arg_val("target_noc_coords_row2"),
            get_named_compile_time_arg_val("target_noc_coords_row3"),
            get_named_compile_time_arg_val("target_noc_coords_row4"),
            get_named_compile_time_arg_val("target_noc_coords_row5"),
            get_named_compile_time_arg_val("target_noc_coords_row6"),
            get_named_compile_time_arg_val("target_noc_coords_row7"),
        },
        receiver_data_addr,
    };

    using CreateQHeadsReceiverCTArgs = deepseek_b1_ops::CreateQHeads::ReceiverCTArgs;
    deepseek_b1_ops::CreateQHeads::ReceiverArgs receiver_args{
        get_semaphore(get_named_compile_time_arg_val("nope_phase1_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("nope_phase2_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("rope_semaphore_id")),
        get_named_compile_time_arg_val("num_nope_senders"),
        get_named_compile_time_arg_val("num_rope_senders"),
        get_named_compile_time_arg_val("receiver_in_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("nope_tiles"),
        get_named_compile_time_arg_val("rope_tiles"),
    };

#elif defined(COMPILE_FOR_TRISC)
    // TRISC (Compute): Tilization args for receiver cores
    using CreateQHeadsReceiverCTArgs = deepseek_b1_ops::CreateQHeads::ComputeCTArgs;
    deepseek_b1_ops::CreateQHeads::ComputeArgs receiver_args{
        get_named_compile_time_arg_val("receiver_in_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("nope_tiles"),
        get_named_compile_time_arg_val("rope_tiles"),
    };
    deepseek_compute_kernel_init();
#endif
#if defined(COMPILE_FOR_NCRISC)
    using CreateQHeadsSenderOp = deepseek_b1_ops::CreateQHeads::
        Op<CreateQHeadsSenderCTArgs, Core::is_sender_core, Core::is_receiver_core, true, true, true>;
    CreateQHeadsSenderOp create_q_heads_sender;
    create_q_heads_sender(sender_args);
#endif
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_TRISC)
    using CreateQHeadsReceiverOp = deepseek_b1_ops::CreateQHeads::
        Op<CreateQHeadsReceiverCTArgs, Core::is_sender_core, Core::is_receiver_core, true, true, true>;
    CreateQHeadsReceiverOp create_q_heads_receiver;
    create_q_heads_receiver(receiver_args);
#endif
}
