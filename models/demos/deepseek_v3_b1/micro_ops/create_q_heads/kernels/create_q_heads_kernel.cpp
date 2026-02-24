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
// Standalone op pattern (matching pre_sdpa gather pattern):
//   - NCRISC: All senders (always sender, never receiver)
//   - BRISC: All receivers (always receiver, never sender)
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
    // NCRISC: Sender args for QNOPE/QROPE cores (matching pre_sdpa gather pattern: NCRISC sender, BRISC receiver)
    uint32_t receiver_data_addr = get_common_arg_val<uint32_t>(0);
    deepseek_b1_ops::CreateQHeads::SenderArgs create_q_heads_args{
        get_named_compile_time_arg_val("sender_grid_start_x"),
        get_named_compile_time_arg_val("sender_grid_start_y"),
        get_named_compile_time_arg_val("qnope_data_size_bytes"),
        get_named_compile_time_arg_val("qrope_head_size_bytes"),
        get_named_compile_time_arg_val("head_stride_bytes"),
        get_named_compile_time_arg_val("qnope_cols"),
        get_named_compile_time_arg_val("qnope_cb"),
        get_named_compile_time_arg_val("qrope_cb"),
        get_named_compile_time_arg_val("src_num_pages"),
        // 3 semaphores for race-free synchronization
        get_named_compile_time_arg_val("nope_phase1_semaphore_id"),
        get_named_compile_time_arg_val("nope_phase2_semaphore_id"),
        get_named_compile_time_arg_val("rope_semaphore_id"),
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

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: Receiver args for SDPA input cores (matching pre_sdpa gather pattern: NCRISC sender, BRISC receiver)
    deepseek_b1_ops::CreateQHeads::ReceiverArgs create_q_heads_args{
        get_named_compile_time_arg_val("nope_phase1_semaphore_id"),
        get_named_compile_time_arg_val("nope_phase2_semaphore_id"),
        get_named_compile_time_arg_val("rope_semaphore_id"),
        get_named_compile_time_arg_val("num_nope_senders"),
        get_named_compile_time_arg_val("num_rope_senders"),
        get_named_compile_time_arg_val("receiver_in_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("nope_tiles"),
        get_named_compile_time_arg_val("rope_tiles"),
    };

#elif defined(COMPILE_FOR_TRISC)
    // TRISC (Compute): Tilization args for receiver cores
    deepseek_b1_ops::CreateQHeads::ComputeArgs create_q_heads_args{
        get_named_compile_time_arg_val("receiver_in_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("nope_tiles"),
        get_named_compile_time_arg_val("rope_tiles"),
    };
    deepseek_compute_kernel_init();
#endif

    using CreateQHeadsOp = deepseek_b1_ops::CreateQHeads::Op<Core::is_sender_core, Core::is_receiver_core, true, true>;
    CreateQHeadsOp create_q_heads;
    create_q_heads(create_q_heads_args);
}
