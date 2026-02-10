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
//   - NCRISC: NOC0 sender OR receiver (if NOC1 sender)
//   - BRISC: NOC1 sender OR receiver (if NOC0 sender)
//   - TRISC: Tilization on receiver cores

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/create_q_heads.hpp"

struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_receiver_core = get_named_compile_time_arg_val("is_receiver_core") == 1;
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    static constexpr bool is_noc0_sender = get_named_compile_time_arg_val("is_noc0_sender") == 1;
    static constexpr bool is_noc1_sender = get_named_compile_time_arg_val("is_noc1_sender") == 1;
#else
    // TRISC doesn't need these
    static constexpr bool is_noc0_sender = false;
    static constexpr bool is_noc1_sender = false;
#endif
};

// Standalone op configuration:
// - pop_src: true (pop source CB after sending)
//
// For standalone op, senders can be on NOC0 (NCRISC) or NOC1 (BRISC).
// Receivers run on the opposite RISC from their sender role.

void kernel_main() {
// ============================================================================
// NCRISC - NOC0 sender OR receiver (if NOC1 sender)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Determine role on this RISC
    constexpr bool is_ncrisc_sender = Core::is_sender_core && Core::is_noc0_sender;
    constexpr bool is_ncrisc_receiver = Core::is_receiver_core && !Core::is_noc0_sender;

    // setup_sharded_input=true: explicitly mark sharded input CB pages as available
    //   (cb_descriptor_from_sharded_tensor binds buffer but does NOT pre-populate)
    using CreateQHeadsOp = deepseek_b1_ops::CreateQHeads::Op<is_ncrisc_sender, is_ncrisc_receiver, true, true>;
    CreateQHeadsOp create_q_heads;

    if constexpr (is_ncrisc_sender) {
        // NOC0 sender on NCRISC
        uint32_t receiver_data_addr = get_common_arg_val<uint32_t>(0);
        deepseek_b1_ops::CreateQHeads::SenderArgs sender_args{
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
        create_q_heads(sender_args);
    } else if constexpr (is_ncrisc_receiver) {
        // Receiver on NCRISC (when sender is NOC1/BRISC)
        deepseek_b1_ops::CreateQHeads::ReceiverArgs receiver_args{
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
        create_q_heads(receiver_args);
    }

// ============================================================================
// BRISC - NOC1 sender OR receiver (if NOC0 sender)
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Determine role on this RISC
    constexpr bool is_brisc_sender = Core::is_sender_core && Core::is_noc1_sender;
    constexpr bool is_brisc_receiver = Core::is_receiver_core && !Core::is_noc1_sender;

    // setup_sharded_input=true: explicitly mark sharded input CB pages as available
    using CreateQHeadsOp = deepseek_b1_ops::CreateQHeads::Op<is_brisc_sender, is_brisc_receiver, true, true>;
    CreateQHeadsOp create_q_heads;

    if constexpr (is_brisc_sender) {
        // NOC1 sender on BRISC
        uint32_t receiver_data_addr = get_common_arg_val<uint32_t>(0);
        deepseek_b1_ops::CreateQHeads::SenderArgs sender_args{
            get_named_compile_time_arg_val("sender_grid_start_x"),
            get_named_compile_time_arg_val("sender_grid_start_y"),
            get_named_compile_time_arg_val("qnope_data_size_bytes"),
            get_named_compile_time_arg_val("qrope_head_size_bytes"),
            get_named_compile_time_arg_val("head_stride_bytes"),
            get_named_compile_time_arg_val("qnope_cols"),
            get_named_compile_time_arg_val("qnope_cb"),
            get_named_compile_time_arg_val("qrope_cb"),
            get_named_compile_time_arg_val("src_num_pages"),
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
        create_q_heads(sender_args);
    } else if constexpr (is_brisc_receiver) {
        // Receiver on BRISC (when sender is NOC0/NCRISC)
        deepseek_b1_ops::CreateQHeads::ReceiverArgs receiver_args{
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
        create_q_heads(receiver_args);
    }

// ============================================================================
// TRISC (Compute) - Tilization on receiver cores
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    if constexpr (Core::is_receiver_core) {
        using CreateQHeadsOp = deepseek_b1_ops::CreateQHeads::Op<false, true, false, true>;
        CreateQHeadsOp create_q_heads;

        deepseek_b1_ops::CreateQHeads::ComputeArgs compute_args{
            get_named_compile_time_arg_val("receiver_in_cb"),
            get_named_compile_time_arg_val("out_cb"),
            get_named_compile_time_arg_val("nope_tiles"),
            get_named_compile_time_arg_val("rope_tiles"),
        };
        create_q_heads(compute_args);
    }

#endif
}
