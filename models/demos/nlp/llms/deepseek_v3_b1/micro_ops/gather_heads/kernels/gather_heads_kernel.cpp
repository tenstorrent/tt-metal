// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Gather Heads unified kernel
// Single kernel file, compiles correctly for BRISC, NCRISC, and TRISC
// Note: This is a dataflow-only op - no compute kernel
//
// Gathers data from 12x8 sender cores to 4x2 receiver cores.
// Each sender row maps to a different receiver core.
//
// Receiver layout (8 heads per core, each head = 576 elements):
//   Head N: qnope (512 elements) + qrope (64 elements)
//
// Standalone op pattern:
//   - NCRISC: NOC0 sender OR receiver (if NOC1 sender)
//   - BRISC: NOC1 sender OR receiver (if NOC0 sender)
//   - Uses RTArgs pattern like Gather op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/gather_heads.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_receiver_core = get_named_compile_time_arg_val("is_receiver_core") == 1;
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    // NOC sender flags only available for dataflow kernels
    static constexpr bool is_noc0_sender = get_named_compile_time_arg_val("is_noc0_sender") == 1;
    static constexpr bool is_noc1_sender = get_named_compile_time_arg_val("is_noc1_sender") == 1;
#else
    // TRISC doesn't need these - it's a no-op
    static constexpr bool is_noc0_sender = false;
    static constexpr bool is_noc1_sender = false;
#endif
};

// Standalone op configuration:
// - setup_sharded_input: true (input tensors are sharded)
// - pop_src: true (pop source CB after sending)
// - use_cb_output: false (output is sharded tensor, not CB)
// - count_heads: false (count cores, not heads)
//
// For standalone op, senders can be on NOC0 (NCRISC) or NOC1 (BRISC).
// Receivers run on the opposite RISC from their sender role.
// This is different from the mega kernel where all senders are on BRISC.

void kernel_main() {
// ============================================================================
// NCRISC - NOC0 sender OR receiver (if NOC1 sender)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Determine role on this RISC
    constexpr bool is_ncrisc_sender = Core::is_sender_core && Core::is_noc0_sender;
    constexpr bool is_ncrisc_receiver = Core::is_receiver_core && !Core::is_noc0_sender;

    using GatherHeadsOp = deepseek_b1_ops::GatherHeads::Op<is_ncrisc_sender, is_ncrisc_receiver, true, true, false>;

    if constexpr (is_ncrisc_sender) {
        // NOC0 sender on NCRISC
        uint32_t receiver_data_addr = get_arg_val<uint32_t>(0);
        deepseek_b1_ops::GatherHeads::SenderArgs gather_heads_args{
            get_named_compile_time_arg_val("sender_grid_start_x"),
            get_named_compile_time_arg_val("sender_grid_start_y"),
            get_named_compile_time_arg_val("qnope_data_size_bytes"),
            get_named_compile_time_arg_val("qrope_head_size_bytes"),
            get_named_compile_time_arg_val("head_stride_bytes"),
            get_named_compile_time_arg_val("qnope_cols"),
            get_named_compile_time_arg_val("qnope_cb"),
            get_named_compile_time_arg_val("qrope_cb"),
            get_named_compile_time_arg_val("src_num_pages"),
            get_named_compile_time_arg_val("receiver_semaphore_id"),
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
        GatherHeadsOp gather_heads;
        gather_heads(gather_heads_args);
    } else if constexpr (is_ncrisc_receiver) {
        // Receiver on NCRISC (when sender is NOC1/BRISC)
        constexpr uint32_t noc0_senders[] = {
            get_named_compile_time_arg_val("noc0_senders_row0"),
            get_named_compile_time_arg_val("noc0_senders_row1"),
            get_named_compile_time_arg_val("noc0_senders_row2"),
            get_named_compile_time_arg_val("noc0_senders_row3"),
            get_named_compile_time_arg_val("noc0_senders_row4"),
            get_named_compile_time_arg_val("noc0_senders_row5"),
            get_named_compile_time_arg_val("noc0_senders_row6"),
            get_named_compile_time_arg_val("noc0_senders_row7"),
        };
        constexpr uint32_t noc1_senders[] = {
            get_named_compile_time_arg_val("noc1_senders_row0"),
            get_named_compile_time_arg_val("noc1_senders_row1"),
            get_named_compile_time_arg_val("noc1_senders_row2"),
            get_named_compile_time_arg_val("noc1_senders_row3"),
            get_named_compile_time_arg_val("noc1_senders_row4"),
            get_named_compile_time_arg_val("noc1_senders_row5"),
            get_named_compile_time_arg_val("noc1_senders_row6"),
            get_named_compile_time_arg_val("noc1_senders_row7"),
        };
        constexpr uint32_t receiver_grid_start_x = get_named_compile_time_arg_val("receiver_grid_start_x");
        constexpr uint32_t receiver_grid_start_y = get_named_compile_time_arg_val("receiver_grid_start_y");
        constexpr uint32_t receiver_cols = get_named_compile_time_arg_val("receiver_cols");
        uint32_t rx = my_logical_x_ - receiver_grid_start_x;
        uint32_t ry = my_logical_y_ - receiver_grid_start_y;
        uint32_t sender_row = (ry == 0) ? rx : (rx + receiver_cols);

        deepseek_b1_ops::GatherHeads::ReceiverArgs gather_heads_args{
            get_named_compile_time_arg_val("noc0_receiver_semaphore_id"),
            noc0_senders[sender_row] + noc1_senders[sender_row],
            get_named_compile_time_arg_val("out_cb"),
            get_named_compile_time_arg_val("dst_num_pages"),
        };
        GatherHeadsOp gather_heads;
        gather_heads(gather_heads_args);
    }

// ============================================================================
// BRISC - NOC1 sender OR receiver (if NOC0 sender)
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Determine role on this RISC
    constexpr bool is_brisc_sender = Core::is_sender_core && Core::is_noc1_sender;
    constexpr bool is_brisc_receiver = Core::is_receiver_core && Core::is_noc0_sender;

    using GatherHeadsOp = deepseek_b1_ops::GatherHeads::Op<is_brisc_sender, is_brisc_receiver, true, true, false>;

    if constexpr (is_brisc_sender) {
        // NOC1 sender on BRISC
        uint32_t receiver_data_addr = get_arg_val<uint32_t>(0);
        deepseek_b1_ops::GatherHeads::SenderArgs gather_heads_args{
            get_named_compile_time_arg_val("sender_grid_start_x"),
            get_named_compile_time_arg_val("sender_grid_start_y"),
            get_named_compile_time_arg_val("qnope_data_size_bytes"),
            get_named_compile_time_arg_val("qrope_head_size_bytes"),
            get_named_compile_time_arg_val("head_stride_bytes"),
            get_named_compile_time_arg_val("qnope_cols"),
            get_named_compile_time_arg_val("qnope_cb"),
            get_named_compile_time_arg_val("qrope_cb"),
            get_named_compile_time_arg_val("src_num_pages"),
            get_named_compile_time_arg_val("receiver_semaphore_id"),
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
        GatherHeadsOp gather_heads;
        gather_heads(gather_heads_args);
    } else if constexpr (is_brisc_receiver) {
        // Receiver on BRISC (when sender is NOC0/NCRISC)
        constexpr uint32_t noc0_senders[] = {
            get_named_compile_time_arg_val("noc0_senders_row0"),
            get_named_compile_time_arg_val("noc0_senders_row1"),
            get_named_compile_time_arg_val("noc0_senders_row2"),
            get_named_compile_time_arg_val("noc0_senders_row3"),
            get_named_compile_time_arg_val("noc0_senders_row4"),
            get_named_compile_time_arg_val("noc0_senders_row5"),
            get_named_compile_time_arg_val("noc0_senders_row6"),
            get_named_compile_time_arg_val("noc0_senders_row7"),
        };
        constexpr uint32_t noc1_senders[] = {
            get_named_compile_time_arg_val("noc1_senders_row0"),
            get_named_compile_time_arg_val("noc1_senders_row1"),
            get_named_compile_time_arg_val("noc1_senders_row2"),
            get_named_compile_time_arg_val("noc1_senders_row3"),
            get_named_compile_time_arg_val("noc1_senders_row4"),
            get_named_compile_time_arg_val("noc1_senders_row5"),
            get_named_compile_time_arg_val("noc1_senders_row6"),
            get_named_compile_time_arg_val("noc1_senders_row7"),
        };
        constexpr uint32_t receiver_grid_start_x = get_named_compile_time_arg_val("receiver_grid_start_x");
        constexpr uint32_t receiver_grid_start_y = get_named_compile_time_arg_val("receiver_grid_start_y");
        constexpr uint32_t receiver_cols = get_named_compile_time_arg_val("receiver_cols");
        uint32_t rx = my_logical_x_ - receiver_grid_start_x;
        uint32_t ry = my_logical_y_ - receiver_grid_start_y;
        uint32_t sender_row = (ry == 0) ? rx : (rx + receiver_cols);

        deepseek_b1_ops::GatherHeads::ReceiverArgs gather_heads_args{
            get_named_compile_time_arg_val("noc0_receiver_semaphore_id"),
            noc0_senders[sender_row] + noc1_senders[sender_row],
            get_named_compile_time_arg_val("out_cb"),
            get_named_compile_time_arg_val("dst_num_pages"),
        };
        GatherHeadsOp gather_heads;
        gather_heads(gather_heads_args);
    }

// ============================================================================
// TRISC (Compute) - No-op (gather is dataflow only)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Gather is a dataflow-only operation, no compute needed
#endif
}
