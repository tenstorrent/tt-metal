// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified CCL All-Reduce Kernel
//
// This kernel handles both sender and receiver cores via compile-time dispatch.
// The core role is determined by the `is_sender` compile-time arg:
// - is_sender=1: Sender core (reads local data, sends to neighbor)
// - is_sender=0: Receiver core (receives remote data, performs reduction)
//
// Sender Core:
//   - NCRISC: Reads local tensor data into CB
//   - BRISC: Sends data to remote device via fabric
//   - TRISC: No-op
//
// Receiver Core:
//   - NCRISC: Waits for remote data, pushes to compute
//   - BRISC: No-op
//   - TRISC: Reduction compute (adds local + remote data)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_reduce_sender.hpp"
#include "../../../unified_kernels/all_reduce_receiver.hpp"

void kernel_main() {
    constexpr bool is_sender = get_named_compile_time_arg_val("is_sender") == 1;

#if defined(COMPILE_FOR_NCRISC)
    // ========================================================================
    // NCRISC: Both sender and receiver have reader logic
    // ========================================================================
    if constexpr (is_sender) {
        using Sender = deepseek_b1_ops::AllReduceSender;

        using ReaderCTArgs = Sender::ReaderCTArgs<
            get_named_compile_time_arg_val("cb0_id"),
            get_named_compile_time_arg_val("num_tiles"),
            get_named_compile_time_arg_val("tensor_page_size"),
            get_named_compile_time_arg_val("core_noc_x"),
            get_named_compile_time_arg_val("core_noc_y")>;

        // Dummy WriterCTArgs - not used by NCRISC but needed for Op template
        using WriterCTArgs = Sender::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        Sender::RTArgs args{};
        args.tensor_address = get_common_arg_val<uint32_t>(0);
        size_t fabric_arg_idx = 0;

        Sender::Op<ReaderCTArgs, WriterCTArgs> op;
        op(args, fabric_arg_idx);
    } else {
        using Receiver = deepseek_b1_ops::AllReduceReceiver;

        using ReaderCTArgs = Receiver::ReaderCTArgs<
            get_named_compile_time_arg_val("packet_header_cb_id"),
            get_named_compile_time_arg_val("cb_in1"),
            get_named_compile_time_arg_val("l1_alignment"),
            get_named_compile_time_arg_val("cb_in2"),
            get_named_compile_time_arg_val("remote_sender_noc_x"),
            get_named_compile_time_arg_val("remote_sender_noc_y"),
            get_named_compile_time_arg_val("num_standard_tiles"),
            get_named_compile_time_arg_val("cb_residual"),
            get_named_compile_time_arg_val("has_residual")>;

        // Dummy ComputeCTArgs - not used by NCRISC but needed for Op template
        using ComputeCTArgs = Receiver::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0>;

        Receiver::RTArgs args{};
        args.sender_semaphore_addr = get_common_arg_val<uint32_t>(0);
        size_t fabric_arg_idx = 0;

        Receiver::Op<ReaderCTArgs, ComputeCTArgs> op;
        op(args, fabric_arg_idx);
    }

#elif defined(COMPILE_FOR_BRISC)
    // ========================================================================
    // BRISC: Only sender has writer logic; receiver is no-op
    // ========================================================================
    if constexpr (is_sender) {
        using Sender = deepseek_b1_ops::AllReduceSender;

        // Dummy ReaderCTArgs - not used by BRISC but needed for Op template
        using ReaderCTArgs = Sender::ReaderCTArgs<0, 0, 0, 0, 0>;

        using WriterCTArgs = Sender::WriterCTArgs<
            get_named_compile_time_arg_val("packet_header_cb_id"),
            get_named_compile_time_arg_val("packet_cb_id"),
            get_named_compile_time_arg_val("l1_alignment"),
            get_named_compile_time_arg_val("input_num_tiles"),
            get_named_compile_time_arg_val("page_size_bytes"),
            get_named_compile_time_arg_val("payload_size_bytes"),
            get_named_compile_time_arg_val("data_noc_x"),
            get_named_compile_time_arg_val("data_noc_y"),
            get_named_compile_time_arg_val("remote_receiver_noc_x"),
            get_named_compile_time_arg_val("remote_receiver_noc_y"),
            get_named_compile_time_arg_val("dst_num_hops"),
            get_named_compile_time_arg_val("num_connections")>;

        Sender::RTArgs args{};
        args.receiver_base_address = get_common_arg_val<uint32_t>(0);
        args.receive_semaphore_addr = get_common_arg_val<uint32_t>(1);
        size_t fabric_arg_idx = 0;

        Sender::Op<ReaderCTArgs, WriterCTArgs> op;
        op(args, fabric_arg_idx);
    }
    // else: receiver BRISC is no-op

#elif defined(COMPILE_FOR_TRISC)
    // ========================================================================
    // TRISC: Only receiver has compute logic; sender is no-op
    // ========================================================================
    if constexpr (!is_sender) {
        using Receiver = deepseek_b1_ops::AllReduceReceiver;

        // Dummy ReaderCTArgs - not used by TRISC but needed for Op template
        using ReaderCTArgs = Receiver::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;

        using ComputeCTArgs = Receiver::ComputeCTArgs<
            get_named_compile_time_arg_val("cb_in0"),
            get_named_compile_time_arg_val("cb_in1"),
            get_named_compile_time_arg_val("cb_out0"),
            get_named_compile_time_arg_val("cb_residual"),
            get_named_compile_time_arg_val("cb_temp"),
            get_named_compile_time_arg_val("has_residual"),
            get_named_compile_time_arg_val("num_tiles")>;

        deepseek_compute_kernel_init();

        Receiver::RTArgs args{};
        size_t fabric_arg_idx = 0;

        Receiver::Op<ReaderCTArgs, ComputeCTArgs> op;
        op(args, fabric_arg_idx);
    }
    // else: sender TRISC is no-op
#endif
}
