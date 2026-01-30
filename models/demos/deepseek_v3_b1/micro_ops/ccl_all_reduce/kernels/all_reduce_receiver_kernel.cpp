// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified CCL All-Reduce Receiver Kernel
// - NCRISC: Receiver Reader (waits for remote data, pushes to compute)
// - BRISC: No-op (writer runs on sender core)
// - TRISC: Reduction Compute (adds local + remote data)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_reduce.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using AllReduce = deepseek_b1_ops::AllReduce;
    // Receiver Reader CTArgs
    using ReceiverReaderCTArgs = AllReduce::ReceiverReaderCTArgs<
        get_named_compile_time_arg_val("packet_header_cb_id"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("l1_alignment"),
        get_named_compile_time_arg_val("cb_in2"),
        get_named_compile_time_arg_val("remote_sender_noc_x"),
        get_named_compile_time_arg_val("remote_sender_noc_y"),
        get_named_compile_time_arg_val("num_standard_tiles"),
        get_named_compile_time_arg_val("cb_residual"),
        get_named_compile_time_arg_val("has_residual"),
        get_named_compile_time_arg_val("using_persistent_buffer")>;

    // Runtime args
    AllReduce::ReceiverReaderArgs reader_args{
        get_arg_val<uint32_t>(0),  // sender_semaphore_addr
    };

    size_t arg_idx = 1;  // fabric args start at index 1

    // Execute receiver reader
    AllReduce::ReceiverReaderOp<ReceiverReaderCTArgs> receiver_reader;
    receiver_reader(reader_args, arg_idx);

#elif defined(COMPILE_FOR_BRISC)
    // No-op for BRISC on receiver core - writer runs on sender core

#elif defined(COMPILE_FOR_TRISC)
    using AllReduce = deepseek_b1_ops::AllReduce;
    // Compute CTArgs
    using ComputeCTArgs = AllReduce::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out0"),
        get_named_compile_time_arg_val("cb_residual"),
        get_named_compile_time_arg_val("cb_temp"),
        get_named_compile_time_arg_val("has_residual"),
        get_named_compile_time_arg_val("num_tiles")>;

    AllReduce::ComputeArgs compute_args{};

    // Execute reduction compute
    AllReduce::ComputeOp<ComputeCTArgs> compute;
    compute(compute_args);
#endif
}
