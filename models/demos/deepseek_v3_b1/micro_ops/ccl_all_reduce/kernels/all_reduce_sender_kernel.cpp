// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified CCL All-Reduce Sender Kernel
// - NCRISC: Sender Reader (reads local data into CB)
// - BRISC: Sender Writer (sends data to remote device via fabric)
// - TRISC: No-op (compute runs on receiver core)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_reduce.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using AllReduce = deepseek_b1_ops::AllReduce;
    // Sender Reader CTArgs
    using SenderReaderCTArgs = AllReduce::SenderReaderCTArgs<
        get_named_compile_time_arg_val("cb0_id"),
        get_named_compile_time_arg_val("num_tiles"),
        get_named_compile_time_arg_val("tensor_page_size"),
        get_named_compile_time_arg_val("core_noc_x"),
        get_named_compile_time_arg_val("core_noc_y")>;

    // Runtime args
    AllReduce::SenderReaderArgs reader_args{
        get_arg_val<uint32_t>(0),  // tensor_address
    };

    // Execute sender reader
    AllReduce::SenderReaderOp<SenderReaderCTArgs> sender_reader;
    sender_reader(reader_args);

#elif defined(COMPILE_FOR_BRISC)
    using AllReduce = deepseek_b1_ops::AllReduce;
    // Sender Writer CTArgs
    using SenderWriterCTArgs = AllReduce::SenderWriterCTArgs<
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
        get_named_compile_time_arg_val("num_connections"),
        get_named_compile_time_arg_val("using_persistent_buffer")>;

    // Runtime args
    AllReduce::SenderWriterArgs writer_args{
        get_arg_val<uint32_t>(0),  // receiver_base_address
        get_arg_val<uint32_t>(1),  // receive_semaphore_addr
    };

    size_t arg_idx = 2;  // fabric args start at index 2

    // Execute sender writer
    AllReduce::SenderWriterOp<SenderWriterCTArgs> sender_writer;
    sender_writer(writer_args, arg_idx);

#elif defined(COMPILE_FOR_TRISC)
    // No-op for TRISC on sender core - compute runs on receiver core
#endif
}
