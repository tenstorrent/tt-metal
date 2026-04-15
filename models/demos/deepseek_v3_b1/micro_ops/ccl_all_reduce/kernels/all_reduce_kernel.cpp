// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_reduce.hpp"

void kernel_main() {
    constexpr bool is_sender = get_named_compile_time_arg_val("is_allreduce_sender_core") == 1;
    constexpr bool is_receiver = get_named_compile_time_arg_val("is_allreduce_receiver_core") == 1;

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    using WriterCT = deepseek_b1_ops::AllReduce::WriterCTArgs<
        get_named_compile_time_arg_val("allreduce_local_data_cb_id"),
        get_named_compile_time_arg_val("allreduce_input_num_tiles"),
        get_named_compile_time_arg_val("allreduce_page_size_bytes"),
        get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
        get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
        get_named_compile_time_arg_val("allreduce_num_chunks"),
        get_named_compile_time_arg_val("allreduce_num_links"),
        get_named_compile_time_arg_val("allreduce_writer_link_index"),
        get_named_compile_time_arg_val("allreduce_writer_signal_local_ready"),
        get_named_compile_time_arg_val("allreduce_skip_local_push")>;

    deepseek_b1_ops::AllReduce::SenderArgs sender_args{};
    if constexpr (is_sender) {
        sender_args.intermediate_buffer_address = get_common_arg_val<uint32_t>(0);
        sender_args.dest_noc_x = get_common_arg_val<uint32_t>(1);
        sender_args.dest_noc_y = get_common_arg_val<uint32_t>(2);
        sender_args.per_core_rta_start_idx = 0;
    }
#endif

#if defined(COMPILE_FOR_NCRISC)
    using ReaderCT = deepseek_b1_ops::AllReduce::ReaderCTArgs<
        get_named_compile_time_arg_val("allreduce_recv_local_data_cb_id"),
        get_named_compile_time_arg_val("allreduce_remote_data_cb_id"),
        get_named_compile_time_arg_val("allreduce_residual_cb_id"),
        get_named_compile_time_arg_val("allreduce_has_residual"),
        get_named_compile_time_arg_val("allreduce_total_num_tiles"),
        get_named_compile_time_arg_val("allreduce_page_size_bytes"),
        get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
        get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
        get_named_compile_time_arg_val("allreduce_num_chunks"),
        get_named_compile_time_arg_val("allreduce_num_links")>;

    deepseek_b1_ops::AllReduce::ReceiverArgs receiver_args{};
    if constexpr (is_receiver) {
        receiver_args.sem_bank_addr_0 = get_common_arg_val<uint32_t>(0);
        receiver_args.sem_bank_addr_1 = get_common_arg_val<uint32_t>(1);
        receiver_args.sender_noc_x = get_common_arg_val<uint32_t>(2);
        receiver_args.sender_noc_y = get_common_arg_val<uint32_t>(3);
        receiver_args.sender_local_data_l1_addr = get_common_arg_val<uint32_t>(4);
        receiver_args.local_ready_sem_bank_addr = get_common_arg_val<uint32_t>(5);
    }
#endif

#if defined(COMPILE_FOR_TRISC)
    using ComputeCT = deepseek_b1_ops::AllReduce::ComputeCTArgs<
        get_named_compile_time_arg_val("allreduce_cb_remote"),
        get_named_compile_time_arg_val("allreduce_cb_local"),
        get_named_compile_time_arg_val("allreduce_cb_out"),
        get_named_compile_time_arg_val("allreduce_cb_residual"),
        get_named_compile_time_arg_val("allreduce_has_residual"),
        get_named_compile_time_arg_val("allreduce_num_tiles")>;

    deepseek_b1_ops::AllReduce::ComputeArgs compute_args{};
    if constexpr (is_receiver) {
        deepseek_compute_kernel_init();
    }
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    if constexpr (is_sender) {
        DeviceZoneScopedN("CCL_SENDER_WRITER");
        deepseek_b1_ops::AllReduce::WriterSingleLink<WriterCT> writer;
        writer(sender_args);
    }
#endif

    if constexpr (is_receiver) {
        DeviceZoneScopedN("CCL_RECEIVER");
#if defined(COMPILE_FOR_NCRISC)
        deepseek_b1_ops::AllReduce::Reader<ReaderCT> reader;
        reader(receiver_args);
#elif defined(COMPILE_FOR_TRISC)
        deepseek_b1_ops::AllReduce::Compute<ComputeCT> compute;
        compute(compute_args);
#endif
    }
}
