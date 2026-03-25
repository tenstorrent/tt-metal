// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_reduce_dual_core.hpp"

void kernel_main() {
    constexpr bool is_sender = get_named_compile_time_arg_val("is_allreduce_sender_core") == 1;
    constexpr bool is_receiver = get_named_compile_time_arg_val("is_allreduce_receiver_core") == 1;

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    if constexpr (is_sender) {
        using WriterCT = deepseek_b1_ops::AllReduceDualCore::WriterLinkCTArgs<
            get_named_compile_time_arg_val("allreduce_local_data_cb_id"),
            get_named_compile_time_arg_val("allreduce_input_num_tiles"),
            get_named_compile_time_arg_val("allreduce_page_size_bytes"),
            get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
            get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
            get_named_compile_time_arg_val("allreduce_num_chunks"),
            get_named_compile_time_arg_val("allreduce_num_links"),
            get_named_compile_time_arg_val("allreduce_writer_link_index"),
            get_named_compile_time_arg_val("allreduce_writer_signal_local_ready")>;

        deepseek_b1_ops::AllReduceDualCore::SenderFabricArgs args{};
        args.intermediate_buffer_address = get_common_arg_val<uint32_t>(0);
        args.dest_noc_x = get_common_arg_val<uint32_t>(1);
        args.dest_noc_y = get_common_arg_val<uint32_t>(2);
        args.per_core_rta_start_idx = 0;

        deepseek_b1_ops::AllReduceDualCore::WriterSingleLink<WriterCT> writer;
        writer(args);
    }
#endif

#if defined(COMPILE_FOR_BRISC)
    if constexpr (is_receiver) {
        using ReaderCT = deepseek_b1_ops::AllReduceDualCore::ReaderTwoCoreCTArgs<
            get_named_compile_time_arg_val("allreduce_local_data_cb_id"),
            get_named_compile_time_arg_val("allreduce_remote_data_cb_id"),
            get_named_compile_time_arg_val("allreduce_residual_cb_id"),
            get_named_compile_time_arg_val("allreduce_has_residual"),
            get_named_compile_time_arg_val("allreduce_skip_local_push"),
            get_named_compile_time_arg_val("allreduce_total_num_tiles"),
            get_named_compile_time_arg_val("allreduce_page_size_bytes"),
            get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
            get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
            get_named_compile_time_arg_val("allreduce_num_chunks"),
            get_named_compile_time_arg_val("allreduce_num_links")>;

        deepseek_b1_ops::AllReduceDualCore::ReceiverArgs args{};
        args.sem_bank_addr_0 = get_common_arg_val<uint32_t>(0);
        args.sem_bank_addr_1 = get_common_arg_val<uint32_t>(1);
        args.sender_noc_x = get_common_arg_val<uint32_t>(2);
        args.sender_noc_y = get_common_arg_val<uint32_t>(3);
        args.sender_local_data_l1_addr = get_common_arg_val<uint32_t>(4);
        args.local_ready_sem_bank_addr = get_common_arg_val<uint32_t>(5);

        deepseek_b1_ops::AllReduceDualCore::ReaderTwoCore<ReaderCT> reader;
        reader(args);
    }
#endif

#if defined(COMPILE_FOR_TRISC)
    if constexpr (is_receiver) {
        using ComputeCTArgs = deepseek_b1_ops::AllReduceDualCore::ComputeDualCoreCTArgs<
            get_named_compile_time_arg_val("allreduce_cb_remote"),
            get_named_compile_time_arg_val("allreduce_cb_local"),
            get_named_compile_time_arg_val("allreduce_cb_out"),
            get_named_compile_time_arg_val("allreduce_cb_residual"),
            get_named_compile_time_arg_val("allreduce_cb_temp"),
            get_named_compile_time_arg_val("allreduce_has_residual"),
            get_named_compile_time_arg_val("allreduce_num_tiles")>;

        deepseek_compute_kernel_init();

        deepseek_b1_ops::AllReduceDualCore::ComputeArgs args{};
        deepseek_b1_ops::AllReduceDualCore::ComputeDualCore<ComputeCTArgs> compute;
        compute(args);
    }
#endif
}
