// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/all_reduce.hpp"

void kernel_main() {
    constexpr bool is_allreduce_core = get_named_compile_time_arg_val("is_allreduce_core") == 1;

#if defined(COMPILE_FOR_NCRISC)
    if constexpr (is_allreduce_core) {
        using WriterCTArgs = deepseek_b1_ops::AllReduce::WriterCTArgs<
            get_named_compile_time_arg_val("allreduce_local_data_cb_id"),
            get_named_compile_time_arg_val("allreduce_sync_cb_id"),
            get_named_compile_time_arg_val("allreduce_input_num_tiles"),
            get_named_compile_time_arg_val("allreduce_page_size_bytes"),
            get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
            get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
            get_named_compile_time_arg_val("allreduce_num_chunks"),
            get_named_compile_time_arg_val("allreduce_num_links")>;

        deepseek_b1_ops::AllReduce::RTArgs args{};
        args.intermediate_buffer_address = get_common_arg_val<uint32_t>(0);
        args.my_noc_x = get_common_arg_val<uint32_t>(1);
        args.my_noc_y = get_common_arg_val<uint32_t>(2);
        args.sem_bank_addr_0 = get_common_arg_val<uint32_t>(3);
        args.sem_bank_addr_1 = get_common_arg_val<uint32_t>(4);

        deepseek_b1_ops::AllReduce::Writer<WriterCTArgs> writer;
        writer(args);
    }

#elif defined(COMPILE_FOR_BRISC)
    if constexpr (is_allreduce_core) {
        using ReaderCTArgs = deepseek_b1_ops::AllReduce::ReaderCTArgs<
            get_named_compile_time_arg_val("allreduce_local_data_cb_id"),
            get_named_compile_time_arg_val("allreduce_remote_data_cb_id"),
            get_named_compile_time_arg_val("allreduce_residual_cb_id"),
            get_named_compile_time_arg_val("allreduce_has_residual"),
            get_named_compile_time_arg_val("allreduce_skip_local_push"),
            get_named_compile_time_arg_val("allreduce_total_num_tiles"),
            get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
            get_named_compile_time_arg_val("allreduce_last_chunk_tiles"),
            get_named_compile_time_arg_val("allreduce_num_chunks"),
            get_named_compile_time_arg_val("allreduce_num_links")>;

        deepseek_b1_ops::AllReduce::RTArgs args{};
        args.sem_bank_addr_0 = get_common_arg_val<uint32_t>(0);
        args.sem_bank_addr_1 = get_common_arg_val<uint32_t>(1);

        deepseek_b1_ops::AllReduce::Reader<ReaderCTArgs> reader;
        reader(args);
    }

#elif defined(COMPILE_FOR_TRISC)
    if constexpr (is_allreduce_core) {
        using ComputeCTArgs = deepseek_b1_ops::AllReduce::ComputeCTArgs<
            get_named_compile_time_arg_val("allreduce_cb_remote"),
            get_named_compile_time_arg_val("allreduce_cb_local"),
            get_named_compile_time_arg_val("allreduce_cb_out"),
            get_named_compile_time_arg_val("allreduce_sync_cb_id"),
            get_named_compile_time_arg_val("allreduce_cb_residual"),
            get_named_compile_time_arg_val("allreduce_cb_temp"),
            get_named_compile_time_arg_val("allreduce_has_residual"),
            get_named_compile_time_arg_val("allreduce_num_tiles"),
            get_named_compile_time_arg_val("allreduce_num_chunks"),
            get_named_compile_time_arg_val("allreduce_tiles_per_chunk"),
            get_named_compile_time_arg_val("allreduce_last_chunk_tiles")>;

        deepseek_compute_kernel_init();

        deepseek_b1_ops::AllReduce::RTArgs args{};
        deepseek_b1_ops::AllReduce::Compute<ComputeCTArgs> compute;
        compute(args);
    }
#endif
}
