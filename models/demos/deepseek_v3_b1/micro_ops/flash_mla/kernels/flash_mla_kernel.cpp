// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Flash MLA Decode kernel: uses FlashMLADecode Op from flash_mla_kernel.hpp
//
// NCRISC (Reader): Read Q from sharded memory, pipelined DRAM reads of K chunks
// BRISC (Writer):  Multicast K to S block receivers, tree reduction send/receive
// TRISC (Compute): SDPA flash attention chunking, tree reduction tail

#include "../../../unified_kernels/flash_mla.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    uint32_t arg_idx = 0;
    deepseek_b1_ops::FlashMLADecode::ReaderArgs args{
        .k_addr = get_common_arg_val<uint32_t>(0),
        .pos_addr = get_common_arg_val<uint32_t>(1),
        .cur_batch = get_arg_val<uint32_t>(arg_idx++),
        .core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++),
        .is_mcast_sender = get_arg_val<uint32_t>(arg_idx++),
        .is_output_core = get_arg_val<uint32_t>(arg_idx++),
        .output_core_noc_x = get_arg_val<uint32_t>(arg_idx++),
        .output_core_noc_y = get_arg_val<uint32_t>(arg_idx++),
        .mcast_start_x = get_arg_val<uint32_t>(arg_idx++),
        .mcast_start_y = get_arg_val<uint32_t>(arg_idx++),
        .mcast_end_x = get_arg_val<uint32_t>(arg_idx++),
        .mcast_end_y = get_arg_val<uint32_t>(arg_idx++),
        .vc = get_arg_val<uint32_t>(arg_idx++),
        .St = get_named_compile_time_arg_val("St"),
        .DHt = get_named_compile_time_arg_val("DHt"),
        .Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t"),
        .num_cores_per_head = get_named_compile_time_arg_val("num_cores_per_head"),
        .k_chunk_size = get_named_compile_time_arg_val("k_chunk_size"),
        .num_mcast_dests = get_named_compile_time_arg_val("num_mcast_dests"),
        .mcast_semaphore_id = get_named_compile_time_arg_val("mcast_semaphore_id"),
        .k_page_size = get_named_compile_time_arg_val("k_page_size"),
        .k_num_pages = get_named_compile_time_arg_val("k_num_pages"),
        .q_chunk_size_bytes = get_named_compile_time_arg_val("q_chunk_size_bytes"),
        .full_grid_mcast_start_x = get_named_compile_time_arg_val("full_grid_mcast_start_x"),
        .full_grid_mcast_start_y = get_named_compile_time_arg_val("full_grid_mcast_start_y"),
        .full_grid_mcast_end_x = get_named_compile_time_arg_val("full_grid_mcast_end_x"),
        .full_grid_mcast_end_y = get_named_compile_time_arg_val("full_grid_mcast_end_y"),
        .full_grid_mcast_num_dests = get_named_compile_time_arg_val("full_grid_mcast_num_dests"),
        .q_input_mcast_semaphore_id = get_named_compile_time_arg_val("q_input_mcast_semaphore_id"),
        .ncrisc_brisc_sync_semaphore_id = get_named_compile_time_arg_val("ncrisc_brisc_sync_semaphore_id"),
        .receiver_ready_semaphore_id = get_named_compile_time_arg_val("receiver_ready_semaphore_id"),
        .kv_cache_cur_pos_ready_semaphore_id = get_named_compile_time_arg_val("kv_cache_cur_pos_ready_semaphore_id"),
        .kv_cache_cur_pos_ready_value = get_named_compile_time_arg_val("kv_cache_cur_pos_ready_value"),
        .cb_k_in = get_named_compile_time_arg_val("cb_k_in"),
        .cb_q_in = get_named_compile_time_arg_val("cb_q_in"),
        .cb_compute_in = get_named_compile_time_arg_val("cb_compute_in"),
    };

    using FlashMLACTArgs = deepseek_b1_ops::FlashMLADecode::ReaderCTArgs;

#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t num_tree_reduction_steps = get_named_compile_time_arg_val("num_tree_reduction_steps");
    uint32_t arg_idx = 0;
    uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    uint32_t is_mcast_sender = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mcast_start_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mcast_start_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mcast_end_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mcast_end_y = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* tree_reduction_info = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_tree_reduction_steps * 4;

    deepseek_b1_ops::FlashMLADecode::WriterArgs args{
        .pos_addr = get_common_arg_val<uint32_t>(0),
        .cur_batch = cur_batch,
        .core_num_in_reduce = core_num_in_reduce,
        .is_mcast_sender = is_mcast_sender,
        .mcast_start_x = mcast_start_x,
        .mcast_start_y = mcast_start_y,
        .mcast_end_x = mcast_end_x,
        .mcast_end_y = mcast_end_y,
        .tree_reduction_info = tree_reduction_info,
        .Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t"),
        .num_cores_per_head = get_named_compile_time_arg_val("num_cores_per_head"),
        .reducer_semaphore_id = get_named_compile_time_arg_val("reducer_semaphore_id"),
        .k_chunk_size = get_named_compile_time_arg_val("k_chunk_size"),
        .q_tile_height = get_named_compile_time_arg_val("q_tile_height"),
        .DHt = get_named_compile_time_arg_val("DHt"),
        .num_mcast_dests = get_named_compile_time_arg_val("num_mcast_dests"),
        .mcast_semaphore_id = get_named_compile_time_arg_val("mcast_semaphore_id"),
        .ncrisc_brisc_sync_semaphore_id = get_named_compile_time_arg_val("ncrisc_brisc_sync_semaphore_id"),
        .k_num_pages = get_named_compile_time_arg_val("k_num_pages"),
        .num_tree_reduction_steps = num_tree_reduction_steps,
        .receiver_ready_semaphore_id = get_named_compile_time_arg_val("receiver_ready_semaphore_id"),
        .cb_k_in = get_named_compile_time_arg_val("cb_k_in"),
        .cb_out_in = get_named_compile_time_arg_val("cb_out_in"),
        .cb_ms_in = get_named_compile_time_arg_val("cb_ms_in"),
        .cb_out_ms = get_named_compile_time_arg_val("cb_out_ms"),
    };

    using FlashMLACTArgs = deepseek_b1_ops::FlashMLADecode::WriterCTArgs<
        get_named_compile_time_arg_val("k_page_size"),
        get_named_compile_time_arg_val("vDHt"),
        get_named_compile_time_arg_val("cb_out_o")>;

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t num_tree_reduction_steps = get_named_compile_time_arg_val("num_tree_reduction_steps");
    uint32_t arg_idx = 0;
    uint32_t do_reduce = get_arg_val<uint32_t>(arg_idx++);
    uint32_t do_output = get_arg_val<uint32_t>(arg_idx++);
    uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    uint32_t is_sender_after_reduce = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* tree_reduction_info = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_tree_reduction_steps * 2;

    deepseek_b1_ops::FlashMLADecode::ComputeArgs args{
        .pos_addr = get_common_arg_val<uint32_t>(0),
        .do_reduce = do_reduce,
        .do_output = do_output,
        .cur_head = cur_head,
        .cur_batch = cur_batch,
        .core_num_in_reduce = core_num_in_reduce,
        .core_num_in_output = core_num_in_output,
        .is_sender_after_reduce = is_sender_after_reduce,
        .tree_reduction_info = tree_reduction_info,
        .k_chunk_size = get_named_compile_time_arg_val("k_chunk_size"),
        .num_cores_per_head = get_named_compile_time_arg_val("num_cores_per_head"),
        .num_tree_reduction_steps = num_tree_reduction_steps,
    };

    using FlashMLACTArgs = deepseek_b1_ops::FlashMLADecode::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_q_in"),
        get_named_compile_time_arg_val("cb_compute_in"),
        get_named_compile_time_arg_val("cb_k_in"),
        get_named_compile_time_arg_val("cb_interm_out"),
        get_named_compile_time_arg_val("cb_interm_ms"),
        get_named_compile_time_arg_val("cb_out_in"),
        get_named_compile_time_arg_val("cb_ms_in"),
        get_named_compile_time_arg_val("cb_out_o"),
        get_named_compile_time_arg_val("cb_out_ms"),
        get_named_compile_time_arg_val("cb_out_final")>;

    deepseek_compute_kernel_init();
#endif

#if defined(COMPILE_FOR_NCRISC)
    if (args.is_output_core == 1) {
        unified_kernels::setup_sharded_buffer(args.cb_q_in, args.DHt);
    }
#endif

    deepseek_b1_ops::FlashMLADecode::Op<FlashMLACTArgs, true, false> op;
    op(args);
}
