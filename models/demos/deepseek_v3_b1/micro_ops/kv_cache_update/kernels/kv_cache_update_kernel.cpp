// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// KV Cache Update kernel: uses KVCacheUpdate Op from kv_cache_update.hpp
//
// BRISC: Stream existing KV cache pages from DRAM into kv_cache_input_cb
// NCRISC: Wait on compute, patch new data, wait on compute, write back to DRAM
// TRISC: Untilize input_cb -> intermed_cb, tilize intermed_cb -> output_cb

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/kv_cache_update.hpp"
#include "../../../metadata/metadata.hpp"

struct Core {
    static constexpr bool is_nope_core = get_named_compile_time_arg_val("is_nope_core") == 1;
    static constexpr bool is_rope_core = get_named_compile_time_arg_val("is_rope_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    if (Core::is_nope_core) {
        uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
        unified_kernels::setup_sharded_buffer(kv_rmsnorm_output_cb, 1);
    }
    if (Core::is_rope_core) {
        uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");
        unified_kernels::setup_sharded_buffer(krope_output_cb, 1);
    }
    deepseek_b1_ops::KVCacheUpdate::WriterArgs args{
        .kv_cache_buffer_base_addr = get_common_arg_val<uint32_t>(0),
        .local_cur_pos = 0,
        .slot_id = 0,
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb"),
        .krope_output_cb = get_named_compile_time_arg_val("krope_output_cb"),
        .grid_start_y = get_named_compile_time_arg_val("kv_cache_grid_start_y"),
        .full_grid_mcast_start_x = get_named_compile_time_arg_val("full_grid_mcast_start_x"),
        .full_grid_mcast_start_y = get_named_compile_time_arg_val("full_grid_mcast_start_y"),
        .full_grid_mcast_end_x = get_named_compile_time_arg_val("full_grid_mcast_end_x"),
        .full_grid_mcast_end_y = get_named_compile_time_arg_val("full_grid_mcast_end_y"),
        .full_grid_mcast_num_dests = get_named_compile_time_arg_val("full_grid_mcast_num_dests"),
        .kv_cache_cur_pos_ready_semaphore_addr =
            get_semaphore(get_named_compile_time_arg_val("kv_cache_cur_pos_ready_semaphore_id")),
    };
#elif defined(COMPILE_FOR_BRISC)
    deepseek_b1_ops::KVCacheUpdate::ReaderArgs args{
        .kv_cache_buffer_base_addr = get_common_arg_val<uint32_t>(0),
        .local_cur_pos = 0,
        .slot_id = 0,
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .grid_start_y = get_named_compile_time_arg_val("kv_cache_grid_start_y"),
    };
#elif defined(COMPILE_FOR_TRISC)
    deepseek_b1_ops::KVCacheUpdate::ComputeArgs args{
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
    };

    deepseek_compute_kernel_init();
#endif

    deepseek_b1_ops::KVCacheUpdate::Op<Core::is_nope_core, Core::is_rope_core> op;
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    {
        uint32_t metadata_addr = get_common_arg_val<uint32_t>(1);
        volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata* metadata_ptr =
            reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(metadata_addr);
        op.set_pos_and_slot(args, metadata_ptr->position_id, metadata_ptr->slot_id);
    }
#endif
    op(args);
}
