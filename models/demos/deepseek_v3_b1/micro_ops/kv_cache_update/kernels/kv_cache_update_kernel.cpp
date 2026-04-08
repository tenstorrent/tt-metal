// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// KV Cache Update kernel: uses KVCacheUpdate Op from kv_cache_update.hpp
//
// NOPE path:
//   nope_cache_tensor lives on a single "nope sender" core.
//   NCRISC on that core multicasts the data to a 2x8 "knope" grid (16 cores).
//   Each knope core then independently handles 1 DRAM tile (1x32).
//   BRISC starts DRAM reads in parallel with the mcast.
//
// ROPE path:
//   rope_cache_tensor is already split across 2 cores (1x32 each).
//   Each core independently handles its own tile.
//
// BRISC: DRAM read (knope/krope cores)
// NCRISC: Mcast sender/receiver (knope cores) + patch + DRAM write
// TRISC: Untilize + tilize

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/kv_cache_update.hpp"
struct Core {
    static constexpr bool is_nope_sender_core = get_named_compile_time_arg_val("is_nope_sender_core") == 1;
    static constexpr bool is_nope_core = get_named_compile_time_arg_val("is_nope_core") == 1;
    static constexpr bool is_rope_core = get_named_compile_time_arg_val("is_rope_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_nope_sender_core) {
        uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
        unified_kernels::setup_sharded_buffer(kv_rmsnorm_output_cb, 1);
    }
    if constexpr (Core::is_rope_core) {
        uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");
        unified_kernels::setup_sharded_buffer(krope_output_cb, 1);
    }
    deepseek_b1_ops::KVCacheUpdate::WriterArgs kv_cache_args{
        .kv_cache_buffer_base_addr = get_common_arg_val<uint32_t>(0),
        .local_cur_pos = 0,
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
        .kv_cache_intermed_sync_cb = get_named_compile_time_arg_val("kv_cache_intermed_sync_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb"),
        .krope_output_cb = get_named_compile_time_arg_val("krope_output_cb"),
        .grid_start_y = get_named_compile_time_arg_val("kv_cache_grid_start_y"),
        .kv_cache_cur_pos_ready_semaphore_addr =
            get_semaphore(get_named_compile_time_arg_val("kv_cache_cur_pos_ready_semaphore_id")),
        .k_chunk_size = 0,
        .num_cores_per_head = 0,
        .mla_sender_noc_x = {},
        .mla_sender_noc_y = {},
        .knope_core_index = get_named_compile_time_arg_val("knope_core_index"),
        .nope_mcast_dest_noc_start_x = get_named_compile_time_arg_val("nope_mcast_dest_noc_start_x"),
        .nope_mcast_dest_noc_start_y = get_named_compile_time_arg_val("nope_mcast_dest_noc_start_y"),
        .nope_mcast_dest_noc_end_x = get_named_compile_time_arg_val("nope_mcast_dest_noc_end_x"),
        .nope_mcast_dest_noc_end_y = get_named_compile_time_arg_val("nope_mcast_dest_noc_end_y"),
        .nope_mcast_sender_semaphore_addr =
            get_semaphore(get_named_compile_time_arg_val("nope_mcast_sender_semaphore_id")),
        .nope_mcast_receiver_semaphore_addr =
            get_semaphore(get_named_compile_time_arg_val("nope_mcast_receiver_semaphore_id")),
        .nope_mcast_data_size_bytes = get_named_compile_time_arg_val("nope_mcast_data_size_bytes"),
        .nope_mcast_num_dests = get_named_compile_time_arg_val("nope_mcast_num_dests"),
        .kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles"),
    };

#elif defined(COMPILE_FOR_BRISC)
    deepseek_b1_ops::KVCacheUpdate::ReaderArgs kv_cache_args{
        .kv_cache_buffer_base_addr = get_common_arg_val<uint32_t>(0),
        .local_cur_pos = 0,
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .grid_start_y = get_named_compile_time_arg_val("kv_cache_grid_start_y"),
        .knope_core_index = get_named_compile_time_arg_val("knope_core_index"),
    };

#elif defined(COMPILE_FOR_TRISC)
    deepseek_b1_ops::KVCacheUpdate::ComputeArgs kv_cache_args{
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
        .kv_cache_intermed_sync_cb = get_named_compile_time_arg_val("kv_cache_intermed_sync_cb"),
    };

    deepseek_compute_kernel_init();
#endif

    deepseek_b1_ops::KVCacheUpdate::Op<Core::is_nope_sender_core, Core::is_nope_core, Core::is_rope_core>
        kv_cache_update;
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    {
        uint32_t pos_addr = get_common_arg_val<uint32_t>(1);
        volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr);
        kv_cache_update.set_local_cur_pos(kv_cache_args, pos_ptr[0]);
    }
#endif
    kv_cache_update(kv_cache_args);
}
