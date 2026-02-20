// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// KV Cache Update kernel: uses KVCacheUpdate Op from kv_cache_update.hpp
//
// NCRISC (IsNopeCore): Read 16 BFP8 tiles from DRAM into kv_cache_input_cb
// NCRISC (IsRopeCore): Pop krope_output_cb
// BRISC (IsNopeCore): Wait on intermed/rmsnorm, push intermed, read output_cb
// TRISC (IsNopeCore): Untilize input_cb -> intermed_cb, tilize intermed_cb -> output_cb

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/kv_cache_update.hpp"

struct Core {
    static constexpr bool is_nope_core = get_named_compile_time_arg_val("is_nope_core") == 1;
    static constexpr bool is_rope_core = get_named_compile_time_arg_val("is_rope_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    deepseek_b1_ops::KVCacheUpdate::ReaderArgs args{};
    if (Core::is_nope_core) {
        uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
        unified_kernels::setup_sharded_buffer(kv_rmsnorm_output_cb, 1);
    }
    if (Core::is_rope_core) {
        uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");
        unified_kernels::setup_sharded_buffer(krope_output_cb, 1);
    }
#elif defined(COMPILE_FOR_BRISC)
    deepseek_b1_ops::KVCacheUpdate::WriterArgs args{
        .kv_cache_buffer_base_addr = get_common_arg_val<uint32_t>(0),
        .position_id = get_common_arg_val<uint32_t>(1),
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb"),
        .krope_output_cb = get_named_compile_time_arg_val("krope_output_cb"),
        .grid_start_y = get_named_compile_time_arg_val("kv_cache_grid_start_y"),
    };
#elif defined(COMPILE_FOR_TRISC)
    deepseek_b1_ops::KVCacheUpdate::ComputeArgs args{
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
    };

    compute_kernel_hw_startup(0, 0, 0);
#endif

    deepseek_b1_ops::KVCacheUpdate::Op<Core::is_nope_core, Core::is_rope_core> op;
    op(args);
}
