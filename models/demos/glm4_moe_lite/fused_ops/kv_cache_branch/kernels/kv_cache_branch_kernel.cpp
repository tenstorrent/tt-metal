// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// GLM KV Cache Branch unified kernel (adapted from DSv3 KVCacheBranch)
//
// Fuses: DKV Matmul + Gather + RMSNorm + RoPE
// Output: RMSNorm result (nope) in kv_rmsnorm_output_cb, RoPE result (rope) in k_rope_output_cb
// KV cache write is NOT done here — caller uses paged_update_cache separately.
//
// RISC responsibilities:
// - NCRISC: Setup sharded buffers, gather sender (knope->rmsnorm), signal rope buffers
// - BRISC: Gather receiver, wait for output CBs
// - TRISC: Matmul compute, RMSNorm compute, RoPE compute

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/matmul.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/gather.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/rmsnorm.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/rope.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_dkv_matmul_core = get_named_compile_time_arg_val("is_dkv_matmul_core") == 1;
    static constexpr bool is_kv_rmsnorm_core = get_named_compile_time_arg_val("is_kv_rmsnorm_core") == 1;
    static constexpr bool is_knope_core = get_named_compile_time_arg_val("is_knope_core") == 1;
    static constexpr bool is_krope_core = get_named_compile_time_arg_val("is_krope_core") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using DKV_MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs dkv_matmul_args{};

    deepseek_b1_ops::Gather::SenderArgs dkv_gather_args{
        get_named_compile_time_arg_val("dkv_gather_dest_noc_x"),
        get_named_compile_time_arg_val("dkv_gather_dest_noc_y"),
        get_named_compile_time_arg_val("dkv_gather_data_size_bytes"),
        get_named_compile_time_arg_val("dkv_gather_receiver_semaphore_id"),
        get_named_compile_time_arg_val("dkv_gather_src_cb"),
        get_named_compile_time_arg_val("dkv_gather_src_num_pages"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_start_x"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_start_y"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_end_x"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_end_y"),
        get_named_compile_time_arg_val("dkv_gather_row_major"),
        get_write_ptr(get_named_compile_time_arg_val("kv_rmsnorm_input_cb")),
    };

    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    deepseek_b1_ops::RMSNorm::ReaderArgs kv_rmsnorm_args{};

    using K_RopeCTArgs =
        deepseek_b1_ops::Rope::ReaderCTArgs<get_named_compile_time_arg_val("Wt"), get_named_compile_time_arg_val("Ht")>;
    constexpr uint32_t k_rope_input_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");

    deepseek_b1_ops::Rope::ReaderArgs k_rope_args{
        .in_cb = k_rope_input_cb,
        .cos_cb = cos_cb,
        .sin_cb = sin_cb,
        .trans_mat_cb = trans_mat_cb,
    };

// ============================================================================
// BRISC (Writer)
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    using DKV_MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;

    deepseek_b1_ops::Matmul::WriterArgs dkv_matmul_args{};

    deepseek_b1_ops::Gather::ReceiverArgs dkv_gather_args{
        get_named_compile_time_arg_val("dkv_gather_noc0_num_senders"),
        get_named_compile_time_arg_val("dkv_gather_noc1_num_senders"),
        get_named_compile_time_arg_val("dkv_gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("dkv_gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("dkv_gather_dst_cb"),
        get_named_compile_time_arg_val("dkv_gather_dst_num_pages"),
    };

    deepseek_b1_ops::RMSNorm::WriterArgs kv_rmsnorm_args{};

    using K_RopeCTArgs = deepseek_b1_ops::Rope::WriterCTArgs;
    deepseek_b1_ops::Rope::WriterArgs k_rope_args{};

// ============================================================================
// TRISC (Compute)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using DKV_MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("dkv_matmul_out_w_per_core")>;

    deepseek_b1_ops::Matmul::ComputeArgs dkv_matmul_args{
        get_named_compile_time_arg_val("dkv_matmul_in0"),
        get_named_compile_time_arg_val("dkv_matmul_in1"),
        get_named_compile_time_arg_val("dkv_matmul_out"),
        get_named_compile_time_arg_val("dkv_matmul_k_num_tiles"),
    };

    deepseek_b1_ops::Gather::ComputeArgs dkv_gather_args{};

    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("kv_rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1>;

    deepseek_b1_ops::RMSNorm::ComputeArgs kv_rmsnorm_args{
        get_named_compile_time_arg_val("kv_rmsnorm_input_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_output_cb"),
        get_common_arg_val<uint32_t>(0),  // epsilon
        get_common_arg_val<float>(1),     // scalar (1/sqrt(512))
    };

    using K_RopeCTArgs = deepseek_b1_ops::Rope::
        ComputeCTArgs<get_named_compile_time_arg_val("Wt"), get_named_compile_time_arg_val("Ht")>;

    constexpr uint32_t k_rope_input_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
    constexpr uint32_t rotated_in_interm_cb = get_named_compile_time_arg_val("rotated_in_interm_cb");
    constexpr uint32_t cos_interm_cb = get_named_compile_time_arg_val("cos_interm_cb");
    constexpr uint32_t sin_interm_cb = get_named_compile_time_arg_val("sin_interm_cb");
    constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("out_cb");

    deepseek_b1_ops::Rope::ComputeArgs k_rope_args{
        .in_cb = k_rope_input_cb,
        .cos_cb = cos_cb,
        .sin_cb = sin_cb,
        .trans_mat_cb = trans_mat_cb,
        .rotated_in_interm_cb = rotated_in_interm_cb,
        .cos_interm_cb = cos_interm_cb,
        .sin_interm_cb = sin_interm_cb,
        .out_cb = k_rope_output_cb,
    };
#endif

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers
    if constexpr (Core::is_dkv_matmul_core) {
        constexpr uint32_t dkv_matmul_in0 = get_named_compile_time_arg_val("dkv_matmul_in0");
        constexpr uint32_t dkv_matmul_k_num_tiles = get_named_compile_time_arg_val("dkv_matmul_k_num_tiles");
        unified_kernels::setup_sharded_buffer(dkv_matmul_in0, dkv_matmul_k_num_tiles);

        constexpr uint32_t dkv_matmul_in1 = get_named_compile_time_arg_val("dkv_matmul_in1");
        constexpr uint32_t dkv_matmul_out_w_per_core = get_named_compile_time_arg_val("dkv_matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(dkv_matmul_in1, dkv_matmul_k_num_tiles * dkv_matmul_out_w_per_core);
    }
    if constexpr (Core::is_kv_rmsnorm_core) {
        constexpr uint32_t kv_rmsnorm_gamma_cb = get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb");
        constexpr uint32_t kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(kv_rmsnorm_gamma_cb, kv_rmsnorm_num_tiles);
    }
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
        constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
        constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
        unified_kernels::setup_sharded_buffer(cos_cb, Wt);
        unified_kernels::setup_sharded_buffer(sin_cb, Wt);
        unified_kernels::setup_sharded_buffer(trans_mat_cb, 1);
    }
#endif

    // ========================================================================
    // Phase 1: DKV Matmul
    // ========================================================================
    {
        DeviceZoneScopedN("DKV_MATMUL");
        deepseek_b1_ops::Matmul::Op<DKV_MatmulCTArgs, Core::is_dkv_matmul_core, true, false> dkv_matmul;
        dkv_matmul(dkv_matmul_args);
    }

    // ========================================================================
    // Phase 2: Gather — knope matmul cores -> rmsnorm core
    // ========================================================================
    {
        DeviceZoneScopedN("DKV_GATHER");
        deepseek_b1_ops::Gather::Op<Core::is_knope_core, Core::is_kv_rmsnorm_core, true> dkv_gather;
        dkv_gather(dkv_gather_args);
    }

    // ========================================================================
    // Phase 3: RMSNorm on gathered kv_nope data
    // ========================================================================
    {
        DeviceZoneScopedN("KV_RMSNORM");
        deepseek_b1_ops::RMSNorm::Op<KV_RMSNormCTArgs, Core::is_kv_rmsnorm_core, true> kv_rmsnorm;
        kv_rmsnorm(kv_rmsnorm_args);
    }

    // ========================================================================
    // Phase 4: RoPE on k_rope data
    // ========================================================================
    {
        DeviceZoneScopedN("K_ROPE");
        deepseek_b1_ops::Rope::Op<K_RopeCTArgs, Core::is_krope_core> k_rope;
        k_rope(k_rope_args);
    }

    // Output: RMSNorm result sits in kv_rmsnorm_output_cb (on rmsnorm core)
    //         RoPE result sits in k_rope_output_cb (on rope core)
    // These are backed by sharded output tensors — no explicit DRAM write needed.
    // The caller reads them back via the output tensor shard spec.

#if defined(COMPILE_FOR_BRISC)
    // Wait for output CBs to be ready (compute has pushed them)
    if constexpr (Core::is_kv_rmsnorm_core) {
        constexpr uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
        constexpr uint32_t kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        cb_wait_front(kv_rmsnorm_output_cb, kv_rmsnorm_num_tiles);
    }
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("k_rope_output_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
        cb_wait_front(k_rope_output_cb, Wt);
    }
#endif
}
