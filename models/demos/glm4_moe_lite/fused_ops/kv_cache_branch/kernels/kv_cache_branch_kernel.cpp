// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// GLM KV Cache Branch unified kernel (adapted from DSv3 KVCacheBranch)
//
// Fuses: DKV Matmul + Gather + RMSNorm + RoPE
// Output: RMSNorm'd nope data in nope_output_cb, RoPE result in k_rope_output_cb
// KV cache write is NOT done here — caller uses paged_update_cache separately.
//
// RISC responsibilities:
// - NCRISC: Setup sharded buffers, gather sender, RMSNorm scaler/eps fill
// - BRISC: Gather receiver, wait for output CBs
// - TRISC: Matmul compute, RMSNorm compute, RoPE compute

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "matmul_wormhole.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/gather.hpp"
#include "rmsnorm_wormhole.hpp"
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
    using DKV_MatmulCTArgs = glm4_matmul::Matmul::ReaderCTArgs;
    glm4_matmul::Matmul::ReaderArgs dkv_matmul_args{};

    deepseek_b1_ops::Gather::SenderArgs dkv_gather_args{
        get_named_compile_time_arg_val("dkv_gather_dest_noc_x"),
        get_named_compile_time_arg_val("dkv_gather_dest_noc_y"),
        get_named_compile_time_arg_val("dkv_gather_data_size_bytes"),
        get_semaphore(get_named_compile_time_arg_val("dkv_gather_receiver_semaphore_id")),
        get_named_compile_time_arg_val("dkv_gather_src_cb"),
        get_named_compile_time_arg_val("dkv_gather_src_num_pages"),
        0,  // sender_grid_start_x (unused with UsePerCoreSenderIdx=true)
        0,  // sender_grid_start_y
        0,  // sender_grid_end_x
        0,  // sender_grid_end_y
        1,  // row_major (unused with UsePerCoreSenderIdx=true)
        get_write_ptr(get_named_compile_time_arg_val("dkv_gather_dst_cb")),
        get_named_compile_time_arg_val("dkv_gather_sender_idx"),
    };

    // RMSNorm reader args (runtime — scaler/eps values passed at dispatch time)
    using KV_RMSNormCTArgs = glm4_rmsnorm::RMSNorm::ReaderCTArgs;

    // No RoPE NCRISC declarations needed: cos/sin/trans_mat are sharded
    // (already in L1). The NCRISC RoPE reader is skipped in Phase 4.

// ============================================================================
// BRISC (Writer)
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using DKV_MatmulCTArgs = glm4_matmul::Matmul::WriterCTArgs;

    glm4_matmul::Matmul::WriterArgs dkv_matmul_args{};

    deepseek_b1_ops::Gather::ReceiverArgs dkv_gather_args{
        get_named_compile_time_arg_val("dkv_gather_noc0_num_senders"),
        get_named_compile_time_arg_val("dkv_gather_noc1_num_senders"),
        get_semaphore(get_named_compile_time_arg_val("dkv_gather_noc0_receiver_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("dkv_gather_noc1_receiver_semaphore_id")),
        get_named_compile_time_arg_val("dkv_gather_dst_cb"),
        get_named_compile_time_arg_val("dkv_gather_dst_num_pages"),
    };

    using KV_RMSNormCTArgs = glm4_rmsnorm::RMSNorm::WriterCTArgs;

    using K_RopeCTArgs = deepseek_b1_ops::Rope::WriterCTArgs;
    deepseek_b1_ops::Rope::WriterArgs k_rope_args{};

// ============================================================================
// TRISC (Compute)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using DKV_MatmulCTArgs =
        glm4_matmul::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("dkv_matmul_out_w_per_core")>;

    glm4_matmul::Matmul::ComputeArgs dkv_matmul_args{
        get_named_compile_time_arg_val("dkv_matmul_in0"),
        get_named_compile_time_arg_val("dkv_matmul_in1"),
        get_named_compile_time_arg_val("dkv_matmul_out"),
        get_named_compile_time_arg_val("dkv_matmul_k_num_tiles"),
    };

    deepseek_b1_ops::Gather::ComputeArgs dkv_gather_args{};

    // RMSNorm compute args
    constexpr uint32_t kv_nope_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
    using KV_RMSNormCTArgs = glm4_rmsnorm::RMSNorm::ComputeCTArgs<false, kv_nope_num_tiles, true>;

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

    // RMSNorm runtime args (same struct on all RISCs, populated from common runtime args)
    glm4_rmsnorm::RMSNorm::RTArgs kv_rmsnorm_args{};
#if defined(COMPILE_FOR_NCRISC)
    kv_rmsnorm_args = {
        get_common_arg_val<uint32_t>(0),  // cb_scaler
        get_common_arg_val<uint32_t>(1),  // cb_eps
        get_common_arg_val<uint32_t>(2),  // scaler_packed (bf16 as uint16)
        get_common_arg_val<uint32_t>(3),  // eps_packed (bf16 as uint16)
    };
#elif defined(COMPILE_FOR_TRISC)
    kv_rmsnorm_args = {
        get_common_arg_val<uint32_t>(0),  // input_cb (gather dst = rmsnorm input)
        get_common_arg_val<uint32_t>(1),  // gamma_cb
        get_common_arg_val<uint32_t>(2),  // output_cb (nope output tensor)
        get_common_arg_val<uint32_t>(3),  // cb_x2
        get_common_arg_val<uint32_t>(4),  // cb_var
        get_common_arg_val<uint32_t>(5),  // cb_scaler
        get_common_arg_val<uint32_t>(6),  // cb_eps
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
        // Setup gamma sharded buffer on rmsnorm core
        constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb");
        constexpr uint32_t gamma_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(gamma_cb, gamma_num_tiles);
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
        glm4_matmul::Matmul::Op<DKV_MatmulCTArgs, Core::is_dkv_matmul_core, true, false> dkv_matmul;
        dkv_matmul(dkv_matmul_args);
    }

    // ========================================================================
    // Phase 2: Gather — knope matmul cores -> rmsnorm core
    // ========================================================================
    {
        DeviceZoneScopedN("DKV_GATHER");
        deepseek_b1_ops::Gather::Op<Core::is_knope_core, Core::is_kv_rmsnorm_core, true, true> dkv_gather;
        dkv_gather(dkv_gather_args);
    }

    // ========================================================================
    // Phase 3: RMSNorm on gathered nope data (on rmsnorm core only)
    // Input: gather dst CB (16 tiles of 1x32, pushed by gather receiver)
    // Output: nope output CB (16 tiles of 1x32, backed by output tensor)
    // ========================================================================
    {
        DeviceZoneScopedN("KV_RMSNORM");
        glm4_rmsnorm::RMSNorm::Op<KV_RMSNormCTArgs, Core::is_kv_rmsnorm_core, true> kv_rmsnorm;
        kv_rmsnorm(kv_rmsnorm_args);
    }

    // ========================================================================
    // Phase 4: RoPE on k_rope data (on rope cores only)
    //
    // NCRISC is skipped: cos/sin/trans_mat are already in L1 as sharded
    // buffers (populated by setup_sharded_buffer above). The generic
    // Rope::Op NCRISC reader would try to re-read them from DRAM using
    // uninitialized addresses (cos_tensor_address=0, position_ids=0),
    // causing a NOC hang on an invalid DRAM read.
    // ========================================================================
    {
        DeviceZoneScopedN("K_ROPE");
#if !defined(COMPILE_FOR_NCRISC)
        deepseek_b1_ops::Rope::Op<K_RopeCTArgs, Core::is_krope_core> k_rope;
        k_rope(k_rope_args);
#endif
    }

    // Output: RMSNorm'd nope in nope_output_cb (on rmsnorm core)
    //         RoPE result in k_rope_output_cb (on rope core)
    // These are backed by sharded output tensors — no explicit DRAM write needed.

#if defined(COMPILE_FOR_BRISC)
    // Wait for output CBs to be ready
    if constexpr (Core::is_kv_rmsnorm_core) {
        // Wait for RMSNorm output (which writes to the nope output tensor CB)
        constexpr uint32_t nope_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
        constexpr uint32_t nope_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        cb_wait_front(nope_output_cb, nope_num_tiles);
    }
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("k_rope_output_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
        cb_wait_front(k_rope_output_cb, Wt);
    }
#endif
}
