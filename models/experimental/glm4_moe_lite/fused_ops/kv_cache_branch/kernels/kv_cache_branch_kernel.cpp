// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// GLM KV Cache Branch unified kernel (adapted from DSv3 KVCacheBranch)
//
// Fuses: DKV Matmul + Gather + RMSNorm + RoPE
// Kernel reads x, cos, sin from DRAM and writes nope+rope output to DRAM.
// Only gamma, weights, and trans_mat remain as sharded L1 tensors.
//
// RISC responsibilities:
// - NCRISC: Phase 0 (DRAM read x/cos/sin), setup sharded buffers,
//           RMSNorm scaler/eps fill, Phase 5 (DRAM write output)
// - BRISC: Gather receiver, wait for output CBs
// - TRISC: Matmul compute, RMSNorm compute, RoPE compute

#include "api/tensor/tensor_accessor.h"
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

    // ========================================================================
    // NCRISC Phase 0: Read x, cos, sin from DRAM into CBs
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t TILE_1x32_BYTES = 64;  // 1 * 32 * sizeof(bf16)

    // Read x from DRAM on all matmul cores
    if constexpr (Core::is_dkv_matmul_core) {
        constexpr uint32_t dkv_matmul_in0 = get_named_compile_time_arg_val("dkv_matmul_in0");
        constexpr uint32_t dkv_matmul_k_num_tiles = get_named_compile_time_arg_val("dkv_matmul_k_num_tiles");
        constexpr uint32_t x_total_bytes = dkv_matmul_k_num_tiles * TILE_1x32_BYTES;

        uint32_t x_dram_addr = get_common_arg_val<uint32_t>(4);

        cb_reserve_back(dkv_matmul_in0, dkv_matmul_k_num_tiles);
        uint32_t l1_write_addr = get_write_ptr(dkv_matmul_in0);

        uint64_t x_noc_addr = tensor_accessor::get_dram_bank_base_offset(0, noc_index) + x_dram_addr;
        noc_async_read(x_noc_addr, l1_write_addr, x_total_bytes);
        noc_async_read_barrier();

        cb_push_back(dkv_matmul_in0, dkv_matmul_k_num_tiles);
    }

    // Read cos/sin from DRAM on rope cores
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t cos_cb_id = get_named_compile_time_arg_val("cos_cb");
        constexpr uint32_t sin_cb_id = get_named_compile_time_arg_val("sin_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
        constexpr uint32_t tile_offset = get_named_compile_time_arg_val("krope_core_tile_offset");
        constexpr uint32_t cos_sin_page_size = get_named_compile_time_arg_val("cos_sin_dram_page_size");

        uint32_t cos_dram_addr = get_common_arg_val<uint32_t>(5);
        uint32_t sin_dram_addr = get_common_arg_val<uint32_t>(6);

        uint64_t dram_bank0_base = tensor_accessor::get_dram_bank_base_offset(0, noc_index);

        // Read cos tile for this core
        cb_reserve_back(cos_cb_id, Wt);
        {
            uint32_t cos_l1_addr = get_write_ptr(cos_cb_id);
            uint64_t cos_noc_addr = dram_bank0_base + cos_dram_addr + tile_offset * TILE_1x32_BYTES;
            noc_async_read(cos_noc_addr, cos_l1_addr, TILE_1x32_BYTES);
        }

        // Read sin tile for this core
        cb_reserve_back(sin_cb_id, Wt);
        {
            uint32_t sin_l1_addr = get_write_ptr(sin_cb_id);
            uint64_t sin_noc_addr = dram_bank0_base + sin_dram_addr + tile_offset * TILE_1x32_BYTES;
            noc_async_read(sin_noc_addr, sin_l1_addr, TILE_1x32_BYTES);
        }

        noc_async_read_barrier();
        cb_push_back(cos_cb_id, Wt);
        cb_push_back(sin_cb_id, Wt);
    }

    // Setup remaining sharded persistent buffers (weights, gamma, trans_mat)
    if constexpr (Core::is_dkv_matmul_core) {
        constexpr uint32_t dkv_matmul_in1 = get_named_compile_time_arg_val("dkv_matmul_in1");
        constexpr uint32_t dkv_matmul_k_num_tiles = get_named_compile_time_arg_val("dkv_matmul_k_num_tiles");
        constexpr uint32_t dkv_matmul_out_w_per_core = get_named_compile_time_arg_val("dkv_matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(dkv_matmul_in1, dkv_matmul_k_num_tiles * dkv_matmul_out_w_per_core);
    }
    if constexpr (Core::is_kv_rmsnorm_core) {
        constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb");
        constexpr uint32_t gamma_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(gamma_cb, gamma_num_tiles);
    }
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t trans_mat_cb_id = get_named_compile_time_arg_val("trans_mat_cb");
        unified_kernels::setup_sharded_buffer(trans_mat_cb_id, 1);
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
    // ========================================================================
    {
        DeviceZoneScopedN("KV_RMSNORM");
        glm4_rmsnorm::RMSNorm::Op<KV_RMSNormCTArgs, Core::is_kv_rmsnorm_core, true> kv_rmsnorm;
        kv_rmsnorm(kv_rmsnorm_args);
    }

    // ========================================================================
    // Phase 4: RoPE on k_rope data (on rope cores only)
    // NCRISC is skipped: cos/sin were loaded in Phase 0, trans_mat is sharded.
    // ========================================================================
    {
        DeviceZoneScopedN("K_ROPE");
#if !defined(COMPILE_FOR_NCRISC)
        deepseek_b1_ops::Rope::Op<K_RopeCTArgs, Core::is_krope_core> k_rope;
        k_rope(k_rope_args);
#endif
    }

    // ========================================================================
    // NCRISC Phase 5: Write nope+rope output to DRAM
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    {
        DeviceZoneScopedN("DRAM_WRITE_OUTPUT");

        constexpr uint32_t kvpe_out_nope_bytes = get_named_compile_time_arg_val("kvpe_out_nope_bytes");
        uint32_t kvpe_out_dram_addr = get_common_arg_val<uint32_t>(7);

        uint64_t out_base_noc_addr = tensor_accessor::get_dram_bank_base_offset(0, noc_index) + kvpe_out_dram_addr;

        // RMSNorm core: write nope (16 tiles) to output offset 0
        if constexpr (Core::is_kv_rmsnorm_core) {
            constexpr uint32_t nope_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
            constexpr uint32_t nope_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
            cb_wait_front(nope_output_cb, nope_num_tiles);
            uint32_t nope_l1_addr = get_read_ptr(nope_output_cb);
            noc_async_write(nope_l1_addr, out_base_noc_addr, kvpe_out_nope_bytes);
        }

        // Rope cores: write rope (1 tile each) to output after nope
        if constexpr (Core::is_krope_core) {
            constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("k_rope_output_cb");
            constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
            constexpr uint32_t tile_offset = get_named_compile_time_arg_val("krope_core_tile_offset");
            cb_wait_front(k_rope_output_cb, Wt);
            uint32_t rope_l1_addr = get_read_ptr(k_rope_output_cb);
            uint64_t rope_dram_addr = out_base_noc_addr + kvpe_out_nope_bytes + tile_offset * TILE_1x32_BYTES;
            noc_async_write(rope_l1_addr, rope_dram_addr, Wt * TILE_1x32_BYTES);
        }

        noc_async_write_barrier();
    }
#endif

#if defined(COMPILE_FOR_BRISC)
    // Wait for output CBs to be ready (synchronization — ensures compute completed)
    if constexpr (Core::is_kv_rmsnorm_core) {
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
