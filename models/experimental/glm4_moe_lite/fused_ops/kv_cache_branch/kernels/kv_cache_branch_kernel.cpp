// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// GLM KV Cache Branch unified kernel (adapted from DSv3 KVCacheBranch)
//
// Fuses: DKV Matmul + RoPE. RMSNorm remains on the validated TTNN path.
// Kernel reads x, cos, sin from DRAM and writes nope+rope output to DRAM.
// Only gamma, weights, and trans_mat remain as sharded L1 tensors.
//
// RISC responsibilities:
// - NCRISC: Phase 0 (DRAM read x/cos/sin), setup sharded buffers,
//           RMSNorm scaler/eps fill, Phase 5 (DRAM write output)
// - BRISC: Wait for RoPE output CBs
// - TRISC: Matmul compute and RoPE compute

#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
// tensor_accessor.h transitively needs NOC_INDEX, which the JIT build only
// defines for NCRISC/BRISC (dataflow) kernels — never for TRISC/compute. Guard
// it so it does not leak into the compute translation unit (kernel_utils.hpp
// above already pulls dataflow_api.h in for NC/BR, and the only tensor_accessor::
// uses in this kernel live inside COMPILE_FOR_NCRISC blocks).
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/tensor/tensor_accessor.h"
#endif
#include "matmul_wormhole.hpp"
#include "rope_wormhole.hpp"

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

// ============================================================================
// BRISC (Writer)
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using DKV_MatmulCTArgs = glm4_matmul::Matmul::WriterCTArgs;

    glm4_matmul::Matmul::WriterArgs dkv_matmul_args{};

    using K_RopeCTArgs = glm4_rope::Rope::WriterCTArgs;
    glm4_rope::Rope::WriterArgs k_rope_args{};

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

    using K_RopeCTArgs =
        glm4_rope::Rope::ComputeCTArgs<get_named_compile_time_arg_val("Wt"), get_named_compile_time_arg_val("Ht")>;

    constexpr uint32_t k_rope_input_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
    constexpr uint32_t rotated_in_interm_cb = get_named_compile_time_arg_val("rotated_in_interm_cb");
    constexpr uint32_t cos_interm_cb = get_named_compile_time_arg_val("cos_interm_cb");
    constexpr uint32_t sin_interm_cb = get_named_compile_time_arg_val("sin_interm_cb");
    constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("out_cb");

    glm4_rope::Rope::ComputeArgs k_rope_args{
        .in_cb = k_rope_input_cb,
        .cos_cb = cos_cb,
        .sin_cb = sin_cb,
        .trans_mat_cb = trans_mat_cb,
        .rotated_in_interm_cb = rotated_in_interm_cb,
        .cos_interm_cb = cos_interm_cb,
        .sin_interm_cb = sin_interm_cb,
        .out_cb = k_rope_output_cb,
    };
    deepseek_compute_kernel_init();
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

        auto x_accessor =
            TensorAccessor(tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), x_dram_addr, x_total_bytes);
        uint64_t x_noc_addr = x_accessor.get_noc_addr(0);
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

        auto cos_accessor = TensorAccessor(
            tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), cos_dram_addr, cos_sin_page_size);
        auto sin_accessor = TensorAccessor(
            tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), sin_dram_addr, cos_sin_page_size);

        // Read cos tile for this core
        cb_reserve_back(cos_cb_id, Wt);
        {
            uint32_t cos_l1_addr = get_write_ptr(cos_cb_id);
            uint64_t cos_noc_addr = cos_accessor.get_noc_addr(0) + tile_offset * TILE_1x32_BYTES;
            noc_async_read(cos_noc_addr, cos_l1_addr, TILE_1x32_BYTES);
        }

        // Read sin tile for this core
        cb_reserve_back(sin_cb_id, Wt);
        {
            uint32_t sin_l1_addr = get_write_ptr(sin_cb_id);
            uint64_t sin_noc_addr = sin_accessor.get_noc_addr(0) + tile_offset * TILE_1x32_BYTES;
            noc_async_read(sin_noc_addr, sin_l1_addr, TILE_1x32_BYTES);
        }

        noc_async_read_barrier();
        cb_push_back(cos_cb_id, Wt);
        cb_push_back(sin_cb_id, Wt);
    }

    // Pull this core's 32-column weight shard directly from interleaved DRAM.
    // Tile order for [K, N] is k-major, so tile(k, core) is k * Nt + core.
    if constexpr (Core::is_dkv_matmul_core) {
        constexpr uint32_t dkv_matmul_in1 = get_named_compile_time_arg_val("dkv_matmul_in1");
        constexpr uint32_t dkv_matmul_k_num_tiles = get_named_compile_time_arg_val("dkv_matmul_k_num_tiles");
        constexpr uint32_t weight_tile_bytes = get_named_compile_time_arg_val("dkv_matmul_weight_tile_bytes");
        constexpr uint32_t weight_num_output_tiles =
            get_named_compile_time_arg_val("dkv_matmul_weight_num_output_tiles");
        constexpr uint32_t core_tile_offset = get_named_compile_time_arg_val("dkv_weight_core_tile_offset");
        uint32_t weights_dram_addr = get_common_arg_val<uint32_t>(8);
        auto weights_accessor = TensorAccessor(
            tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), weights_dram_addr, weight_tile_bytes);

        cb_reserve_back(dkv_matmul_in1, dkv_matmul_k_num_tiles);
        uint32_t weight_l1_addr = get_write_ptr(dkv_matmul_in1);
        for (uint32_t k = 0; k < dkv_matmul_k_num_tiles; ++k) {
            noc_async_read_page(
                k * weight_num_output_tiles + core_tile_offset,
                weights_accessor,
                weight_l1_addr + k * weight_tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(dkv_matmul_in1, dkv_matmul_k_num_tiles);
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
    // Phase 2: Gather is unnecessary while RMSNorm remains on the validated
    // TTNN path. Each no-PE matmul core writes its own 1x32 result directly.
    // ========================================================================
    {
        DeviceZoneScopedN("DKV_GATHER");
    }

    // ========================================================================
    // Phase 3: The TILE_1x32 reduction path cannot represent REDUCE_SCALAR's
    // 32x32 intermediate safely on Wormhole. Keep RMSNorm as a following TTNN
    // op and emit the gathered no-PE projection from this fused kernel.
    // ========================================================================
    {
        DeviceZoneScopedN("KV_RMSNORM");
        // Intentionally empty.
    }

    // ========================================================================
    // Phase 4: RoPE on k_rope data (on rope cores only)
    // NCRISC is skipped: cos/sin were loaded in Phase 0, trans_mat is sharded.
    // ========================================================================
    {
        DeviceZoneScopedN("K_ROPE");
#if !defined(COMPILE_FOR_NCRISC)
        glm4_rope::Rope::Op<K_RopeCTArgs, Core::is_krope_core> k_rope;
        k_rope(k_rope_args);
#endif
    }

    // ========================================================================
    // NCRISC Phase 5: Write nope+rope output to DRAM
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    {
        DeviceZoneScopedN("DRAM_WRITE_OUTPUT");

        uint32_t kvpe_out_dram_addr = get_common_arg_val<uint32_t>(7);

        constexpr uint32_t kvpe_out_dram_page_size = get_named_compile_time_arg_val("kvpe_out_dram_page_size");
        auto output_accessor = TensorAccessor(
            tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), kvpe_out_dram_addr, kvpe_out_dram_page_size);

        // No-PE matmul cores: write one naturally ordered output tile each.
        if constexpr (Core::is_knope_core) {
            constexpr uint32_t nope_output_cb = get_named_compile_time_arg_val("dkv_matmul_out");
            constexpr uint32_t tile_offset = get_named_compile_time_arg_val("knope_core_tile_offset");
            cb_wait_front(nope_output_cb, 1);
            uint32_t nope_l1_addr = get_read_ptr(nope_output_cb);
            uint64_t output_tile_noc_addr = output_accessor.get_noc_addr(tile_offset);
            // A conventional 32x32 tile stores its top row across the first
            // row of face 0 and face 1. Split the packed tiny-tile row into
            // those two locations; the pre-zeroed padding remains untouched.
            noc_async_write(nope_l1_addr, output_tile_noc_addr, 32);
            noc_async_write(nope_l1_addr + 32, output_tile_noc_addr + 512, 32);
        }

        // Rope cores: write rope (1 tile each) to output after nope
        if constexpr (Core::is_krope_core) {
            constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("k_rope_output_cb");
            constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
            constexpr uint32_t tile_offset = get_named_compile_time_arg_val("krope_core_tile_offset");
            cb_wait_front(k_rope_output_cb, Wt);
            uint32_t rope_l1_addr = get_read_ptr(k_rope_output_cb);
            uint64_t rope_tile_noc_addr = output_accessor.get_noc_addr(16 + tile_offset);
            noc_async_write(rope_l1_addr, rope_tile_noc_addr, 32);
            noc_async_write(rope_l1_addr + 32, rope_tile_noc_addr + 512, 32);
        }

        noc_async_write_barrier();
    }
#endif

#if defined(COMPILE_FOR_BRISC)
    // Wait for output CBs to be ready (synchronization — ensures compute completed)
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("k_rope_output_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
        cb_wait_front(k_rope_output_cb, Wt);
    }
#endif
}
