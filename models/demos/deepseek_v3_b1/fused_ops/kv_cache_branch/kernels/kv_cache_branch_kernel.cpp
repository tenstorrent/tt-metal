// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// KV Cache Branch unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: Matmul + RMSNorm + Rope
// TODO: Implements: <list operations here, e.g., Matmul + Mcast + Gather>
// TODO: Define RISC responsibilities:
// - NCRISC: <describe data movement operations>
// - BRISC: <describe data movement operations>
// - TRISC: <describe compute operations>

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"
#include "../../../unified_kernels/rope.hpp"

// Compile-time role flags for dead code elimination via if constexpr
// Defined at namespace scope (local classes cannot have static data members)
struct Core {
    static constexpr bool is_dkv_matmul_core = get_named_compile_time_arg_val("is_dkv_matmul_core") == 1;
    static constexpr bool is_kv_rmsnorm_core = get_named_compile_time_arg_val("is_kv_rmsnorm_core") == 1;
    static constexpr bool is_knope_core = get_named_compile_time_arg_val("is_knope_core") == 1;
    static constexpr bool is_krope_core = get_named_compile_time_arg_val("is_krope_core") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Matmul CTArgs type alias (NCRISC uses ReaderCTArgs)
    using DKV_MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;

    // Matmul reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs dkv_matmul_args{};

    // Gather sender args (from compile-time args, passed to op as runtime args)
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
        get_write_ptr(get_named_compile_time_arg_val(
            "kv_rmsnorm_input_cb")),  // receiver_data_addr from CB write ptr (single-buffered)
    };

    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    // kv cache rmsnorm reader args
    deepseek_b1_ops::RMSNorm::ReaderArgs kv_rmsnorm_args{};

    using K_RopeCTArgs = deepseek_b1_ops::Rope::ReaderCTArgs<
        get_named_compile_time_arg_val("Wt"),
        get_named_compile_time_arg_val("Ht"),
        get_named_compile_time_arg_val("cos_sin_page_size"),
        get_named_compile_time_arg_val("total_Wt"),
        get_named_compile_time_arg_val("start_tile_offset")>;
    constexpr uint32_t k_rope_input_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t cos_tensor_address = get_named_compile_time_arg_val("cos_tensor_address");
    constexpr uint32_t sin_tensor_address = get_named_compile_time_arg_val("sin_tensor_address");
    constexpr uint32_t position_ids_tensor_address = get_named_compile_time_arg_val("position_ids_tensor_address");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");

    deepseek_b1_ops::Rope::ReaderArgs k_rope_args{
        .in_cb = k_rope_input_cb,
        .cos_cb = cos_cb,
        .sin_cb = sin_cb,
        .cos_tensor_address = cos_tensor_address,
        .sin_tensor_address = sin_tensor_address,
        .position_ids_tensor_address = position_ids_tensor_address,
        .trans_mat_cb = trans_mat_cb,
    };

// ============================================================================
// BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: TODO
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    using DKV_MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;

    // Matmul writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs dkv_matmul_args{};

    // Gather receiver args (from compile-time args, passed to op as runtime args)
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

    // Writer args (empty - no-op)
    deepseek_b1_ops::Rope::WriterArgs k_rope_args{};

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Named compile-time args: TODO
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using DKV_MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("dkv_matmul_out_w_per_core")>;

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Matmul::ComputeArgs dkv_matmul_args{
        get_named_compile_time_arg_val("dkv_matmul_in0"),
        get_named_compile_time_arg_val("dkv_matmul_in1"),
        get_named_compile_time_arg_val("dkv_matmul_out"),
        get_named_compile_time_arg_val("dkv_matmul_k_num_tiles"),
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs dkv_gather_args{};

    // CTArgs type aliases (required for Op templates)
    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("kv_rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("kv_rmsnorm_input_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_output_cb")>;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs kv_rmsnorm_args{
        get_common_arg_val<uint32_t>(0),  // epsilon
        get_common_arg_val<float>(1),     // scalar (1/sqrt(512))
    };

    using K_RopeCTArgs = deepseek_b1_ops::Rope::
        ComputeCTArgs<get_named_compile_time_arg_val("Wt"), get_named_compile_time_arg_val("Ht")>;

    // CB indices (passed as runtime args to ComputeArgs)
    constexpr uint32_t k_rope_input_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
    constexpr uint32_t rotated_in_interm_cb = get_named_compile_time_arg_val("rotated_in_interm_cb");
    constexpr uint32_t cos_interm_cb = get_named_compile_time_arg_val("cos_interm_cb");
    constexpr uint32_t sin_interm_cb = get_named_compile_time_arg_val("sin_interm_cb");
    constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("out_cb");

    // Compute args: all CB indices
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
    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);
#endif
#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers
    if constexpr (Core::is_dkv_matmul_core) {
        // Matmul activations (in0)
        constexpr uint32_t dkv_matmul_in0 = get_named_compile_time_arg_val("dkv_matmul_in0");
        constexpr uint32_t dkv_matmul_k_num_tiles = get_named_compile_time_arg_val("dkv_matmul_k_num_tiles");
        unified_kernels::setup_sharded_buffer(dkv_matmul_in0, dkv_matmul_k_num_tiles);

        // Matmul weights (in1)
        constexpr uint32_t dkv_matmul_in1 = get_named_compile_time_arg_val("dkv_matmul_in1");
        constexpr uint32_t dkv_matmul_out_w_per_core = get_named_compile_time_arg_val("dkv_matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(dkv_matmul_in1, dkv_matmul_k_num_tiles * dkv_matmul_out_w_per_core);
    }
    if constexpr (Core::is_kv_rmsnorm_core) {
        // RMSNorm gamma (sharded weights)
        constexpr uint32_t kv_rmsnorm_gamma_cb = get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb");
        constexpr uint32_t kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(kv_rmsnorm_gamma_cb, kv_rmsnorm_num_tiles);
    }
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
        unified_kernels::setup_sharded_buffer(trans_mat_cb, 1);
    }
#endif

    // ========================================================================
    // DKV Matmul
    // ========================================================================
    {
        DeviceZoneScopedN("DKV_MATMUL");
        // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
        deepseek_b1_ops::Matmul::Op<DKV_MatmulCTArgs, Core::is_dkv_matmul_core, true, false> dkv_matmul;
        dkv_matmul(dkv_matmul_args);
    }
    // ========================================================================
    // Gather: dkv matmul cores (senders) -> input core (receiver)
    // NCRISC sends from knope grid of dkv matmul cores, BRISC receives on rmsnorm grid, TRISC no-op
    // ========================================================================
    {
        DeviceZoneScopedN("DKV_GATHER");
        deepseek_b1_ops::Gather::Op<Core::is_knope_core, Core::is_kv_rmsnorm_core, true> dkv_gather;
        dkv_gather(dkv_gather_args);
    }

    // ========================================================================
    // RMSNorm: Apply RMSNorm to the gathered data
    {
        DeviceZoneScopedN("KV_RMSNORM");
        deepseek_b1_ops::RMSNorm::Op<KV_RMSNormCTArgs, Core::is_kv_rmsnorm_core, true> kv_rmsnorm;
        kv_rmsnorm(kv_rmsnorm_args);
    }

    // ========================================================================
    // Rope: Apply Rope to the gathered data
    // ========================================================================
    {
        DeviceZoneScopedN("K_ROPE");
        deepseek_b1_ops::Rope::Op<K_RopeCTArgs, Core::is_krope_core> k_rope;
        k_rope(k_rope_args);
    }
    // ========================================================================
    // KV Cache Update: Write results to DRAM interleaved tensor
    // BRISC handles writing from output CBs to DRAM
    // ========================================================================
#if defined(COMPILE_FOR_BRISC)
    // Unit testing the KV Cache write to DRAM.
    // Support 8 shards, one per DRAM core, each shard is 576x2 bytes (BFLOAT16)
    // KNOPE writes to first 512x2 bytes, each KROPE core writes to the remaining 64x2 bytes.
    DeviceZoneScopedN("KV_CACHE_UPDATE");
    // Get runtime args: buffer address and starting tile ID
    uint32_t kv_cache_buffer_addr = get_common_arg_val<uint32_t>(0);
    uint32_t position_ids_addr = get_common_arg_val<uint32_t>(1);

    volatile tt_l1_ptr uint32_t* position_ids_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(position_ids_addr);
    uint32_t kv_cache_start_tile_id = position_ids_ptr[0];

    // Create TensorAccessor for DRAM interleaved tensor
    auto kv_cache_addr_gen = TensorAccessor(
        tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(),
        kv_cache_buffer_addr,
        576 * 2  // page_size
    );
    // Actual calculations will differ in deepseek fused kernel, depending on the sharding scheme
    // Write RMSNorm output (nope portion) to KV cache
    if constexpr (Core::is_kv_rmsnorm_core) {
        constexpr uint32_t kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb");
        constexpr uint32_t kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");

        // Get tile size from the output CB
        uint32_t tile_size = get_tile_size(kv_rmsnorm_output_cb);

        cb_wait_front(kv_rmsnorm_output_cb, kv_rmsnorm_num_tiles);
        uint32_t l1_read_addr = get_read_ptr(kv_rmsnorm_output_cb);

        for (uint32_t tile_idx = 0; tile_idx < kv_rmsnorm_num_tiles; tile_idx++) {
            uint32_t tile_id = kv_cache_start_tile_id + tile_idx;
            noc_async_write_page(tile_id, kv_cache_addr_gen, l1_read_addr, tile_size, 0);
            l1_read_addr += tile_size;
        }
        noc_async_write_barrier();
        cb_pop_front(kv_rmsnorm_output_cb, kv_rmsnorm_num_tiles);
    }

    // Write Rope output to KV cache (after nope tiles)
    if constexpr (Core::is_krope_core) {
        constexpr uint32_t k_rope_output_cb = get_named_compile_time_arg_val("k_rope_output_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");

        // Get tile size from the output CB
        uint32_t tile_size = get_tile_size(k_rope_output_cb);

        cb_wait_front(k_rope_output_cb, Wt);
        uint32_t l1_read_addr = get_read_ptr(k_rope_output_cb);

        uint32_t rope_offset = get_absolute_logical_y() - 8;  // yea...
        for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
            // Rope tiles come after nope tiles
            uint32_t tile_id = kv_cache_start_tile_id + tile_idx;
            noc_async_write_page(
                tile_id, kv_cache_addr_gen, l1_read_addr, tile_size, 512 * 2 + tile_size * rope_offset);
            l1_read_addr += tile_size;
        }
        noc_async_write_barrier();
        cb_pop_front(k_rope_output_cb, Wt);
    }
#endif
}
