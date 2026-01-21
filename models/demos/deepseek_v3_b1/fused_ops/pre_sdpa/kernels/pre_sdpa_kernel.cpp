// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Pre-SDPA unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: RMSNorm + Mcast + Matmul + Gather + RMSNorm2 + Mcast2 + Matmul2
// - NCRISC: RMSNorm reader + Mcast receiver (on matmul cores), Matmul reader + Gather sender (on matmul cores),
//           RMSNorm2 reader + Mcast2 receiver (on matmul2 cores), Matmul2 reader (on matmul2 cores)
// - BRISC: RMSNorm writer + Mcast sender (on input core), Matmul writer (on matmul cores), Gather receiver (on
//          input core), Mcast2 sender (on input core), Matmul2 writer (on matmul2 cores)
// - TRISC: RMSNorm compute (on input core), Matmul compute (on matmul cores), RMSNorm2 compute (on input core),
//          Matmul2 compute (on matmul2 cores)
//
// Matmul2 output uses interleaved Qnope/Qrope layout (with shuffled weights):
// - Grid: 12 cols × 8 rows = 96 cores (P150)
// - Qnope cores (cols 0-7): 64 cores, 1 head × 128 elements per core
// - Qrope cores (cols 8-11): 32 cores, 2 heads × 64 elements per core
// - Each row: [Qnope heads 0-7 (1024)] [Qrope heads 0-7 (512)] = 1536 elements
// - Total: 8 rows × 1536 = 12288 elements

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"

// Compile-time role flags for dead code elimination via if constexpr
// Defined at namespace scope (local classes cannot have static data members)
struct Core {
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_matmul2_core = get_named_compile_time_arg_val("is_matmul2_core") == 1;
    // Qnope/Qrope core differentiation for interleaved Q head layout after matmul2
    // Qnope cores: 64 cores (8x8 grid), each handles 1 head of 128 elements
    // Qrope cores: 32 cores (4x8 grid), each handles 2 heads of 64 elements
    static constexpr bool is_qnope_core = get_named_compile_time_arg_val("is_qnope_core") == 1;
    static constexpr bool is_qrope_core = get_named_compile_time_arg_val("is_qrope_core") == 1;
    // SDPA core: receives interleaved QNOPE/QROPE unicasts (1 per row = 8 cores)
    static constexpr bool is_sdpa_core = get_named_compile_time_arg_val("is_sdpa_core") == 1;
};

KERNEL_ENTRY {
// ============================================================================
// NCRISC (Reader + Mcast Receiver) - ReaderConfigDescriptor compiles as NCRISC
// Named compile-time args: rmsnorm reader, mcast receiver, matmul reader, gather sender
// Runtime args: [scalar, scalar2]
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs<get_named_compile_time_arg_val("rmsnorm_num_faces")>;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs<get_named_compile_time_arg_val("rmsnorm2_num_faces")>;
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;

    // RMSNorm reader runtime args
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_arg_val<uint32_t>(0),  // scalar (1/sqrt(7168))
    };

    // Mcast receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Matmul CTArgs type alias (NCRISC uses ReaderCTArgs)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    using Matmul3CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;

    // Matmul reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    // Gather sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::SenderArgs gather_args{
        get_named_compile_time_arg_val("gather_dest_noc_x"),
        get_named_compile_time_arg_val("gather_dest_noc_y"),
        get_named_compile_time_arg_val("gather_data_size_bytes"),
        get_named_compile_time_arg_val("gather_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_src_cb"),
        get_named_compile_time_arg_val("gather_src_num_pages"),
        get_named_compile_time_arg_val("gather_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather_row_major"),
        get_write_ptr(get_named_compile_time_arg_val(
            "rmsnorm2_input_cb")),  // receiver_data_addr from CB write ptr (single-buffered)
    };

    // RMSNorm2 reader args (uses same scalars_cb, different scalar value)
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm2_args{
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_arg_val<uint32_t>(1),  // scalar2 (1/sqrt(1536))
    };

    // Matmul2 reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul2_args{};

    // Mcast2 receiver args (for matmul2 cores to receive matmul2 input from input core)
    // Uses same semaphore as first mcast
    deepseek_b1_ops::Mcast::ReceiverArgs mcast2_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("matmul2_in0"),
        get_named_compile_time_arg_val("mcast2_dst_num_pages"),
    };

    // Matmul3 reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul3_args{};

// ============================================================================
// BRISC (Writer + Mcast Sender) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: rmsnorm writer, mcast sender, matmul writer, gather receiver
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;  // BRISC is no-op
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_input_core && Core::is_matmul2_core>;  // Always mcast to the main grid

    // RMSNorm writer args (BRISC is no-op)
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

    // RMSNorm2 writer args (BRISC is no-op)
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm2_args{};

    // Mcast CB indices from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");

    // Mcast sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Mcast::SenderArgs mcast_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_data_size_bytes"),
        mcast_src_cb,
        get_named_compile_time_arg_val("mcast_src_num_pages"),
        get_read_ptr(mcast_src_cb),
        get_write_ptr(mcast_dst_cb),
    };

    // Matmul CTArgs type alias (BRISC uses WriterCTArgs)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul3CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;

    // Matmul writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

    // Matmul3 writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs matmul3_args{};

    // Gather receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::ReceiverArgs gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
    };

    // Matmul2 writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs matmul2_args{};

    // Matmul2 CB indices and parameters from named compile-time args
    constexpr uint32_t matmul2_in0 = get_named_compile_time_arg_val("matmul2_in0");

    // Mcast2 sender args (for input core to mcast rmsnorm2 output to all matmul2 cores)
    // Uses same grid and semaphores as first mcast
    // Reads from rmsnorm2_output_cb, writes to matmul2_in0 with loopback
    constexpr uint32_t mcast2_src_cb = get_named_compile_time_arg_val("rmsnorm2_output_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast2_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast2_data_size_bytes"),
        mcast2_src_cb,  // Wait for rmsnorm2_output_cb
        get_named_compile_time_arg_val("mcast2_src_num_pages"),
        get_read_ptr(mcast2_src_cb),  // Read from rmsnorm2_output_cb
        get_write_ptr(matmul2_in0),   // Write to matmul2_in0 (loopback)
    };

    // Matmul3 is implemented inline in TRISC

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Named compile-time args: rmsnorm compute, matmul compute
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1>;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm2_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1>;
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_named_compile_time_arg_val("rmsnorm_interm_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_arg_val<uint32_t>(0),  // epsilon
    };

    // Mcast compute args (no-op for TRISC)
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul CTArgs type alias (out_w is compile-time for TRISC)
    using MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs gather_args{};

    // RMSNorm2 compute args (separate CBs with exact sizes for testing)
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm2_args{
        get_named_compile_time_arg_val("rmsnorm2_input_cb"),   // separate input CB (3 tiles of 16x32)
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),  // reuse scalars cb
        get_named_compile_time_arg_val("rmsnorm2_interm_cb"),  // separate interm CB (3 tiles)
        get_named_compile_time_arg_val("rmsnorm2_gamma_cb"),   // new gamma for 1536 elements
        get_named_compile_time_arg_val("rmsnorm2_output_cb"),  // separate output CB (3 tiles of 16x32)
        get_arg_val<uint32_t>(0),                              // epsilon (same as rmsnorm1)
    };

    // Matmul2 CTArgs type alias (out_w is compile-time for TRISC)
    using Matmul2CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul2_out_w_per_core")>;

    // Matmul2 compute args (from compile-time args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul2_args{
        get_named_compile_time_arg_val("matmul2_in0"),
        get_named_compile_time_arg_val("matmul2_in1"),
        get_named_compile_time_arg_val("matmul2_out"),
        get_named_compile_time_arg_val("matmul2_k_num_tiles"),
    };

    // Mcast2 compute args (no-op for TRISC)
    deepseek_b1_ops::Mcast::ComputeArgs mcast2_args{};

    // Matmul3 CTArgs type alias (out_w is compile-time for TRISC)
    using Matmul3CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul3_out_w_per_core")>;

    // Matmul3 compute args (from compile-time args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul3_args{
        get_named_compile_time_arg_val("matmul3_in0"),
        get_named_compile_time_arg_val("matmul3_in1"),
        get_named_compile_time_arg_val("matmul3_out"),
        get_named_compile_time_arg_val("matmul3_k_num_tiles"),
    };
#endif

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers
    if constexpr (Core::is_input_core) {
        // RMSNorm input and gamma buffers
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);

        // RMSNorm2 gamma buffer (3 tiles of 16x32)
        constexpr uint32_t rmsnorm2_gamma_cb = get_named_compile_time_arg_val("rmsnorm2_gamma_cb");
        constexpr uint32_t rmsnorm2_num_tiles = get_named_compile_time_arg_val("rmsnorm2_num_tiles");
        unified_kernels::setup_sharded_buffer(rmsnorm2_gamma_cb, rmsnorm2_num_tiles);
    }
    if constexpr (Core::is_matmul_core) {
        // Matmul weights
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_k_num_tiles = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t matmul_out_w_per_core = get_named_compile_time_arg_val("matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul_in1, matmul_k_num_tiles * matmul_out_w_per_core);
    }
    if constexpr (Core::is_matmul2_core) {
        // Matmul2 CB indices and parameters from named compile-time args
        constexpr uint32_t matmul2_in1 = get_named_compile_time_arg_val("matmul2_in1");
        constexpr uint32_t matmul2_k_num_tiles = get_named_compile_time_arg_val("matmul2_k_num_tiles");
        constexpr uint32_t matmul2_out_w_per_core = get_named_compile_time_arg_val("matmul2_out_w_per_core");

        // Matmul2 weights (on all cores in main grid, 4 tiles per core)
        unified_kernels::setup_sharded_buffer(matmul2_in1, matmul2_k_num_tiles * matmul2_out_w_per_core);
    }
    if constexpr (Core::is_qnope_core) {
        // Matmul3 CB indices and parameters from named compile-time args
        constexpr uint32_t matmul3_in1 = get_named_compile_time_arg_val("matmul3_in1");
        constexpr uint32_t matmul3_k_num_tiles = get_named_compile_time_arg_val("matmul3_k_num_tiles");
        constexpr uint32_t matmul3_out_w_per_core = get_named_compile_time_arg_val("matmul3_out_w_per_core");

        // Matmul3 weights (on Qnope cores, [128, 512] = 4 * 16 = 64 tiles per core)
        unified_kernels::setup_sharded_buffer(matmul3_in1, matmul3_k_num_tiles * matmul3_out_w_per_core);
    }
#endif

    // ========================================================================
    // Input core: RMSNorm + Mcast send
    // ========================================================================
    {
        DeviceZoneScopedN("RMSNORM");
        // pop_input = true (input is consumed after RMSNorm)
        deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_input_core, true> rmsnorm;
        rmsnorm(rmsnorm_args);
    }

    // pop_src = true (rmsnorm output is consumed after mcast)
    deepseek_b1_ops::Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul2_core, Core::is_matmul_core, true>
        mcast;
    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST");
        // Mcast: NCRISC sends from input core, BRISC receives on matmul cores, TRISC no-op
        // pop_src = true (input is consumed after mcast)
        mcast(mcast_args);
    }

    // ========================================================================
    // Matmul operation
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
        deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, false> matmul;
        matmul(matmul_args);
    }

    // ========================================================================
    // Gather: matmul cores (senders) -> input core (receiver)
    // NCRISC sends from matmul cores, BRISC receives on input core, TRISC no-op
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        // pop_src = true (matmul output is consumed after gather)
        deepseek_b1_ops::Gather::Op<Core::is_matmul_core, Core::is_input_core, true> gather;
        gather(gather_args);
    }

    // ========================================================================
    // RMSNorm2: Apply RMSNorm to the gathered data (1536 elements = 3 tiles of 16x32)
    // Gather writes directly to rmsnorm2_input_cb (3 tiles of 16x32)
    // Uses SEPARATE CBs with exact sizes:
    //   - Input: rmsnorm2_input_cb (3 tiles from gather)
    //   - Interm: rmsnorm2_interm_cb (3 tiles)
    //   - Output: rmsnorm2_output_cb (3 tiles)
    //   - Gamma: rmsnorm2_gamma_cb (3 tiles)
    //   - Scalars: reuses scalars_cb (same epsilon, different scalar)
    // ========================================================================
    // pop_input = true (gathered data is consumed after RMSNorm2)
    {
        DeviceZoneScopedN("RMSNORM2");
        deepseek_b1_ops::RMSNorm::Op<RMSNorm2CTArgs, Core::is_input_core, true> rmsnorm2;
        rmsnorm2(rmsnorm2_args);
    }

    // ========================================================================
    // Mcast2: Broadcast rmsnorm2 output from input core to all matmul2 cores
    // Reads from rmsnorm2_output_cb, writes to matmul2_in0 with loopback
    // Uses same grid and semaphores as first mcast
    // ========================================================================
    // pop_src = true (rmsnorm2 output is consumed after mcast)
    {
        DeviceZoneScopedN("MCAST2");
        // Mcast2: NCRISC sends from input core, BRISC receives on matmul2 cores, TRISC no-op
        deepseek_b1_ops::Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul2_core, Core::is_matmul2_core, true>
            mcast2;
        mcast2(mcast2_args);
    }
    mcast.teardown();

    // ========================================================================
    // Matmul2: matmul2_input[1, 1536] @ matmul2_weights[1536, N]
    // N = 12288 for P150 (96 cores * 4 tiles * 32) or 11264 for non-P150
    // Each core computes 1x4 output tiles (4 1x32 tiles)
    // Output is interleaved Qnope/Qrope with shuffled weights:
    //   - Qnope cores (cols 0-7): 1 head × 128 elements -> matmul3 input
    //   - Qrope cores (cols 8-11): 2 heads × 64 elements -> qrope output
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL2");
        // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
        // On Qnope cores: output stays in matmul2_output_cb for matmul3 input
        // On Qrope cores: output will be copied to qrope_output_cb
        deepseek_b1_ops::Matmul::Op<Matmul2CTArgs, Core::is_matmul2_core, true, false> matmul2;
        matmul2(matmul2_args);
    }

    {
        DeviceZoneScopedN("MATMUL3");
        // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
        // On Qnope cores: output stays in matmul2_output_cb for matmul3 input
        // On Qrope cores: output will be copied to qrope_output_cb
        deepseek_b1_ops::Matmul::Op<Matmul3CTArgs, Core::is_qnope_core, true, false, true> matmul3;
        matmul3(matmul3_args);
    }

    // ========================================================================
    // Qrope output: Copy matmul2 output to qrope_output_cb on Qrope cores
    // Qrope cores have [2, 1, 64] data (2 heads per core, 128 elements total)
    // This output will later be used for RoPE (not implemented in this kernel)
    // ========================================================================
#if defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_qrope_core) {
        DeviceZoneScopedN("QROPE_COPY");
        // Copy matmul2 output to qrope_output_cb
        constexpr uint32_t matmul2_out_cb = get_named_compile_time_arg_val("matmul2_out");  // matmul2_output_cb
        constexpr uint32_t qrope_out_cb = get_named_compile_time_arg_val("qrope_output_cb");
        constexpr uint32_t matmul2_out_w = get_named_compile_time_arg_val("matmul2_out_w_per_core");

        // Wait for matmul2 output to be ready
        cb_wait_front(matmul2_out_cb, matmul2_out_w);

        // Reserve space in qrope output
        cb_reserve_back(qrope_out_cb, matmul2_out_w);

        // Copy data locally within L1 (same core)
        uint32_t src_addr = get_read_ptr(matmul2_out_cb);
        uint32_t dst_addr = get_write_ptr(qrope_out_cb);
        uint32_t tile_size = get_tile_size(matmul2_out_cb);
        uint32_t copy_size = matmul2_out_w * tile_size;

        // Use local NOC write (my_x, my_y) for same-core L1 copy
        uint64_t noc_dst_addr = get_noc_addr(my_x[0], my_y[0], dst_addr);
        noc_async_write(src_addr, noc_dst_addr, copy_size);
        noc_async_write_barrier();

        // Pop matmul2 output, push qrope output
        cb_pop_front(matmul2_out_cb, matmul2_out_w);
        cb_push_back(qrope_out_cb, matmul2_out_w);
    }
#endif

    // ========================================================================
    // TODO: RoPE for Qrope heads
    // Qrope cores have [2, 1, 64] data (2 heads per core)
    // Apply rotary position embedding (not implemented in this kernel)
    // ========================================================================

    // ========================================================================
    // Unicast: QNOPE/QROPE -> SDPA interleaved transfer
    // QNOPE cores (cols 0-7): unicast [1, 512] to SDPA at offset = head_idx * 576
    // QROPE cores (cols 8-11): unicast 2x [1, 64] to SDPA at offsets:
    //   - head_idx * 576 + 512
    //   - (head_idx + 1) * 576 + 512
    // ========================================================================
#if defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_qnope_core) {
        DeviceZoneScopedN("UNICAST_QNOPE");

        // Get compile-time args
        constexpr uint32_t unicast_sdpa_noc_x = get_named_compile_time_arg_val("unicast_sdpa_noc_x");
        constexpr uint32_t unicast_head_stride_bytes = get_named_compile_time_arg_val("unicast_head_stride_bytes");
        constexpr uint32_t unicast_receiver_semaphore_id =
            get_named_compile_time_arg_val("unicast_receiver_semaphore_id");
        constexpr uint32_t unicast_qnope_src_cb = get_named_compile_time_arg_val("unicast_qnope_src_cb");
        constexpr uint32_t unicast_qnope_src_num_pages = get_named_compile_time_arg_val("unicast_qnope_src_num_pages");

        // NOC y-coordinates for each row's SDPA core (lookup table)
        constexpr uint32_t sdpa_noc_y_table[] = {
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row0"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row1"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row2"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row3"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row4"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row5"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row6"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row7"),
        };

        // Compute head index based on logical x coordinate
        uint32_t head_idx = my_logical_x_;  // 0-7 for QNOPE cores

        // Wait for matmul3 output to be ready
        cb_wait_front(unicast_qnope_src_cb, unicast_qnope_src_num_pages);

        // Get source and destination addresses
        uint32_t src_addr = get_read_ptr(unicast_qnope_src_cb);
        uint32_t tile_size = get_tile_size(unicast_qnope_src_cb);
        uint32_t data_size = unicast_qnope_src_num_pages * tile_size;

        // Get SDPA destination NOC coordinates based on row
        uint32_t sdpa_noc_y = sdpa_noc_y_table[my_logical_y_];
        constexpr uint32_t unicast_receive_cb = get_named_compile_time_arg_val("unicast_receive_cb");
        uint32_t sdpa_dst_addr = get_write_ptr(unicast_receive_cb);

        // Compute destination offset: QNOPE at offset = head_idx * stride
        uint32_t dst_offset = head_idx * unicast_head_stride_bytes;

        // NOC write to SDPA core
        uint64_t dst_noc_addr = get_noc_addr(unicast_sdpa_noc_x, sdpa_noc_y, sdpa_dst_addr + dst_offset);
        noc_async_write(src_addr, dst_noc_addr, data_size);
        noc_async_write_barrier();

        // Increment receiver semaphore (1 head sent)
        uint32_t receiver_semaphore_addr = get_semaphore(unicast_receiver_semaphore_id);
        uint64_t dst_semaphore_noc_addr = get_noc_addr(unicast_sdpa_noc_x, sdpa_noc_y, receiver_semaphore_addr);
        noc_semaphore_inc(dst_semaphore_noc_addr, 1);
        noc_async_posted_writes_flushed();

        // Pop source CB
        cb_pop_front(unicast_qnope_src_cb, unicast_qnope_src_num_pages);
    }

    if constexpr (Core::is_qrope_core) {
        DeviceZoneScopedN("UNICAST_QROPE");

        // Get compile-time args
        constexpr uint32_t unicast_sdpa_noc_x = get_named_compile_time_arg_val("unicast_sdpa_noc_x");
        constexpr uint32_t unicast_head_stride_bytes = get_named_compile_time_arg_val("unicast_head_stride_bytes");
        constexpr uint32_t unicast_qnope_data_size_bytes =
            get_named_compile_time_arg_val("unicast_qnope_data_size_bytes");
        constexpr uint32_t unicast_qrope_data_size_bytes =
            get_named_compile_time_arg_val("unicast_qrope_data_size_bytes");
        constexpr uint32_t unicast_receiver_semaphore_id =
            get_named_compile_time_arg_val("unicast_receiver_semaphore_id");
        constexpr uint32_t unicast_qrope_src_cb = get_named_compile_time_arg_val("unicast_qrope_src_cb");
        constexpr uint32_t unicast_qrope_src_num_pages = get_named_compile_time_arg_val("unicast_qrope_src_num_pages");
        constexpr uint32_t unicast_qnope_grid_cols = get_named_compile_time_arg_val("unicast_qnope_grid_cols");

        // NOC y-coordinates for each row's SDPA core (lookup table)
        constexpr uint32_t sdpa_noc_y_table[] = {
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row0"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row1"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row2"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row3"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row4"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row5"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row6"),
            get_named_compile_time_arg_val("unicast_sdpa_noc_y_row7"),
        };

        // Compute head indices: QROPE core at col X has heads 2*(X-8) and 2*(X-8)+1
        uint32_t qrope_col_idx = my_logical_x_ - unicast_qnope_grid_cols;  // 0-3
        uint32_t head_idx0 = 2 * qrope_col_idx;                            // 0, 2, 4, 6
        uint32_t head_idx1 = 2 * qrope_col_idx + 1;                        // 1, 3, 5, 7

        // Wait for qrope output to be ready
        cb_wait_front(unicast_qrope_src_cb, unicast_qrope_src_num_pages);

        // Get source and destination addresses
        uint32_t src_addr = get_read_ptr(unicast_qrope_src_cb);

        // Get SDPA destination NOC coordinates based on row
        uint32_t sdpa_noc_y = sdpa_noc_y_table[my_logical_y_];
        constexpr uint32_t unicast_receive_cb = get_named_compile_time_arg_val("unicast_receive_cb");
        uint32_t sdpa_dst_addr = get_write_ptr(unicast_receive_cb);

        // QROPE unicast: each QROPE core sends 2 heads interleaved
        // Head 0: offset = head_idx0 * stride + qnope_data_size
        // Head 1: offset = head_idx1 * stride + qnope_data_size
        uint32_t dst_offset0 = head_idx0 * unicast_head_stride_bytes + unicast_qnope_data_size_bytes;
        uint32_t dst_offset1 = head_idx1 * unicast_head_stride_bytes + unicast_qnope_data_size_bytes;

        // Single head size (64 elements * 2 bytes = 128 bytes)
        uint32_t single_head_size = unicast_qrope_data_size_bytes;

        // NOC write head 0
        uint64_t dst_noc_addr0 = get_noc_addr(unicast_sdpa_noc_x, sdpa_noc_y, sdpa_dst_addr + dst_offset0);
        noc_async_write(src_addr, dst_noc_addr0, single_head_size);

        // NOC write head 1 (offset by single_head_size in source)
        uint64_t dst_noc_addr1 = get_noc_addr(unicast_sdpa_noc_x, sdpa_noc_y, sdpa_dst_addr + dst_offset1);
        noc_async_write(src_addr + single_head_size, dst_noc_addr1, single_head_size);

        noc_async_write_barrier();

        // Increment receiver semaphore (2 heads sent)
        uint32_t receiver_semaphore_addr = get_semaphore(unicast_receiver_semaphore_id);
        uint64_t dst_semaphore_noc_addr = get_noc_addr(unicast_sdpa_noc_x, sdpa_noc_y, receiver_semaphore_addr);
        noc_semaphore_inc(dst_semaphore_noc_addr, 2);
        noc_async_posted_writes_flushed();

        // Pop source CB
        cb_pop_front(unicast_qrope_src_cb, unicast_qrope_src_num_pages);
    }
#endif

#if defined(COMPILE_FOR_NCRISC)
    // ========================================================================
    // SDPA Receiver: Wait for all unicasts to complete, then copy to output CB
    // ========================================================================
    if constexpr (Core::is_sdpa_core) {
        DeviceZoneScopedN("UNICAST_SDPA_RECEIVER");

        constexpr uint32_t unicast_num_senders = get_named_compile_time_arg_val("unicast_num_senders");
        constexpr uint32_t unicast_receiver_semaphore_id =
            get_named_compile_time_arg_val("unicast_receiver_semaphore_id");
        constexpr uint32_t unicast_receive_cb = get_named_compile_time_arg_val("unicast_receive_cb");
        constexpr uint32_t unicast_output_cb = get_named_compile_time_arg_val("unicast_output_cb");
        constexpr uint32_t unicast_dst_num_pages = get_named_compile_time_arg_val("unicast_dst_num_pages");

        // Reserve space in receive CB (senders write directly here)
        cb_reserve_back(unicast_receive_cb, unicast_dst_num_pages);

        // Wait for all senders (8 QNOPE + 8 QROPE = 16)
        uint32_t receiver_semaphore_addr = get_semaphore(unicast_receiver_semaphore_id);
        volatile tt_l1_ptr uint32_t* receiver_semaphore_ptr = (volatile tt_l1_ptr uint32_t*)receiver_semaphore_addr;
        noc_semaphore_wait(receiver_semaphore_ptr, unicast_num_senders);
        noc_semaphore_set(receiver_semaphore_ptr, 0);

        // Push to receive CB (data arrived)
        cb_push_back(unicast_receive_cb, unicast_dst_num_pages);

        // Now copy from receive CB to output CB (which is linked to tensor)
        cb_wait_front(unicast_receive_cb, unicast_dst_num_pages);
        cb_reserve_back(unicast_output_cb, unicast_dst_num_pages);

        uint32_t src_addr = get_read_ptr(unicast_receive_cb);
        uint32_t dst_addr = get_write_ptr(unicast_output_cb);
        uint32_t tile_size = get_tile_size(unicast_receive_cb);
        uint32_t copy_size = unicast_dst_num_pages * tile_size;

        // Local copy within same core's L1
        uint64_t noc_dst_addr = get_noc_addr(my_x[0], my_y[0], dst_addr);
        noc_async_write(src_addr, noc_dst_addr, copy_size);
        noc_async_write_barrier();

        cb_pop_front(unicast_receive_cb, unicast_dst_num_pages);
        cb_push_back(unicast_output_cb, unicast_dst_num_pages);
    }
#endif
}
KERNEL_END
