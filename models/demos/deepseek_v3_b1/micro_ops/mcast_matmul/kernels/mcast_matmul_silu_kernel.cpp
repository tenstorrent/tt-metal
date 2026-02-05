// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Multi-core fused matmul+SiLU with mcast input distribution
//
// Architecture:
//   - Sender core: Mcast input activations to all matmul cores
//   - Matmul cores: Receive input, compute fused matmul+SiLU with local weight shard
//
// Fusion benefit: SiLU is applied directly to DST registers after matmul,
// avoiding the L1 round-trip that would occur with separate ops.
//
// NCRISC: Mcast receiver + weight buffer setup
// BRISC: Mcast sender
// TRISC: Fused matmul+SiLU compute

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/mcast_matmul_silu.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
};

KERNEL_ENTRY {
    using McastMatmulSiLU = deepseek_b1_ops::McastMatmulSiLU;

// ============================================================================
// NCRISC (Reader) - Mcast receiver + buffer setup
// Named compile-time args: mcast receiver params, weight buffer params
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded input buffer on sender core so BRISC can read it
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);
    }

    // Reader args for NCRISC
    McastMatmulSiLU::ReaderArgs args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_in1_num_pages"),
    };

    // CTArgs type alias (NCRISC uses ReaderCTArgs - no compile-time params)
    using McastMatmulSiLUCTArgs = McastMatmulSiLU::ReaderCTArgs;

// ============================================================================
// BRISC (Writer) - Mcast sender
// Named compile-time args: mcast sender params
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // CB indices from named compile-time args
    constexpr uint32_t src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");

    // Writer args for BRISC (mcast sender)
    McastMatmulSiLU::WriterArgs args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_data_size_bytes"),
        src_cb,
        get_named_compile_time_arg_val("mcast_src_num_pages"),
        get_read_ptr(src_cb),
        get_write_ptr(dst_cb),
    };

    // CTArgs type alias for BRISC (mcast sender parameters)
    using McastMatmulSiLUCTArgs = McastMatmulSiLU::WriterCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_sender_core && Core::is_matmul_core>;  // loopback = sender is also a matmul core

// ============================================================================
// TRISC (Compute) - Fused Matmul+SiLU
// Named compile-time args: matmul params
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Compute args for TRISC
    McastMatmulSiLU::ComputeArgs args{
        get_named_compile_time_arg_val("mcast_dst_cb"),  // in0 = mcast destination CB
        get_named_compile_time_arg_val("matmul_in1"),    // in1 = weights CB
        get_named_compile_time_arg_val("matmul_out"),    // out = output CB
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };

    // CTArgs type alias for TRISC (matmul parameters)
    using McastMatmulSiLUCTArgs =
        McastMatmulSiLU::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;
#endif

    // ========================================================================
    // Execute mcast + fused matmul+SiLU operation
    // ========================================================================
    // pop_input = true (mcast input is consumed after matmul+SiLU)
    McastMatmulSiLU::Op<McastMatmulSiLUCTArgs, Core::is_sender_core, Core::is_matmul_core, true> mcast_matmul_silu;

    {
        DeviceZoneScopedN("MCAST_MATMUL_SILU_INIT");
        mcast_matmul_silu.init(args);
    }

    {
        DeviceZoneScopedN("MCAST_MATMUL_SILU");
        mcast_matmul_silu(args);
    }

    {
        DeviceZoneScopedN("MCAST_MATMUL_SILU_TEARDOWN");
        mcast_matmul_silu.teardown();
    }
}
KERNEL_END
