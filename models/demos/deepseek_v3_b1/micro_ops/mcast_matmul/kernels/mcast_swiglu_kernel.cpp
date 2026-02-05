// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Multi-core fused SwiGLU with mcast input distribution
//
// Architecture:
//   - Sender core: Mcast input activations to all matmul cores
//   - Matmul cores: Receive input, compute fused SwiGLU:
//     1. gate = SiLU(input @ W_gate)
//     2. up = input @ W_up
//     3. output = gate * up
//
// Fusion benefit: All three operations execute on the same core using local CBs,
// avoiding any cross-core data movement between operations.
//
// NCRISC: Mcast receiver + weight buffer setup (both W_gate and W_up)
// BRISC: Mcast sender
// TRISC: Fused SwiGLU compute (MatmulSiLU + Matmul + EltwiseMul)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/mcast_swiglu.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
};

KERNEL_ENTRY {
    using McastSwiGLU = deepseek_b1_ops::McastSwiGLU;

// ============================================================================
// NCRISC (Reader) - Mcast receiver + buffer setup
// Named compile-time args: mcast receiver params, weight buffer params (gate & up)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded input buffer on sender core so BRISC can read it
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);
    }

    // Reader args for NCRISC
    McastSwiGLU::ReaderArgs args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
        get_named_compile_time_arg_val("gate_weights_cb"),
        get_named_compile_time_arg_val("gate_weights_num_pages"),
        get_named_compile_time_arg_val("up_weights_cb"),
        get_named_compile_time_arg_val("up_weights_num_pages"),
    };

    // CTArgs type alias (NCRISC uses ReaderCTArgs - no compile-time params)
    using McastSwiGLUCTArgs = McastSwiGLU::ReaderCTArgs;

// ============================================================================
// BRISC (Writer) - Mcast sender
// Named compile-time args: mcast sender params
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // CB indices from named compile-time args
    constexpr uint32_t src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");

    // Writer args for BRISC (mcast sender)
    McastSwiGLU::WriterArgs args{
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
    using McastSwiGLUCTArgs = McastSwiGLU::WriterCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_sender_core && Core::is_matmul_core>;  // loopback = sender is also a matmul core

// ============================================================================
// TRISC (Compute) - Fused SwiGLU
// Named compile-time args: swiglu params
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Compute args for TRISC
    McastSwiGLU::ComputeArgs args{
        get_named_compile_time_arg_val("mcast_dst_cb"),          // in0 = mcast destination CB
        get_named_compile_time_arg_val("gate_weights_cb"),       // W_gate CB
        get_named_compile_time_arg_val("up_weights_cb"),         // W_up CB
        get_named_compile_time_arg_val("gate_intermediate_cb"),  // gate output intermediate
        get_named_compile_time_arg_val("up_intermediate_cb"),    // up output intermediate
        get_named_compile_time_arg_val("out_cb"),                // final output CB
        get_named_compile_time_arg_val("k_num_tiles"),
    };

    // CTArgs type alias for TRISC (swiglu parameters)
    using McastSwiGLUCTArgs = McastSwiGLU::ComputeCTArgs<get_named_compile_time_arg_val("out_w_per_core")>;
#endif

    // ========================================================================
    // Execute mcast + fused SwiGLU operation
    // ========================================================================
    // pop_input = true (mcast input is consumed after SwiGLU)
    McastSwiGLU::Op<McastSwiGLUCTArgs, Core::is_sender_core, Core::is_matmul_core, true> mcast_swiglu;

    {
        DeviceZoneScopedN("MCAST_SWIGLU_INIT");
        mcast_swiglu.init(args);
    }

    {
        DeviceZoneScopedN("MCAST_SWIGLU");
        mcast_swiglu(args);
    }

    {
        DeviceZoneScopedN("MCAST_SWIGLU_TEARDOWN");
        mcast_swiglu.teardown();
    }
}
KERNEL_END
