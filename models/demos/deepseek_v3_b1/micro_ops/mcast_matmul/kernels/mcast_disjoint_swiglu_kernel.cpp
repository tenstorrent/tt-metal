// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Multi-core fused SwiGLU with DISJOINT gate/up grids
//
// Architecture:
//   - Sender core: Mcast input activations to both gate and up grids
//   - Gate cores (e.g., 72): Receive input, compute SiLU(input @ W_gate), send to up cores
//   - Up cores (e.g., 36): Receive input, receive gate results, compute input @ W_up, multiply
//
// Data flow:
//   1. Mcast: sender → all matmul cores (gate grid ∪ up grid)
//   2. Gate compute: input @ W_gate + SiLU
//   3. Gate→Up transfer: each 2 gate cores send to 1 up core
//   4. Up compute: input @ W_up
//   5. Multiply: gate * up → output
//
// NCRISC: Mcast receiver + weight buffer setup + gate recv setup (for up cores)
// BRISC: Mcast sender + gate→up transfer (for gate cores)
// TRISC: Gate compute (gate cores) / Up compute + multiply (up cores)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/mcast_disjoint_swiglu.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_gate_core = get_named_compile_time_arg_val("is_gate_core") == 1;
    static constexpr bool is_up_core = get_named_compile_time_arg_val("is_up_core") == 1;
};

KERNEL_ENTRY {
    using McastDisjointSwiGLU = deepseek_b1_ops::McastDisjointSwiGLU;

// ============================================================================
// NCRISC (Reader) - Mcast receiver + buffer setup
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded input buffer on sender core so BRISC can read it
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);
    }

    // Reader args for NCRISC
    McastDisjointSwiGLU::ReaderArgs args{
        // Mcast receiver
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
        // Gate weights setup
        get_named_compile_time_arg_val("gate_weights_cb"),
        get_named_compile_time_arg_val("gate_weights_num_pages"),
        // Up weights setup
        get_named_compile_time_arg_val("up_weights_cb"),
        get_named_compile_time_arg_val("up_weights_num_pages"),
        // Gate recv setup
        get_named_compile_time_arg_val("gate_recv_cb"),
        get_named_compile_time_arg_val("gate_recv_num_pages"),
        get_named_compile_time_arg_val("gate_recv_semaphore"),
    };

    // CTArgs type alias for NCRISC
    using DisjointSwiGLUCTArgs =
        McastDisjointSwiGLU::ReaderCTArgs<get_named_compile_time_arg_val("gate_cores_per_up_core")>;

// ============================================================================
// BRISC (Writer) - Mcast sender + gate→up transfer
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // CB indices from named compile-time args
    constexpr uint32_t src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    constexpr uint32_t gate_output_cb = get_named_compile_time_arg_val("gate_output_cb");

    // Additional CB for gate→up transfer
    constexpr uint32_t gate_recv_cb = get_named_compile_time_arg_val("gate_recv_cb");

    // Writer args for BRISC
    McastDisjointSwiGLU::WriterArgs args{
        // Mcast sender params
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
        // Gate→Up transfer params
        gate_output_cb,
        get_named_compile_time_arg_val("gate_output_num_pages"),
        gate_recv_cb,
        get_named_compile_time_arg_val("gate_recv_semaphore"),
        // Grid params for computing target up core (all NOC coordinates)
        get_named_compile_time_arg_val("gate_grid_start_noc_x"),
        get_named_compile_time_arg_val("gate_grid_start_noc_y"),
        get_named_compile_time_arg_val("up_grid_start_noc_x"),
        get_named_compile_time_arg_val("up_grid_start_noc_y"),
        get_named_compile_time_arg_val("out_tile_size_bytes"),
    };

    // CTArgs type alias for BRISC
    using DisjointSwiGLUCTArgs = McastDisjointSwiGLU::WriterCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_sender_core && (Core::is_gate_core || Core::is_up_core),  // loopback
        get_named_compile_time_arg_val("out_w_per_gate_core"),
        get_named_compile_time_arg_val("gate_cores_per_up_core")>;

// ============================================================================
// TRISC (Compute) - Gate compute / Up compute + multiply
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Compute args for TRISC
    McastDisjointSwiGLU::ComputeArgs args{
        // Common
        get_named_compile_time_arg_val("mcast_dst_cb"),
        // Gate compute args
        get_named_compile_time_arg_val("gate_weights_cb"),
        get_named_compile_time_arg_val("gate_output_cb"),
        // Up compute args
        get_named_compile_time_arg_val("up_weights_cb"),
        get_named_compile_time_arg_val("gate_recv_cb"),
        get_named_compile_time_arg_val("up_output_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("k_num_tiles"),
    };

    // CTArgs type alias for TRISC
    using DisjointSwiGLUCTArgs = McastDisjointSwiGLU::ComputeCTArgs<
        get_named_compile_time_arg_val("out_w_per_gate_core"),
        get_named_compile_time_arg_val("out_w_per_up_core")>;
#endif

    // ========================================================================
    // Execute mcast + disjoint SwiGLU operation
    // ========================================================================
    // pop_input = true (mcast input is consumed after compute)
    McastDisjointSwiGLU::Op<DisjointSwiGLUCTArgs, Core::is_sender_core, Core::is_gate_core, Core::is_up_core, true>
        mcast_disjoint_swiglu;

    {
        DeviceZoneScopedN("MCAST_DISJOINT_SWIGLU_INIT");
        mcast_disjoint_swiglu.init(args);
    }

    {
        DeviceZoneScopedN("MCAST_DISJOINT_SWIGLU");
        mcast_disjoint_swiglu(args);
    }

    {
        DeviceZoneScopedN("MCAST_DISJOINT_SWIGLU_TEARDOWN");
        mcast_disjoint_swiglu.teardown();
    }
}
KERNEL_END
