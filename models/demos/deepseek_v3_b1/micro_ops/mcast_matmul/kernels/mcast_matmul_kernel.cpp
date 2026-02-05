// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Multi-core matmul with mcast input distribution
//
// Architecture:
//   - Sender core: Mcast input activations to all matmul cores
//   - Matmul cores: Receive input, compute matmul with local weight shard
//
// NCRISC: Mcast receiver + weight buffer setup
// BRISC: Mcast sender
// TRISC: Matmul compute

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/mcast_matmul.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    // is_mcast_grid_core: true for cores in mcast bounding box (includes matmul cores + phantom cores)
    // Phantom cores are in the bounding box but not actual matmul cores - they still need to ack the mcast
    static constexpr bool is_mcast_grid_core = get_named_compile_time_arg_val("is_mcast_grid_core") == 1;
};

void kernel_main() {
    using McastMatmul = deepseek_b1_ops::McastMatmul;

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
    McastMatmul::ReaderArgs args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_in1_num_pages"),
    };

    // CTArgs type alias (NCRISC uses ReaderCTArgs - no compile-time params)
    using McastMatmulCTArgs = McastMatmul::ReaderCTArgs;

// ============================================================================
// BRISC (Writer) - Mcast sender
// Named compile-time args: mcast sender params
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // CB indices from named compile-time args
    constexpr uint32_t src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");

    // Writer args for BRISC (mcast sender)
    McastMatmul::WriterArgs args{
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
    // loopback: sender receives its own mcast (set independently from is_matmul_core)
    using McastMatmulCTArgs = McastMatmul::WriterCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        get_named_compile_time_arg_val("mcast_loopback") == 1>;

// ============================================================================
// TRISC (Compute) - Matmul
// Named compile-time args: matmul params
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Compute args for TRISC
    McastMatmul::ComputeArgs args{
        get_named_compile_time_arg_val("mcast_dst_cb"),  // in0 = mcast destination CB
        get_named_compile_time_arg_val("matmul_in1"),    // in1 = weights CB
        get_named_compile_time_arg_val("matmul_out"),    // out = output CB
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };

    // CTArgs type alias for TRISC (matmul parameters)
    using McastMatmulCTArgs = McastMatmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;
#endif

    // ========================================================================
    // Execute mcast + matmul operation
    // ========================================================================
    // pop_input = true (mcast input is consumed after matmul)
    // is_mcast_grid_core is passed separately to handle phantom cores (in bounding box but not matmul)
    McastMatmul::Op<McastMatmulCTArgs, Core::is_sender_core, Core::is_matmul_core, Core::is_mcast_grid_core, true>
        mcast_matmul;

    {
        DeviceZoneScopedN("MCAST_MATMUL_INIT");
        mcast_matmul.init(args);
    }

    {
        DeviceZoneScopedN("MCAST_MATMUL");
        mcast_matmul(args);
    }

    {
        DeviceZoneScopedN("MCAST_MATMUL_TEARDOWN");
        mcast_matmul.teardown();
    }
}
