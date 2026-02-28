// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified CCL Broadcast kernel
// - NCRISC: Broadcast reader (reads local data into CB)
// - BRISC: Broadcast writer (sends to fabric / waits for data)
// - TRISC: No-op (dataflow only)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/broadcast.hpp"

void kernel_main() {
    using Broadcast = deepseek_b1_ops::Broadcast;

#if defined(COMPILE_FOR_NCRISC)
    // Writer CTArgs
    using BcastCTArgs = Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_tensor0_page_size"),
        get_named_compile_time_arg_val("bcast_num_connections"),
        get_named_compile_time_arg_val("bcast_is_root"),
        get_named_compile_time_arg_val("bcast_chunk_size_bytes"),
        get_named_compile_time_arg_val("bcast_last_chunk_size_bytes"),
        get_named_compile_time_arg_val("bcast_num_chunks")>;

    // Writer runtime args
    Broadcast::WriterArgs bcast_args{
        get_common_arg_val<uint32_t>(0),  // tensor_address0
        get_common_arg_val<uint32_t>(1),  // sem_bank_addr
        get_common_arg_val<uint32_t>(2),  // my_noc_x
        get_common_arg_val<uint32_t>(3),  // my_noc_y
    };

#elif defined(COMPILE_FOR_BRISC)
    // Reader CTArgs
    using BcastCTArgs = Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender")>;

    // Runtime args:
    Broadcast::ReaderArgs bcast_args{};

#elif defined(COMPILE_FOR_TRISC)
    // TRISC: Compute args unused for broadcast
    Broadcast::ComputeArgs bcast_args{};
    Broadcast::ComputeCTArgs BcastCTArgs = {};
#endif

    // Execute ccl broadcast op
    constexpr uint32_t num_iterations = get_named_compile_time_arg_val("bcast_num_iterations");

    auto body = [&]() {
        Broadcast::Op<BcastCTArgs, true> bcast;
        bcast(bcast_args);
    };

    for (uint32_t i = 0; i < num_iterations; i++) {
        body();
    }
}
