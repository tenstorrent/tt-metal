// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Dest Accumulation unified kernel
// Simple element-wise addition: output = in_cb[0] + in_cb[1]
// Both input tiles from single CB
//
// NCRISC: Signals sharded CB is ready
// BRISC: No-op
// TRISC: Performs add via DestAccum::Op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/dest_accum.hpp"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using DestAccumCTArgs = deepseek_b1_ops::DestAccum::ReaderCTArgs;

    constexpr uint32_t in_cb = get_named_compile_time_arg_val("dest_accum_in_cb");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("dest_accum_num_tiles");

    // Setup sharded buffer
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(in_cb, num_tiles);
    }

    deepseek_b1_ops::DestAccum::ReaderArgs dest_accum_args{};

#elif defined(COMPILE_FOR_BRISC)
    using DestAccumCTArgs = deepseek_b1_ops::DestAccum::WriterCTArgs;

    deepseek_b1_ops::DestAccum::WriterArgs dest_accum_args{};

#elif defined(COMPILE_FOR_TRISC)
    using DestAccumCTArgs = deepseek_b1_ops::DestAccum::ComputeCTArgs;

    constexpr uint32_t in_cb = get_named_compile_time_arg_val("dest_accum_in_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("dest_accum_out_cb");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("dest_accum_num_tiles");

    deepseek_b1_ops::DestAccum::ComputeArgs dest_accum_args{
        .in_cb = in_cb,
        .out_cb = out_cb,
        .num_tiles = num_tiles,
    };
#endif

    deepseek_b1_ops::DestAccum::Op<DestAccumCTArgs, Core::is_active_core> dest_accum;
    dest_accum(dest_accum_args);
}
