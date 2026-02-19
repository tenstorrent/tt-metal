// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Extract sub-tensor from an overlapped (fused) sharded tensor.
// NCRISC copies a contiguous byte range from the input shard to the output shard.
// BRISC and TRISC are no-ops.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t src_cb = get_named_compile_time_arg_val("src_cb");
    constexpr uint32_t dst_cb = get_named_compile_time_arg_val("dst_cb");
    constexpr uint32_t src_num_tiles = get_named_compile_time_arg_val("src_num_tiles");
    constexpr uint32_t dst_num_tiles = get_named_compile_time_arg_val("dst_num_tiles");
    constexpr uint32_t byte_offset = get_named_compile_time_arg_val("byte_offset");
    constexpr uint32_t copy_size_bytes = get_named_compile_time_arg_val("copy_size_bytes");

    // Make sharded tensor data available via CB
    unified_kernels::setup_sharded_buffer(src_cb, src_num_tiles);
    unified_kernels::setup_sharded_buffer(dst_cb, dst_num_tiles);

    // Get L1 addresses
    uint32_t src_addr = get_read_ptr(src_cb) + byte_offset;
    uint32_t dst_addr = get_write_ptr(dst_cb);

    // Copy from input shard (at offset) to output shard (local L1 -> local L1)
    noc_async_write(src_addr, get_noc_addr(dst_addr), copy_size_bytes);
    noc_async_write_barrier();
#endif
}
