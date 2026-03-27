// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for minimal_binary_op.
//
// Compile-time args:
//   [0]   block_size           — tiles per read batch
//   [1..] TensorAccessorArgs   — for input A
//   If DO_PROCESS_INPUT1 defined: next TensorAccessorArgs for input B
//
// Per-core runtime args:
//   [0] src_addr_a
//   [1] start_tile
//   [2] Wt
//   If DO_PROCESS_INPUT1 defined: [3] src_addr_b

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void print_tile_cb(const experimental::CircularBuffer& cb, uint32_t offset, uint32_t len) {
    uint32_t addr = cb.get_write_ptr();
    volatile uint16_t* ptr = reinterpret_cast<volatile uint16_t*>(addr);

    for (uint32_t i = 0; i < len; i++) {
        DPRINT << " " << BF16(ptr[offset + i]);
    }
    DPRINT << ENDL();
}

void kernel_main() {
    constexpr uint32_t block_size = get_compile_time_arg_val(0);

    constexpr auto src_a_args = TensorAccessorArgs<1, 0>();

#ifdef DO_PROCESS_INPUT1
    constexpr auto src_b_args =
        TensorAccessorArgs<src_a_args.next_compile_time_args_offset(), src_a_args.next_common_runtime_args_offset()>();
#endif

    const uint32_t src_addr_a = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
#ifdef DO_PROCESS_INPUT1
    const uint32_t src_addr_b = get_arg_val<uint32_t>(3);
#endif

    constexpr auto cb_a = tt::CBIndex::c_0;
    experimental::CircularBuffer cb_src_a(cb_a);
    const uint32_t tile_size_a = get_tile_size(cb_a);
    const auto src_a = TensorAccessor(src_a_args, src_addr_a, tile_size_a);

#ifdef DO_PROCESS_INPUT1
    constexpr auto cb_b = tt::CBIndex::c_1;
    experimental::CircularBuffer cb_src_b(cb_b);
    const uint32_t tile_size_b = get_tile_size(cb_b);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, tile_size_b);
#endif

    experimental::Noc noc;

    uint32_t tiles_read = 0;

    for (uint32_t tiles_read = 0; tiles_read < Wt; tiles_read += block_size) {
        const uint32_t cur_block = (Wt - tiles_read < block_size) ? (Wt - tiles_read) : block_size;

        // Read input A — always reserve full block_size but only issue cur_block reads
        cb_src_a.reserve_back(block_size);
        uint32_t dst_offset = 0;
        const uint32_t src_a_bytes = 1024 * 2;
        for (uint32_t j = 0; j < cur_block; ++j) {
            noc.async_read(
                src_a, cb_src_a, tile_size_a, {.page_id = start_tile + tiles_read + j}, {.offset_bytes = dst_offset});
            dst_offset += tile_size_a;
        }
        noc.async_read_barrier();

        cb_src_a.push_back(block_size);

#ifdef DO_PROCESS_INPUT1
        // Read input B
        cb_src_b.reserve_back(block_size);
        dst_offset = 0;
        const uint32_t src_b_bytes = 1024 * 2;
        for (uint32_t j = 0; j < cur_block; ++j) {
            noc.async_read(
                src_b, cb_src_b, tile_size_b, {.page_id = start_tile + tiles_read + j}, {.offset_bytes = dst_offset});
            dst_offset += tile_size_b;
        }
        noc.async_read_barrier();

        cb_src_b.push_back(block_size);
#endif
    }
}
