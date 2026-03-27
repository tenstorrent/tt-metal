// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for minimal_binary_op.
// Optionally reads input B (NoC-balanced mode) and always writes output C.
// When DO_PROCESS_INPUT1 is defined, B-read is interleaved with C-write per block.
//
// Compile-time args:
//   [0]   block_size             — tiles per write batch
//   [1..] TensorAccessorArgs     — for output C
//   If DO_PROCESS_INPUT1 defined: next TensorAccessorArgs for input B
//
// Per-core runtime args:
//   [0] dst_addr_c
//   [1] start_tile
//   [2] Wt
//   If DO_PROCESS_INPUT1 defined: [3] src_addr_b
//
// Defines:
//   DO_PROCESS_INPUT1   — if defined, writer reads input B and pushes to CB_1
//   USE_FLUSHED_WRITES  — if defined, use async_writes_flushed() instead of barrier()

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void print_tile_cb(const experimental::CircularBuffer& cb, uint32_t offset, uint32_t len) {
    uint32_t addr = cb.get_read_ptr();
    volatile uint16_t* ptr = reinterpret_cast<volatile uint16_t*>(addr);

    for (uint32_t i = 0; i < len; i++) {
        DPRINT << " " << BF16(ptr[offset + i]);
    }
    DPRINT << ENDL();
}

void kernel_main() {
    constexpr uint32_t block_size = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1, 0>();

#ifdef DO_PROCESS_INPUT1
    constexpr auto src_b_args =
        TensorAccessorArgs<dst_args.next_compile_time_args_offset(), dst_args.next_common_runtime_args_offset()>();
#endif

    const uint32_t dst_addr_c = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
#ifdef DO_PROCESS_INPUT1
    const uint32_t src_addr_b = get_arg_val<uint32_t>(3);
#endif

    constexpr auto cb_out_id = tt::CBIndex::c_2;
    experimental::CircularBuffer cb_out(cb_out_id);
    const uint32_t tile_size_c = get_tile_size(cb_out_id);
    const auto dst = TensorAccessor(dst_args, dst_addr_c, tile_size_c);

#ifdef DO_PROCESS_INPUT1
    constexpr auto cb_b_id = tt::CBIndex::c_1;
    experimental::CircularBuffer cb_src_b(cb_b_id);
    const uint32_t tile_size_b = get_tile_size(cb_b_id);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, tile_size_b);
#endif

    experimental::Noc noc;

    for (uint32_t tiles_done = 0; tiles_done < Wt; tiles_done += block_size) {
        const uint32_t cur_block = (Wt - tiles_done < block_size) ? (Wt - tiles_done) : block_size;

#ifdef DO_PROCESS_INPUT1
        // Read input B block (interleaved with output write below)
        cb_src_b.reserve_back(block_size);
        uint32_t dst_offset = 0;
        for (uint32_t j = 0; j < cur_block; ++j) {
            noc.async_read(
                src_b, cb_src_b, tile_size_b, {.page_id = start_tile + tiles_done + j}, {.offset_bytes = dst_offset});
            dst_offset += tile_size_b;
        }
        noc.async_read_barrier();
        cb_src_b.push_back(block_size);
#endif

        // Write output C block
        cb_out.wait_front(block_size);

        uint32_t src_offset = 0;
        for (uint32_t j = 0; j < cur_block; ++j) {
            noc.async_write(
                cb_out, dst, tile_size_c, {.offset_bytes = src_offset}, {.page_id = start_tile + tiles_done + j});
            src_offset += tile_size_c;
        }
#ifdef USE_FLUSHED_WRITES
        noc.async_writes_flushed();
#else
        noc.async_write_barrier();
#endif
        cb_out.pop_front(block_size);
    }
}
