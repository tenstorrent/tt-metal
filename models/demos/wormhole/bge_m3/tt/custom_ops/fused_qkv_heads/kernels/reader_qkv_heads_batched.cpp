// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 Track A — batched-barrier reader for nlp_create_qkv_heads
// (interleaved input, single Q-input tensor, no K/V tensor).
//
// Replaces ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/
// device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp.
//
// Difference vs stock: stock performs a one-tile CB reserve/read/barrier/push
// sequence for each tile (192 NoC barriers per block-of-96-tiles). This kernel
// batches each of the Q/K/V tile groups into a single CB reservation and a
// single barrier, dropping per-block barrier count from 96 to 3.
//
// Compile-time args (matching stock layout for accessor offsets):
//   0: q_num_tiles      = num_q_heads * head_dim_tiles                  (= 32)
//   1: kv_num_tiles     = num_kv_heads * head_dim_tiles                 (= 32)
//   2+: TensorAccessorArgs for the QKV-fused input tensor (q_args)
//
// Runtime args (matching stock layout):
//   0: in0_tensor_addr           (QKV-fused input base address)
//   1: in1_tensor_addr           (unused; kept for arg-index parity)
//   2: num_blocks                (tile-rows of S to process on this core)
//   3: in0_tensor_tile_id        (starting page-id in the fused tensor)
//   4: in1_tensor_tile_id        (unused; kept for parity)
//
// Defines:
//   (none — TRANSPOSE_K_HEADS and READ_FROM_INPUT_TENSOR_KV are NOT
//    supported by this kernel; BGE-M3 doesn't use them.)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // ---- runtime args ----
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    // arg 1 unused (in1 not used)
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(3);
    // arg 4 unused

    // ---- compile-time args ----
    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t kv_num_tiles = get_compile_time_arg_val(1);
    constexpr auto in0_args = TensorAccessorArgs<2>();

    // ---- CBs (must match Python ProgramDescriptor) ----
    // Single CB 1 is shared by Q, K, V outputs to the writer (stock layout).
    constexpr uint32_t cb_id = 1;

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);
    const uint32_t tile_size_bytes = get_tile_size(cb_id);

    // Device 2.0 data-movement API (see device_api_migration_guide.md).
    Noc noc;
    CircularBuffer cb(cb_id);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // ---- Q chunk: q_num_tiles in one reserve / one barrier / one push ----
        cb.reserve_back(q_num_tiles);
        uint32_t l1_write_offset = 0;
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            noc.async_read(
                s0, cb, tile_size_bytes, {.page_id = in0_tensor_tile_id + i}, {.offset_bytes = l1_write_offset});
            l1_write_offset += tile_size_bytes;
        }
        in0_tensor_tile_id += q_num_tiles;
        noc.async_read_barrier();
        cb.push_back(q_num_tiles);

        // ---- K chunk ----
        cb.reserve_back(kv_num_tiles);
        l1_write_offset = 0;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            noc.async_read(
                s0, cb, tile_size_bytes, {.page_id = in0_tensor_tile_id + i}, {.offset_bytes = l1_write_offset});
            l1_write_offset += tile_size_bytes;
        }
        in0_tensor_tile_id += kv_num_tiles;
        noc.async_read_barrier();
        cb.push_back(kv_num_tiles);

        // ---- V chunk ----
        cb.reserve_back(kv_num_tiles);
        l1_write_offset = 0;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            noc.async_read(
                s0, cb, tile_size_bytes, {.page_id = in0_tensor_tile_id + i}, {.offset_bytes = l1_write_offset});
            l1_write_offset += tile_size_bytes;
        }
        in0_tensor_tile_id += kv_num_tiles;
        noc.async_read_barrier();
        cb.push_back(kv_num_tiles);
    }
}
