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
// Difference vs stock: stock issues `cb_reserve(1) / read_tile / barrier /
// cb_push(1)` once per tile (192 NoC barriers per block-of-96-tiles). This
// kernel batches each of the Q/K/V tile groups into a single CB reservation
// and a single barrier, dropping per-block barrier count from 96 to 3.
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

    for (uint32_t block = 0; block < num_blocks; block++) {
        // ---- Q chunk: q_num_tiles in one reserve / one barrier / one push ----
        cb_reserve_back(cb_id, q_num_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            noc_async_read_tile(in0_tensor_tile_id + i, s0, l1_write_addr);
            l1_write_addr += tile_size_bytes;
        }
        in0_tensor_tile_id += q_num_tiles;
        noc_async_read_barrier();
        cb_push_back(cb_id, q_num_tiles);

        // ---- K chunk ----
        cb_reserve_back(cb_id, kv_num_tiles);
        l1_write_addr = get_write_ptr(cb_id);
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            noc_async_read_tile(in0_tensor_tile_id + i, s0, l1_write_addr);
            l1_write_addr += tile_size_bytes;
        }
        in0_tensor_tile_id += kv_num_tiles;
        noc_async_read_barrier();
        cb_push_back(cb_id, kv_num_tiles);

        // ---- V chunk ----
        cb_reserve_back(cb_id, kv_num_tiles);
        l1_write_addr = get_write_ptr(cb_id);
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            noc_async_read_tile(in0_tensor_tile_id + i, s0, l1_write_addr);
            l1_write_addr += tile_size_bytes;
        }
        in0_tensor_tile_id += kv_num_tiles;
        noc_async_read_barrier();
        cb_push_back(cb_id, kv_num_tiles);
    }
}
