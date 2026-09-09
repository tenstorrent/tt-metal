// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Baseline: the same per-receiver bytes pulled straight from DRAM by the worker core itself
// (what moe_fused_swiglu's phase 1 does), tile by tile through a TensorAccessor, into a small
// L1 staging ring, `inflight` reads deep. The last block is left in the output shard.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);  // tiles this core reads per layer
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t inflight = get_compile_time_arg_val(2);  // staging ring depth in tiles
    constexpr uint32_t layers = get_compile_time_arg_val(3);
    constexpr uint32_t stage_cb = get_compile_time_arg_val(4);
    constexpr auto tensor_args = TensorAccessorArgs<5>();

    const uint32_t tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(1);  // tile id of this core's slice, top-left
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_tiles = get_arg_val<uint32_t>(3);      // trailing tiles of the slice to keep
    const uint32_t row_stride = get_arg_val<uint32_t>(4);     // tile ids per K-row of the tensor (N tiles)
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(5);  // this slice's width in tiles

    const auto accessor = TensorAccessor(tensor_args, tensor_addr, tile_bytes);
    const uint32_t stage = get_write_ptr(stage_cb);
    for (uint32_t l = 0; l < layers; ++l) {
        for (uint32_t t = 0; t < num_tiles; t += inflight) {
            const uint32_t n = (num_tiles - t < inflight) ? num_tiles - t : inflight;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t idx = t + i;  // K-row-major over the slice: row = idx / tiles_per_row
                const uint32_t page = first_page + (idx / tiles_per_row) * row_stride + (idx % tiles_per_row);
                const bool keep = idx + out_tiles >= num_tiles;
                const uint32_t dst =
                    keep ? out_addr + (idx + out_tiles - num_tiles) * tile_bytes : stage + i * tile_bytes;
                noc_async_read_page(page, accessor, dst);
            }
            noc_async_read_barrier();
        }
    }
}
