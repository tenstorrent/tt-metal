// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE conv writer, one core per column block: the tiles_per_core delta tiles (CB 16) -> the output pages
// first..first+tiles_per_core-1 (the state shift is the reader's).  With inject (tiles_per_core = 1): the delta tile's
// rows 0-3 become four row-0 tiles in CB 18 (zero elsewhere: the layer's permute of the delta into the branch-major
// layout), the compute adds the residual blocks, and the four injected tiles (CB 20) go to pages b * 20 + first of the
// injected residual instead of the delta.
// Compile-time args: 0 tiles_per_core (the builder's per_core), 1 inject, then TensorAccessorArgs(out), (injected).
// Runtime args: 0 out addr, 1 the first tile of this core's column block, 2 injected addr.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t inject = get_compile_time_arg_val(1);
constexpr uint32_t bf16_tile_bytes = 2048, bf16_face_bytes = 512, bf16_face_row_bytes = 32, branch_tiles = 20;
constexpr uint32_t cb_out = 16, cb_delta_rows = 18, cb_injected = 20;

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr auto injected_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1);
    const Noc noc;
    DataflowBuffer delta(cb_out);
    {
        delta.wait_front(tiles_per_core);
        if constexpr (!inject) {
            for (uint32_t t = 0; t < tiles_per_core; ++t) {
                noc.async_write(
                    delta,
                    out,
                    bf16_tile_bytes,
                    {.offset_bytes = t * bf16_tile_bytes},
                    {.page_id = first + t, .offset_bytes = 0});
            }
            noc.async_write_barrier();
            delta.pop_front(tiles_per_core);
            return;
        }
    }
    // inject: the delta tile's row b (face 0 row b, face 1 row b: 32 bytes each) -> row 0 of tile b in CB 18
    {
        const auto injected = TensorAccessor(injected_args, get_arg_val<uint32_t>(2));
        DataflowBuffer delta_rows(cb_delta_rows), injected_tiles(cb_injected);
        delta_rows.reserve_back(4);
        CoreLocalMem<volatile uint32_t> src(delta.get_read_ptr());
        CoreLocalMem<volatile uint32_t> dst(delta_rows.get_write_ptr());
        for (uint32_t w = 0; w < 4 * bf16_tile_bytes / 4; ++w) {
            dst[w] = 0;
        }
        for (uint32_t b = 0; b < 4; ++b) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t src_word =
                    (face * bf16_face_bytes + b * bf16_face_row_bytes) / 4;  // delta tile, face `face`, row b
                const uint32_t dst_word =
                    (b * bf16_tile_bytes + face * bf16_face_bytes) / 4;  // tile b, face `face`, row 0
                for (uint32_t k = 0; k < bf16_face_row_bytes / 4; ++k) {
                    dst[dst_word + k] = src[src_word + k];
                }
            }
        }
        delta_rows.push_back(4);
        delta.pop_front(tiles_per_core);
        injected_tiles.wait_front(4);
        for (uint32_t b = 0; b < 4; ++b) {
            noc.async_write(
                injected_tiles,
                injected,
                bf16_tile_bytes,
                {.offset_bytes = b * bf16_tile_bytes},
                {.page_id = b * branch_tiles + first, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        injected_tiles.pop_front(4);
    }
}
