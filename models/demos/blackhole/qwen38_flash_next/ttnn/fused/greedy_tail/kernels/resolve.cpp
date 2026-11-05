// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the devices' gathered packed [value | local id] fp32 pairs (merge.cpp's rows) -> the greedy token row.
// Per row r: owner d's value minus owner_tie_break[d], the first maximum in owner order (ttnn.argmax's lowest index),
// and the owner's id plus lm_head_vocab_starts[owner] (exact below 2^24), written as lane r of row 0 of an fp32 TILE
// [1,1,1,32] whose other lanes are 0.0.  The RISC's soft-float subtract and add are IEEE round-to-nearest-even, as the
// SFPU's.  stride = the fp32 lanes per device in a packed row (2 or 16); gathered row r is page r.
// With copy_into = 1 the same tile is also written to a second TOKEN_ROW tensor.
// Named compile-time args: cb_stage, devices, copy_into, rows, stride.  Compile-time args:
// TensorAccessorArgs(gathered), (tie_break), (vocab_starts), (zero fp32 tile), (token_row), (into).  Runtime args: the
// six buffer addresses in that order (into repeats token_row when copy_into = 0).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t devices = get_named_compile_time_arg_val("devices");
constexpr uint32_t copy_into = get_named_compile_time_arg_val("copy_into");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t stride = get_named_compile_time_arg_val("stride");
constexpr uint32_t fp32_tile_bytes = 4096;
constexpr uint32_t grain_bytes = 64;
constexpr uint32_t gathered_row_bytes =
    (devices * stride * 4 + grain_bytes - 1) & ~(grain_bytes - 1);  // one gathered row, at the read grain

void kernel_main() {
    constexpr auto gathered_args = TensorAccessorArgs<0>();
    constexpr auto tie_args = TensorAccessorArgs<gathered_args.next_compile_time_args_offset()>();
    constexpr auto starts_args = TensorAccessorArgs<tie_args.next_compile_time_args_offset()>();
    constexpr auto zero_args = TensorAccessorArgs<starts_args.next_compile_time_args_offset()>();
    constexpr auto token_args = TensorAccessorArgs<zero_args.next_compile_time_args_offset()>();
    constexpr auto into_args = TensorAccessorArgs<token_args.next_compile_time_args_offset()>();
    const auto gathered = TensorAccessor(gathered_args, get_arg_val<uint32_t>(0));
    const auto tie = TensorAccessor(tie_args, get_arg_val<uint32_t>(1));
    const auto starts = TensorAccessor(starts_args, get_arg_val<uint32_t>(2));
    const auto zero = TensorAccessor(zero_args, get_arg_val<uint32_t>(3));
    const auto token = TensorAccessor(token_args, get_arg_val<uint32_t>(4));
    const auto into = TensorAccessor(into_args, get_arg_val<uint32_t>(5));

    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    constexpr uint32_t stage_tile = 0, stage_gathered = fp32_tile_bytes,
                       stage_tie = stage_gathered + rows * gathered_row_bytes, stage_starts = stage_tie + grain_bytes;
    {
        noc.async_read(zero, stage, fp32_tile_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_tile});
        for (uint32_t r = 0; r < rows; ++r) {
            noc.async_read(
                gathered,
                stage,
                gathered_row_bytes,
                {.page_id = r, .offset_bytes = 0},
                {.offset_bytes = stage_gathered + r * gathered_row_bytes});
        }
        noc.async_read(tie, stage, grain_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_tie});
        noc.async_read(starts, stage, grain_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_starts});
        noc.async_read_barrier();
    }

    {
        CoreLocalMem<volatile float> tie_break(base + stage_tie);
        CoreLocalMem<volatile float> vocab_starts(base + stage_starts);
        CoreLocalMem<volatile float> token_tile(base + stage_tile);
        for (uint32_t r = 0; r < rows; ++r) {
            CoreLocalMem<volatile float> gathered_row(base + stage_gathered + r * gathered_row_bytes);
            uint32_t owner = 0;
            float best = gathered_row[0] - tie_break[0];
            for (uint32_t d = 1; d < devices; ++d) {
                const float ranked = gathered_row[stride * d] - tie_break[d];
                if (ranked > best) {
                    best = ranked;
                    owner = d;
                }
            }
            const float id = gathered_row[stride * owner + 1] + vocab_starts[owner];
            // lane (0, r) of the fp32 token tile: face r >> 4 (1024 bytes = 256 words each), row 0, column r & 15
            token_tile[(r >> 4) * 256 + (r & 15)] = id;
        }
        noc.async_write(stage, token, fp32_tile_bytes, {.offset_bytes = stage_tile}, {.page_id = 0, .offset_bytes = 0});
        if constexpr (copy_into) {
            noc.async_write(
                stage, into, fp32_tile_bytes, {.offset_bytes = stage_tile}, {.page_id = 0, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
