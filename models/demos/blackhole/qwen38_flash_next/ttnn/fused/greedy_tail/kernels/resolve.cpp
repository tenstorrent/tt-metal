// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the four devices' packed [value | local id] fp32 pairs (the all_gather of merge.cpp's rows) -> the greedy
// token row: per row r, owner d's value lowered by owner_tie_break[d] (fp32 subtract, as the chain's SFPU fp32
// subtract), the first maximum in owner order (ttnn.argmax's lowest index), the owner's id plus
// lm_head_vocab_starts[owner] (fp32 add, exact below 2^24), written as lane r of row 0 of an fp32 TILE [1,1,1,32]
// whose other lanes are 0.0 (the chain's unit_column multiply for one row; lane u = row u for the lanes, lanes past
// the row count zero as the lanes chain's zero pad).  STRIDE = the fp32 lanes per device in a packed row: 2 (the
// decode step's [value | id]) or 16 (the lanes' 64-byte rows); gathered row r is page r.  The RISC's soft-float
// subtract and add are IEEE round-to-nearest-even, as the SFPU's.
// With copy_into = 1 the same tile is also written to a second TOKEN_ROW tensor (the server's persistent token row:
// the chain's ttnn.copy(token_row, token_row_io) after the resolve, as one more 4 KB write of this program).
// Named compile-time args: cb_stage, devices, copy_into, rows, stride.  Compile-time args:
// TensorAccessorArgs(gathered), (tie_break), (vocab_starts), (zero fp32 tile), (token_row), (into).  Runtime args: the
// six buffer addresses in that order (into repeats token_row when copy_into = 0).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t DEVICES = get_named_compile_time_arg_val("devices");
constexpr uint32_t COPY_INTO = get_named_compile_time_arg_val("copy_into");
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
constexpr uint32_t STRIDE = get_named_compile_time_arg_val("stride");
constexpr uint32_t FP32_TILE_BYTES = 4096;
constexpr uint32_t GRAIN = 64;
constexpr uint32_t ROW_BYTES =
    (DEVICES * STRIDE * 4 + GRAIN - 1) & ~(GRAIN - 1);  // one gathered row, at the read grain

void kernel_main() {
    constexpr auto a_gathered = TensorAccessorArgs<0>();
    constexpr auto a_tie = TensorAccessorArgs<a_gathered.next_compile_time_args_offset()>();
    constexpr auto a_starts = TensorAccessorArgs<a_tie.next_compile_time_args_offset()>();
    constexpr auto a_zero = TensorAccessorArgs<a_starts.next_compile_time_args_offset()>();
    constexpr auto a_token = TensorAccessorArgs<a_zero.next_compile_time_args_offset()>();
    constexpr auto a_into = TensorAccessorArgs<a_token.next_compile_time_args_offset()>();
    const auto gathered = TensorAccessor(a_gathered, get_arg_val<uint32_t>(0));
    const auto tie = TensorAccessor(a_tie, get_arg_val<uint32_t>(1));
    const auto starts = TensorAccessor(a_starts, get_arg_val<uint32_t>(2));
    const auto zero = TensorAccessor(a_zero, get_arg_val<uint32_t>(3));
    const auto token = TensorAccessor(a_token, get_arg_val<uint32_t>(4));
    const auto into = TensorAccessor(a_into, get_arg_val<uint32_t>(5));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    constexpr uint32_t STAGE_TILE = 0, STAGE_GATHERED = FP32_TILE_BYTES, STAGE_TIE = STAGE_GATHERED + ROWS * ROW_BYTES,
                       STAGE_STARTS = STAGE_TIE + GRAIN;
    {
        FUSED_ZONE("fz_gt_res_setup");
        noc.async_read(zero, stage, FP32_TILE_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TILE});
        for (uint32_t r = 0; r < ROWS; ++r) {
            noc.async_read(
                gathered,
                stage,
                ROW_BYTES,
                {.page_id = r, .offset_bytes = 0},
                {.offset_bytes = STAGE_GATHERED + r * ROW_BYTES});
        }
        noc.async_read(tie, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TIE});
        noc.async_read(starts, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_STARTS});
        noc.async_read_barrier();
    }

    {
        FUSED_ZONE("fz_gt_res_main");
        volatile tt_l1_ptr float* t = reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_TIE);
        volatile tt_l1_ptr float* s = reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_STARTS);
        volatile tt_l1_ptr float* row = reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_TILE);
        for (uint32_t r = 0; r < ROWS; ++r) {
            volatile tt_l1_ptr float* g =
                reinterpret_cast<volatile tt_l1_ptr float*>(base + STAGE_GATHERED + r * ROW_BYTES);
            uint32_t owner = 0;
            float best = g[0] - t[0];
            for (uint32_t d = 1; d < DEVICES; ++d) {
                const float ranked = g[STRIDE * d] - t[d];
                if (ranked > best) {
                    best = ranked;
                    owner = d;
                }
            }
            const float id = g[STRIDE * owner + 1] + s[owner];
            // lane (0, r) of the fp32 token tile: face r >> 4 (1024 bytes = 256 words each), row 0, column r & 15
            row[(r >> 4) * 256 + (r & 15)] = id;
        }
        noc.async_write(stage, token, FP32_TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        if constexpr (COPY_INTO) {
            noc.async_write(
                stage, into, FP32_TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
