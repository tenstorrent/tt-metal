// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE stage 5 reader, one core per column block of T tiles (tiles first..first+T-1 of every tensor): nine bf16 TILE
// [1,1,4,640] tensors into CBs 0-8 in order: conv[0],
// conv[3], conv[6], normalized, taps 0-3, gated.  Then, with SHIFT, the chain's nine in-place state copies
// (conv[k] <- conv[k+1] for k = 0..7, conv[8] <- normalized) while the compute runs: every input the compute needs is
// already in its CBs, so the state rows may be overwritten; each row moves as T tiles through the staging CB with
// one read barrier and one write barrier (the writer's tile-by-tile form cost 165 us).
// Compile-time args: 0 T, 1 SHIFT, then TensorAccessorArgs for conv[0], conv[3], conv[6], normalized, taps 0-3,
// gated, conv[1], conv[2], conv[4], conv[5], conv[7], conv[8], and the branch-major residual [1,4,1,640] (sixteen,
// chained; the six state rows only for the shift, the residual only with INJECT: its four block tiles b * 20 + first
// go to CB 19).  Compile-time args 0 T, 1 SHIFT, 2 INJECT.  Runtime args: the sixteen buffer addresses in that order,
// 16 the first tile of this core's column block, 17 the first tap tile of the block (the column within the lane's
// 20-tile row-block: the taps are one [1,1,4,640] per tap for every lane; the 1-row form passes the same as 16).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t T = get_compile_time_arg_val(0);
constexpr uint32_t SHIFT = get_compile_time_arg_val(1);
constexpr uint32_t INJECT = get_compile_time_arg_val(2);
constexpr uint32_t BLOCK_TILES = 20;  // tiles per residual branch block
constexpr uint32_t BF16_TILE = 2048;
constexpr uint32_t c_stage = 17;

template <typename Acc>
FORCE_INLINE void stream(Noc& noc, const Acc& acc, uint32_t cb, uint32_t first) {
    DataflowBuffer dfb(cb);
    dfb.reserve_back(T);
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(acc, dfb, BF16_TILE, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = t * BF16_TILE});
    }
    noc.async_read_barrier();
    dfb.push_back(T);
}

template <typename Src, typename Dst>
FORCE_INLINE void shift_row(Noc& noc, const Src& src, const Dst& dst, uint32_t first) {
    DataflowBuffer stage(c_stage);
    stage.reserve_back(T);
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(
            src, stage, BF16_TILE, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = t * BF16_TILE});
    }
    noc.async_read_barrier();
    stage.push_back(T);
    stage.wait_front(T);
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_write(
            stage, dst, BF16_TILE, {.offset_bytes = t * BF16_TILE}, {.page_id = first + t, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    stage.pop_front(T);
}

void kernel_main() {
    constexpr auto a0 = TensorAccessorArgs<3>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    constexpr auto a8 = TensorAccessorArgs<a7.next_compile_time_args_offset()>();
    constexpr auto a9 = TensorAccessorArgs<a8.next_compile_time_args_offset()>();
    constexpr auto a10 = TensorAccessorArgs<a9.next_compile_time_args_offset()>();
    constexpr auto a11 = TensorAccessorArgs<a10.next_compile_time_args_offset()>();
    constexpr auto a12 = TensorAccessorArgs<a11.next_compile_time_args_offset()>();
    constexpr auto a13 = TensorAccessorArgs<a12.next_compile_time_args_offset()>();
    constexpr auto a14 = TensorAccessorArgs<a13.next_compile_time_args_offset()>();
    constexpr auto a15 = TensorAccessorArgs<a14.next_compile_time_args_offset()>();
    const uint32_t first = get_arg_val<uint32_t>(16);
    const uint32_t tap_first = get_arg_val<uint32_t>(17);
    Noc noc;
    const auto c0 = TensorAccessor(a0, get_arg_val<uint32_t>(0));
    const auto c3 = TensorAccessor(a1, get_arg_val<uint32_t>(1));
    const auto c6 = TensorAccessor(a2, get_arg_val<uint32_t>(2));
    const auto n = TensorAccessor(a3, get_arg_val<uint32_t>(3));
    {
        FUSED_ZONE("fz_pl_conv_r_streams");
        stream(noc, c0, 0, first);
        stream(noc, c3, 1, first);
        stream(noc, c6, 2, first);
        stream(noc, n, 3, first);
        stream(noc, TensorAccessor(a4, get_arg_val<uint32_t>(4)), 4, tap_first);
        stream(noc, TensorAccessor(a5, get_arg_val<uint32_t>(5)), 5, tap_first);
        stream(noc, TensorAccessor(a6, get_arg_val<uint32_t>(6)), 6, tap_first);
        stream(noc, TensorAccessor(a7, get_arg_val<uint32_t>(7)), 7, tap_first);
        stream(noc, TensorAccessor(a8, get_arg_val<uint32_t>(8)), 8, first);
    }
    if constexpr (INJECT) {
        FUSED_ZONE("fz_pl_conv_r_residual");
        const auto residual = TensorAccessor(a15, get_arg_val<uint32_t>(15));
        DataflowBuffer resid(19);
        resid.reserve_back(4);
        for (uint32_t b = 0; b < 4; ++b) {
            noc.async_read(
                residual,
                resid,
                BF16_TILE,
                {.page_id = b * BLOCK_TILES + first, .offset_bytes = 0},
                {.offset_bytes = b * BF16_TILE});
        }
        noc.async_read_barrier();
        resid.push_back(4);
    }
    if constexpr (SHIFT) {
        FUSED_ZONE("fz_pl_conv_r_shift");
        const auto c1 = TensorAccessor(a9, get_arg_val<uint32_t>(9));
        const auto c2 = TensorAccessor(a10, get_arg_val<uint32_t>(10));
        const auto c4 = TensorAccessor(a11, get_arg_val<uint32_t>(11));
        const auto c5 = TensorAccessor(a12, get_arg_val<uint32_t>(12));
        const auto c7 = TensorAccessor(a13, get_arg_val<uint32_t>(13));
        const auto c8 = TensorAccessor(a14, get_arg_val<uint32_t>(14));
        shift_row(noc, c1, c0, first);
        shift_row(noc, c2, c1, first);
        shift_row(noc, c3, c2, first);
        shift_row(noc, c4, c3, first);
        shift_row(noc, c5, c4, first);
        shift_row(noc, c6, c5, first);
        shift_row(noc, c7, c6, first);
        shift_row(noc, c8, c7, first);
        shift_row(noc, n, c8, first);
    }
}
