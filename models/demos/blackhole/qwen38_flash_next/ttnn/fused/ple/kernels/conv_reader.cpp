// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE conv reader, one core per column block of tiles_per_core tiles (tiles first..first+tiles_per_core-1 of every
// tensor): nine bf16 TILE [1,1,4,640] tensors into CBs 0-8 in order: conv[0], conv[3], conv[6], normalized, taps 0-3,
// gated.  Then, with shift, the chain's nine in-place state copies (conv[k] <- conv[k+1] for k = 0..7, conv[8] <-
// normalized): every input the compute needs is already in its CBs, so the state rows may be overwritten.
// Compile-time args: 0 tiles_per_core (the builder's per_core), 1 shift, 2 inject, then TensorAccessorArgs for
// conv[0], conv[3], conv[6], normalized, taps 0-3, gated, conv[1], conv[2], conv[4], conv[5], conv[7], conv[8], and
// the branch-major residual [1,4,1,640] (sixteen, chained; the six state rows only for the shift, the residual only
// with inject: its four block tiles b * 20 + first go to CB 19).  Runtime args: the sixteen buffer addresses in that
// order, 16 the first tile of this core's column block, 17 the first tap tile of the block (the column within the
// lane's 20-tile row-block: the taps are one [1,1,4,640] per tap for every lane; the 1-row form passes the same as 16).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t shift = get_compile_time_arg_val(1);
constexpr uint32_t inject = get_compile_time_arg_val(2);
constexpr uint32_t branch_tiles = 20;  // tiles per residual branch block
constexpr uint32_t bf16_tile_bytes = 2048;
constexpr uint32_t cb_stage = 17;

template <typename Accessor>
FORCE_INLINE void stream(const Accessor& tensor, uint32_t cb_id, uint32_t first) {
    const Noc noc;
    DataflowBuffer buffer(cb_id);
    buffer.reserve_back(tiles_per_core);
    for (uint32_t t = 0; t < tiles_per_core; ++t) {
        noc.async_read(
            tensor,
            buffer,
            bf16_tile_bytes,
            {.page_id = first + t, .offset_bytes = 0},
            {.offset_bytes = t * bf16_tile_bytes});
    }
    noc.async_read_barrier();
    buffer.push_back(tiles_per_core);
}

template <typename Src, typename Dst>
FORCE_INLINE void shift_row(const Src& src, const Dst& dst, uint32_t first) {
    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(tiles_per_core);
    for (uint32_t t = 0; t < tiles_per_core; ++t) {
        noc.async_read(
            src,
            stage,
            bf16_tile_bytes,
            {.page_id = first + t, .offset_bytes = 0},
            {.offset_bytes = t * bf16_tile_bytes});
    }
    noc.async_read_barrier();
    stage.push_back(tiles_per_core);
    stage.wait_front(tiles_per_core);
    for (uint32_t t = 0; t < tiles_per_core; ++t) {
        noc.async_write(
            stage,
            dst,
            bf16_tile_bytes,
            {.offset_bytes = t * bf16_tile_bytes},
            {.page_id = first + t, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    stage.pop_front(tiles_per_core);
}

void kernel_main() {
    constexpr auto conv0_args = TensorAccessorArgs<3>();
    constexpr auto conv3_args = TensorAccessorArgs<conv0_args.next_compile_time_args_offset()>();
    constexpr auto conv6_args = TensorAccessorArgs<conv3_args.next_compile_time_args_offset()>();
    constexpr auto normalized_args = TensorAccessorArgs<conv6_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<normalized_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto gated_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto conv1_args = TensorAccessorArgs<gated_args.next_compile_time_args_offset()>();
    constexpr auto conv2_args = TensorAccessorArgs<conv1_args.next_compile_time_args_offset()>();
    constexpr auto conv4_args = TensorAccessorArgs<conv2_args.next_compile_time_args_offset()>();
    constexpr auto conv5_args = TensorAccessorArgs<conv4_args.next_compile_time_args_offset()>();
    constexpr auto conv7_args = TensorAccessorArgs<conv5_args.next_compile_time_args_offset()>();
    constexpr auto conv8_args = TensorAccessorArgs<conv7_args.next_compile_time_args_offset()>();
    constexpr auto residual_args = TensorAccessorArgs<conv8_args.next_compile_time_args_offset()>();
    const uint32_t first = get_arg_val<uint32_t>(16);
    const uint32_t tap_first = get_arg_val<uint32_t>(17);
    const Noc noc;
    const auto conv0 = TensorAccessor(conv0_args, get_arg_val<uint32_t>(0));
    const auto conv3 = TensorAccessor(conv3_args, get_arg_val<uint32_t>(1));
    const auto conv6 = TensorAccessor(conv6_args, get_arg_val<uint32_t>(2));
    const auto normalized = TensorAccessor(normalized_args, get_arg_val<uint32_t>(3));
    {
        stream(conv0, 0, first);
        stream(conv3, 1, first);
        stream(conv6, 2, first);
        stream(normalized, 3, first);
        stream(TensorAccessor(tap0_args, get_arg_val<uint32_t>(4)), 4, tap_first);
        stream(TensorAccessor(tap1_args, get_arg_val<uint32_t>(5)), 5, tap_first);
        stream(TensorAccessor(tap2_args, get_arg_val<uint32_t>(6)), 6, tap_first);
        stream(TensorAccessor(tap3_args, get_arg_val<uint32_t>(7)), 7, tap_first);
        stream(TensorAccessor(gated_args, get_arg_val<uint32_t>(8)), 8, first);
    }
    if constexpr (inject) {
        const auto residual = TensorAccessor(residual_args, get_arg_val<uint32_t>(15));
        DataflowBuffer residual_tiles(19);
        residual_tiles.reserve_back(4);
        for (uint32_t b = 0; b < 4; ++b) {
            noc.async_read(
                residual,
                residual_tiles,
                bf16_tile_bytes,
                {.page_id = b * branch_tiles + first, .offset_bytes = 0},
                {.offset_bytes = b * bf16_tile_bytes});
        }
        noc.async_read_barrier();
        residual_tiles.push_back(4);
    }
    if constexpr (shift) {
        const auto conv1 = TensorAccessor(conv1_args, get_arg_val<uint32_t>(9));
        const auto conv2 = TensorAccessor(conv2_args, get_arg_val<uint32_t>(10));
        const auto conv4 = TensorAccessor(conv4_args, get_arg_val<uint32_t>(11));
        const auto conv5 = TensorAccessor(conv5_args, get_arg_val<uint32_t>(12));
        const auto conv7 = TensorAccessor(conv7_args, get_arg_val<uint32_t>(13));
        const auto conv8 = TensorAccessor(conv8_args, get_arg_val<uint32_t>(14));
        shift_row(conv1, conv0, first);
        shift_row(conv2, conv1, first);
        shift_row(conv3, conv2, first);
        shift_row(conv4, conv3, first);
        shift_row(conv5, conv4, first);
        shift_row(conv6, conv5, first);
        shift_row(conv7, conv6, first);
        shift_row(conv8, conv7, first);
        shift_row(normalized, conv8, first);
    }
}
