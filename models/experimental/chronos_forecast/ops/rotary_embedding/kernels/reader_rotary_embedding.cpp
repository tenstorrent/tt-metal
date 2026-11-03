// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Prefill reader for the cached-cos/sin path. cos/sin are batch-shared [1, 1, S, X], so the whole
// Ht x Wt table is read once per core and left resident; the compute kernel indexes it by row.
// The input is streamed in blocks of rows_per_block rows with one read barrier per block.
void kernel_main() {
    Noc noc;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t cos_addr = get_arg_val<uint32_t>(1);
    uint32_t sin_addr = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(3);
    constexpr uint16_t scalar_value = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t HtWt = get_compile_time_arg_val(6);
    constexpr uint32_t rows_per_block = get_compile_time_arg_val(7);
    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto cos_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s0 = TensorAccessor(src_args, src_addr);
    const auto s1 = TensorAccessor(cos_args, cos_addr);
    const auto s2 = TensorAccessor(sin_args, sin_addr);

    CircularBuffer cb_input(input_cb_id);
    CircularBuffer cb_cos(cos_cb_id);
    CircularBuffer cb_sin(sin_cb_id);
    CircularBuffer cb_scalar(scalar_cb_id);

    cb_scalar.reserve_back(1);
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_scalar.get_write_ptr());
    scalar_buffer[0] = scalar_value;
    cb_scalar.push_back(1);

    cb_sin.reserve_back(HtWt);
    cb_cos.reserve_back(HtWt);
    uint32_t sin_l1_write_addr = cb_sin.get_write_ptr();
    uint32_t cos_l1_write_addr = cb_cos.get_write_ptr();
    for (uint32_t t = 0; t < HtWt; ++t) {
        noc.async_read(s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = t}, {});
        noc.async_read(s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = t}, {});
        sin_l1_write_addr += sin_tile_bytes;
        cos_l1_write_addr += cos_tile_bytes;
    }
    noc.async_read_barrier();
    cb_sin.push_back(HtWt);
    cb_cos.push_back(HtWt);

    uint32_t tile_id = start_id;
    for (uint32_t rows_done = 0; rows_done < num_rows; rows_done += rows_per_block) {
        const uint32_t rows = (num_rows - rows_done) < rows_per_block ? (num_rows - rows_done) : rows_per_block;
        const uint32_t num_tiles = rows * Wt;
        cb_input.reserve_back(num_tiles);
        uint32_t l1_write_addr = cb_input.get_write_ptr();
        for (uint32_t t = 0; t < num_tiles; ++t) {
            noc.async_read(s0, CoreLocalMem<uint32_t>(l1_write_addr), input_tile_bytes, {.page_id = tile_id}, {});
            l1_write_addr += input_tile_bytes;
            ++tile_id;
        }
        noc.async_read_barrier();
        cb_input.push_back(num_tiles);
    }
}
