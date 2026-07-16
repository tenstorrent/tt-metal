// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Scatter compact 32-row expert segments into the baseline [E*C,H] tiled
// layout.  Each output tile is owned by one core.  Unused capacity rows are
// written as zero; active rows copy one tile from packed_down.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t map_read_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t map_cb_page_bytes = get_compile_time_arg_val(3);

    constexpr auto map_args = TensorAccessorArgs<4>();
    constexpr auto packed_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();

    const uint32_t map_addr = get_arg_val<uint32_t>(0);
    const uint32_t packed_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);
    const uint32_t end_tile = get_arg_val<uint32_t>(4);

    Noc noc;
    CircularBuffer cb_map(tt::CBIndex::c_0);
    CircularBuffer cb_source(tt::CBIndex::c_1);
    CircularBuffer cb_zero(tt::CBIndex::c_2);
    const auto s_map = TensorAccessor(map_args, map_addr);
    const auto s_packed = TensorAccessor(packed_args, packed_addr);
    const auto s_output = TensorAccessor(output_args, output_addr);

    const uint32_t map_l1_offset = (map_addr - cb_map.get_write_ptr()) & 0x3Fu;
    cb_map.reserve_back(1);
    noc.async_read(s_map, cb_map, map_read_bytes, {.page_id = 0}, {.offset_bytes = map_l1_offset});
    noc.async_read_barrier();
    cb_map.push_back(1);
    cb_map.wait_front(1);
    const uint32_t* dense_to_packed = reinterpret_cast<const uint32_t*>(cb_map.get_read_ptr() + map_l1_offset);

    const uint32_t zero_l1_offset = (output_addr - cb_zero.get_write_ptr()) & 0x3Fu;
    cb_zero.reserve_back(1);
    volatile tt_l1_ptr uint16_t* zero =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_zero.get_write_ptr() + zero_l1_offset);
    for (uint32_t element = 0; element < tile_bytes / sizeof(uint16_t); ++element) {
        zero[element] = 0;
    }
    cb_zero.push_back(1);
    cb_zero.wait_front(1);

    const uint32_t source_l1_offset = (packed_addr - cb_source.get_write_ptr()) & 0x3Fu;
    for (uint32_t output_tile = start_tile; output_tile < end_tile; ++output_tile) {
        const uint32_t dense_tile_row = output_tile / hidden_tiles;
        const uint32_t hidden_tile = output_tile % hidden_tiles;
        const uint32_t packed_tile_row = dense_to_packed[dense_tile_row];
        if (packed_tile_row == 0xFFFFFFFFu) {
            noc.async_write(cb_zero, s_output, tile_bytes, {.offset_bytes = zero_l1_offset}, {.page_id = output_tile});
        } else {
            cb_source.reserve_back(1);
            const uint32_t source_tile = packed_tile_row * hidden_tiles + hidden_tile;
            noc.async_read(
                s_packed, cb_source, tile_bytes, {.page_id = source_tile}, {.offset_bytes = source_l1_offset});
            noc.async_read_barrier();
            cb_source.push_back(1);
            cb_source.wait_front(1);
            noc.async_write(
                cb_source, s_output, tile_bytes, {.offset_bytes = source_l1_offset}, {.page_id = output_tile});
            noc.async_write_barrier();
            cb_source.pop_front(1);
        }
    }
    noc.async_write_barrier();
    cb_zero.pop_front(1);
    cb_map.pop_front(1);
}
