// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// [sdpa_prefill_splitk] Merge reader: per output row tile (plane h, tile row r) the 2S statistics tiles
// (m_0..m_{S-1}, then l_0..l_{S-1}) and the S slots' DHt-tile rows of the partial outputs.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t Sqt = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stats_in = get_compile_time_arg_val(4);
    constexpr uint32_t cb_o_in = get_compile_time_arg_val(5);
    constexpr auto parts_args = TensorAccessorArgs<6>();
    constexpr auto stats_args = TensorAccessorArgs<parts_args.next_compile_time_args_offset()>();

    const uint32_t parts_addr = get_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_stats_in);
    constexpr uint32_t o_tile_bytes = get_tile_size(cb_o_in);
    const auto parts = TensorAccessor(parts_args, parts_addr);
    const auto stats_reader = TensorAccessor(stats_args, stats_addr);
    const auto parts_shape = TensorTileShape(1, S * NQH, Sqt, DHt);
    const auto stats_shape = TensorTileShape(1, S * NQH, Sqt, 2);

    CircularBuffer cb_st(cb_stats_in);
    CircularBuffer cb_o(cb_o_in);
    for (uint32_t t = row_start; t < row_start + num_rows; ++t) {
        const uint32_t h = t / Sqt;
        const uint32_t r = t % Sqt;
        cb_st.reserve_back(2 * S);
        uint32_t wr = cb_st.get_write_ptr();
        for (uint32_t col = 0; col < 2; ++col) {
            for (uint32_t j = 0; j < S; ++j) {
                noc.async_read(
                    stats_reader,
                    CoreLocalMem<uint32_t>(wr),
                    stats_tile_bytes,
                    {.page_id = stats_shape.id_of(0, j * NQH + h, r, col)},
                    {});
                wr += stats_tile_bytes;
            }
        }
        noc.async_read_barrier();
        cb_st.push_back(2 * S);
        for (uint32_t j = 0; j < S; ++j) {
            cb_o.reserve_back(DHt);
            uint32_t owr = cb_o.get_write_ptr();
            const uint32_t first = parts_shape.id_of(0, j * NQH + h, r, 0);
            for (uint32_t c = 0; c < DHt; ++c) {
                noc.async_read(parts, CoreLocalMem<uint32_t>(owr), o_tile_bytes, {.page_id = first + c}, {});
                owr += o_tile_bytes;
            }
            noc.async_read_barrier();
            cb_o.push_back(DHt);
        }
    }
}
