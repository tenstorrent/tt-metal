// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// [sdpa_prefill_splitk] Merge writer: each merged DHt-tile row to [1, NQH, Sq, D].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t NQH = get_compile_time_arg_val(0);
    constexpr uint32_t Sqt = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr bool with_stats = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t cb_stats_out = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr auto stats_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t stats_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto out_shape = TensorTileShape(1, NQH, Sqt, DHt);
    const auto stats_writer = TensorAccessor(stats_args, stats_addr);
    const auto stats_shape = TensorTileShape(1, NQH, Sqt, 2);
    CircularBuffer cb_st(cb_stats_out);
    for (uint32_t t = row_start; t < row_start + num_rows; ++t) {
        if constexpr (with_stats) {
            // the compute pushes M, then L, before the merged row
            constexpr uint32_t st_bytes = get_tile_size(cb_stats_out);
            cb_st.wait_front(2);
            for (uint32_t col = 0; col < 2; ++col) {
                noc.async_write(
                    cb_st,
                    stats_writer,
                    st_bytes,
                    {.offset_bytes = col * st_bytes},
                    {.page_id = stats_shape.id_of(0, t / Sqt, t % Sqt, col)});
            }
            noc.async_write_barrier();
            cb_st.pop_front(2);
        }
        write_block<0>(noc, out_writer, cb_out, DHt, 1, DHt, out_shape.id_of(0, t / Sqt, t % Sqt, 0), tile_bytes, 8);
    }
}
