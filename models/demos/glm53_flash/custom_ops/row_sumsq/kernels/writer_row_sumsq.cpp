// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the fused row sum of squares (tt-blaze G53F-0209): one fp32 output tile per tile row (column 0 holds the
// row sums, the [Ht*32, 1] result's tile padding as ttnn.sum keepdim lays it out), the reader's split of the rows.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t NCORES = get_compile_time_arg_val(1);
constexpr uint32_t GRID_X = get_compile_time_arg_val(2);
constexpr uint32_t cb_out_id = get_compile_time_arg_val(3);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
constexpr auto out_args = TensorAccessorArgs<5>();

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();
    const uint32_t base = Ht / NCORES, extra = Ht % NCORES;
    const uint32_t n = base + (core_idx < extra ? 1u : 0u);
    const uint32_t r0 = core_idx * base + (core_idx < extra ? core_idx : extra);

    const auto s_out = TensorAccessor(out_args, out_addr, tile_bytes);
    Noc noc;
    CircularBuffer cb_out(cb_out_id);
    for (uint32_t r = r0; r < r0 + n; ++r) {
        cb_out.wait_front(1);
        noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(cb_out_id)), s_out, tile_bytes, {}, {.page_id = r});
        noc.async_write_barrier();
        cb_out.pop_front(1);
    }
}
