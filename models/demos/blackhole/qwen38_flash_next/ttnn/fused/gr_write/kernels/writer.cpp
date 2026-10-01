// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Output tile of unit u (branch u % branches, column u / branches) -> page hidden_tiles * branch + column.
// Named compile-time args: cb_out, hidden_tiles, branches.  Compile-time args: TensorAccessorArgs(output) from 0.
// Runtime args: 0 output addr, 1 first unit, 2 units.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t branches = get_named_compile_time_arg_val("branches");
    constexpr auto out_args = TensorAccessorArgs<0>();

    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    Noc noc;
    DataflowBuffer dfb(cb_out);
    for (uint32_t u = first; u < first + units; ++u) {
        FUSED_ZONE("fz_gw_w_unit");
        dfb.wait_front(1);
        noc.async_write(
            dfb, out, tile_bytes, {.offset_bytes = 0}, {.page_id = hidden_tiles * (u % branches) + u / branches});
        noc.async_write_barrier();
        dfb.pop_front(1);
    }
}
