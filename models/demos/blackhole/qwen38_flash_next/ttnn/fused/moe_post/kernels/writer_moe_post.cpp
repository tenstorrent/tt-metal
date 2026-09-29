// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE post program, writer: the local sum's column tile into the reduce-scatter's input.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_col = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t tile_bytes = 2048;
    constexpr auto out_args = TensorAccessorArgs<0, 0>();
    const auto out = TensorAccessor(out_args, out_addr);

    DataflowBuffer out_tile(cb_out);
    {
        FUSED_ZONE("fz_mp_w_write");
        out_tile.wait_front(1);
        noc_async_write(out_tile.get_read_ptr(), out.get_noc_addr(tile_col), tile_bytes);
        noc_async_write_barrier();
    }
    out_tile.pop_front(1);
}
