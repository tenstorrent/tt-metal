// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the DeepSeek-V4.1 decode MoE tail (tt/v41/moe_tail.py): this core's TPC output tiles to tile pages
// j0 .. j0 + TPC of the [1, 1, 32, H] bf16 TILE output.
//
// compile_time_args = [cb_out, TPC, TensorAccessorArgs(out)...]
// runtime args      = [out_addr, j0]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t TPC = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr uint32_t TILE_BYTES = 2048;

    const auto out = TensorAccessor(out_args, out_addr);
    DataflowBuffer cb(cb_out);
    cb.wait_front(TPC);
    const uint32_t l1 = cb.get_read_ptr();
    for (uint32_t jj = 0; jj < TPC; ++jj) {
        noc_async_write(l1 + jj * TILE_BYTES, out.get_noc_addr(j0 + jj, 0), TILE_BYTES);
    }
    noc_async_write_barrier();
    cb.pop_front(TPC);
}
