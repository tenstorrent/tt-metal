// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// glu_split writer (BRISC): the finished h tiles to out [T, I] tile (r, c). Ends with noc_async_full_barrier().

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(1);
constexpr auto out_args = TensorAccessorArgs<2>();
}  // namespace

void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const auto s_out = TensorAccessor(out_args, out_addr);
    CircularBuffer cbo(CB_OUT);
    for (uint32_t i = start; i < start + count; ++i) {
        cbo.wait_front(1);
        noc_async_write(cbo.get_read_ptr(), s_out.get_noc_addr(i), TILE_BYTES);
        noc_async_write_barrier();
        cbo.pop_front(1);
    }
    noc_async_full_barrier();
}
