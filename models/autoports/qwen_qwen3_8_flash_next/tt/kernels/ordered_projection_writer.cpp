// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t column = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 4;
    cb_wait_front(cb_out, 1);
    noc_async_write(get_read_ptr(cb_out), output.get_noc_addr(column), tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
