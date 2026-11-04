// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill rotation, writer (BRISC): output tiles (m, 32 b + c).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t Kt = get_common_arg_val<uint32_t>(1);
    const uint32_t b = get_arg_val<uint32_t>(0);
    const uint32_t m_first = get_arg_val<uint32_t>(1);
    const uint32_t m_count = get_arg_val<uint32_t>(2);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, bytes);
    for (uint32_t m = m_first; m < m_first + m_count; ++m) {
        for (uint32_t c = 0; c < 32; ++c) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(m * Kt + 32 * b + c), bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
