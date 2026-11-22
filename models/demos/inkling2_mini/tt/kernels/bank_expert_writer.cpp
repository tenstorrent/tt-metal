// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0

// Bank-local indexed expert matmul, writer: slot s, this worker's columns -> out tiles s*Nt + n0 + n.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t NW = get_compile_time_arg_val(0);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const uint32_t n_valid = get_arg_val<uint32_t>(2);
    if (n_valid == 0) {
        return;  // idle core
    }
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);

    for (uint32_t s = 0; s < SLOTS; ++s) {
        cb_wait_front(cb_out, NW);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t n = 0; n < n_valid; ++n) {
            noc_async_write_page(s * Nt + n0 + n, out, l1);
            l1 += tile;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, NW);
    }
}
