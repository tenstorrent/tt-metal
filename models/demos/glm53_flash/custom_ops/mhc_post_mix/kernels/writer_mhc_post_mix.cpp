// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per position p = r * Ct + c: writes the n output tiles out[r, j, c] (page (r*n + j)*Ct + c), or
// the single collapsed tile out[r, c] (page r*Ct + c).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_pos = get_arg_val<uint32_t>(1);
    const uint32_t num_pos = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr bool collapse = get_compile_time_arg_val(3) != 0;
    constexpr auto out_args = TensorAccessorArgs<4>();
    constexpr uint32_t tiles_per_pos = collapse ? 1 : n;

    const uint32_t out_page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cbo(cb_out);

    for (uint32_t p = start_pos; p < start_pos + num_pos; ++p) {
        const uint32_t r = p / Ct;
        const uint32_t c = p - r * Ct;
        cbo.wait_front(tiles_per_pos);
        for (uint32_t j = 0; j < tiles_per_pos; ++j) {
            const uint32_t page = collapse ? (r * Ct + c) : ((r * n + j) * Ct + c);
            noc.async_write(cbo, s_out, out_page, {.offset_bytes = j * out_page}, {.page_id = page});
        }
        noc.async_write_barrier();
        cbo.pop_front(tiles_per_pos);
    }
}
