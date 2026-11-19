// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per position p = r * Ct + c: writes the n output tiles out[r, j, c] (page (r*n + j)*Ct + c), or
// the single collapsed tile out[r, c] (page r*Ct + c). MHC_MAX_EXTRA > 0: then up to MHC_MAX_EXTRA extra
// (position, stream j) units (runtime args [n_extra, p0, j0, ...] after `count`), one output tile each.
#ifndef MHC_MAX_EXTRA
#define MHC_MAX_EXTRA 0
#endif

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_pos = get_arg_val<uint32_t>(1);
    const uint32_t num_pos = get_arg_val<uint32_t>(2);
    constexpr uint32_t MAX_EXTRA = MHC_MAX_EXTRA;
    constexpr uint32_t EXTRA_BASE = 3;

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
    if constexpr (MAX_EXTRA > 0) {
        const uint32_t n_extra = get_arg_val<uint32_t>(EXTRA_BASE);
        for (uint32_t e = 0; e < n_extra && e < MAX_EXTRA; ++e) {
            const uint32_t p = get_arg_val<uint32_t>(EXTRA_BASE + 1 + 2 * e);
            const uint32_t j = get_arg_val<uint32_t>(EXTRA_BASE + 2 + 2 * e);
            const uint32_t r = p / Ct;
            const uint32_t c = p - r * Ct;
            cbo.wait_front(1);
            noc.async_write(cbo, s_out, out_page, {.offset_bytes = 0}, {.page_id = (r * n + j) * Ct + c});
            noc.async_write_barrier();
            cbo.pop_front(1);
        }
    }
    // every NoC transaction acknowledged before the kernel returns
    noc_async_full_barrier();
}
