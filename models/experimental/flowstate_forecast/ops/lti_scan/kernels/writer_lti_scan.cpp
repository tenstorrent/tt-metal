// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes x (re, im) of every unit for every step to the (L, B, P) TILE outputs.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_re_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_im_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);
    const uint32_t first_unit = get_arg_val<uint32_t>(3);
    const uint32_t T = get_arg_val<uint32_t>(4);
    const uint32_t Bt = get_arg_val<uint32_t>(5);
    const uint32_t Pt = get_arg_val<uint32_t>(6);

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;

    Noc noc;
    CircularBuffer cb_out(out_cb_id);
    const uint32_t bytes = get_tile_size(out_cb_id);
    const auto x_re = TensorAccessor(x_args, x_re_addr);
    const auto x_im = TensorAccessor(x_args, x_im_addr);

    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t u = 0; u < num_units; ++u) {
            const uint32_t unit = first_unit + u;
            const uint32_t page = (t * Bt + unit / Pt) * Pt + unit % Pt;
            cb_out.wait_front(2);
            noc.async_write(cb_out, x_re, bytes, {.offset_bytes = 0}, {.page_id = page});
            noc.async_write(cb_out, x_im, bytes, {.offset_bytes = bytes}, {.page_id = page});
            noc.async_writes_flushed();
            cb_out.pop_front(2);
        }
    }
    noc.async_write_barrier();
}
