// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads the unit's Lbar tiles once, then Bu (re, im) of every unit for every step, in the compute order.
// Tile page of (t, bt, pt) in an (L, B, P) TILE tensor: (t * Bt + bt) * Pt + pt; Lbar (32, P): page pt.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t bu_re_addr = get_arg_val<uint32_t>(0);
    const uint32_t bu_im_addr = get_arg_val<uint32_t>(1);
    const uint32_t l_re_addr = get_arg_val<uint32_t>(2);
    const uint32_t l_im_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_units = get_arg_val<uint32_t>(4);
    const uint32_t first_unit = get_arg_val<uint32_t>(5);
    const uint32_t T = get_arg_val<uint32_t>(6);
    const uint32_t Bt = get_arg_val<uint32_t>(7);
    const uint32_t Pt = get_arg_val<uint32_t>(8);
    const uint32_t row_tiles = get_arg_val<uint32_t>(9);  // tiles per (t, bt) row: Pt split, 2 Pt fused [re | im]
    const uint32_t im_off = get_arg_val<uint32_t>(10);    // 0 split (separate tensors), Pt fused

    constexpr auto bu_args = TensorAccessorArgs<0>();
    constexpr auto l_args = TensorAccessorArgs<bu_args.next_compile_time_args_offset()>();
    constexpr uint32_t in_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t lbar_cb_id = tt::CBIndex::c_1;

    Noc noc;
    CircularBuffer cb_in(in_cb_id), cb_lbar(lbar_cb_id);
    const uint32_t bytes = get_tile_size(in_cb_id);
    const auto bu_re = TensorAccessor(bu_args, bu_re_addr);
    const auto bu_im = TensorAccessor(bu_args, bu_im_addr);
    const auto l_re = TensorAccessor(l_args, l_re_addr);
    const auto l_im = TensorAccessor(l_args, l_im_addr);

    cb_lbar.reserve_back(2 * num_units);
    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t pt = (first_unit + u) % Pt;
        noc.async_read(l_re, cb_lbar, bytes, {.page_id = pt}, {.offset_bytes = (2 * u) * bytes});
        noc.async_read(l_im, cb_lbar, bytes, {.page_id = pt}, {.offset_bytes = (2 * u + 1) * bytes});
    }
    noc.async_read_barrier();
    cb_lbar.push_back(2 * num_units);

    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t u = 0; u < num_units; ++u) {
            const uint32_t unit = first_unit + u;
            const uint32_t page = (t * Bt + unit / Pt) * row_tiles + unit % Pt;
            cb_in.reserve_back(2);
            noc.async_read(bu_re, cb_in, bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read(bu_im, cb_in, bytes, {.page_id = page + im_off}, {.offset_bytes = bytes});
            noc.async_read_barrier();
            cb_in.push_back(2);
        }
    }
}
