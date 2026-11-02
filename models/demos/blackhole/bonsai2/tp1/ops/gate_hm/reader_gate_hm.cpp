// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for gate_hm (GDN decode gated norm + SiLU(z) gate). One core per value head h: its Dt tiles of o
// ([heads, 1, D] fp32, the recurrence's head-major output), of z and of the norm weight (both bf16 rows whose
// head h channels are tiles h*Dt .. h*Dt + Dt-1, row 0).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_o = 0, cb_z = 1, cb_w = 2;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    constexpr auto z_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t z_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t h = get_arg_val<uint32_t>(3);

    const auto o = TensorAccessor(o_a, o_addr, get_tile_size(cb_o));
    const auto z = TensorAccessor(z_a, z_addr, get_tile_size(cb_z));
    const auto w = TensorAccessor(w_a, w_addr, get_tile_size(cb_w));
    Noc noc;
    CircularBuffer co(cb_o), cz(cb_z), cw(cb_w);
    co.reserve_back(Dt);
    cz.reserve_back(Dt);
    cw.reserve_back(Dt);
    for (uint32_t t = 0; t < Dt; t++) {
        noc.async_read(o, co, get_tile_size(cb_o), {.page_id = h * Dt + t}, {.offset_bytes = t * get_tile_size(cb_o)});
        noc.async_read(z, cz, get_tile_size(cb_z), {.page_id = h * Dt + t}, {.offset_bytes = t * get_tile_size(cb_z)});
        noc.async_read(w, cw, get_tile_size(cb_w), {.page_id = t}, {.offset_bytes = t * get_tile_size(cb_w)});
    }
    noc.async_read_barrier();
    co.push_back(Dt);
    cz.push_back(Dt);
    cw.push_back(Dt);
}
