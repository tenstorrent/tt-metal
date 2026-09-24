// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Row-split add+RMSNorm reader: this core's Wc-tile slice of one row of a and b, plus its gamma slice, scaler, eps.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t sc_addr = get_arg_val<uint32_t>(3);
    const uint32_t eps_addr = get_arg_val<uint32_t>(4);
    const uint32_t row = get_arg_val<uint32_t>(5);
    const uint32_t k = get_arg_val<uint32_t>(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Wc = get_compile_time_arg_val(1);
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto sc_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto eps_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_g = 2, cb_sc = 3, cb_eps = 4;
    const auto sa = TensorAccessor(a_args, a_addr);
    const auto sb = TensorAccessor(b_args, b_addr);
    const auto sg = TensorAccessor(g_args, g_addr);
    const auto ssc = TensorAccessor(sc_args, sc_addr);
    const auto seps = TensorAccessor(eps_args, eps_addr);
    const uint32_t ta = get_tile_size(cb_a), tb = get_tile_size(cb_b), tc = get_tile_size(cb_g);
    Noc noc;
    CircularBuffer ca(cb_a), cbb(cb_b), cg(cb_g), csc(cb_sc), ceps(cb_eps);
    const uint32_t base = row * Wt + k * Wc;
    ca.reserve_back(Wc);
    cbb.reserve_back(Wc);
    cg.reserve_back(Wc);
    csc.reserve_back(1);
    ceps.reserve_back(1);
    for (uint32_t j = 0; j < Wc; ++j) {
        noc.async_read(sa, ca, ta, {.page_id = base + j}, {.offset_bytes = j * ta});
        noc.async_read(sb, cbb, tb, {.page_id = base + j}, {.offset_bytes = j * tb});
        noc.async_read(sg, cg, tc, {.page_id = k * Wc + j}, {.offset_bytes = j * tc});
    }
    noc.async_read(ssc, csc, tc, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(seps, ceps, tc, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cg.push_back(Wc);
    csc.push_back(1);
    ceps.push_back(1);
    ca.push_back(Wc);
    cbb.push_back(Wc);
}
