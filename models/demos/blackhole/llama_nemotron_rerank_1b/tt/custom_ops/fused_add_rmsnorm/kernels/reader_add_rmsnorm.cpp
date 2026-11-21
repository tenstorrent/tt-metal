// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the fused residual-add + RMSNorm op: resident constants (gamma tiles,
// 1/W scaler, eps) once, then one tile-row of a and of b per iteration with a
// single barrier.
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
    const uint32_t num_rows = get_arg_val<uint32_t>(5);
    const uint32_t row_start = get_arg_val<uint32_t>(6);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto a_args = TensorAccessorArgs<1>();
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
    const uint32_t a_tile = get_tile_size(cb_a);
    const uint32_t b_tile = get_tile_size(cb_b);
    const uint32_t c_tile = get_tile_size(cb_g);

    Noc noc;
    CircularBuffer ca(cb_a), cbb(cb_b);
    {
        CircularBuffer cg(cb_g), csc(cb_sc), ceps(cb_eps);
        cg.reserve_back(Wt);
        for (uint32_t i = 0; i < Wt; ++i) {
            noc.async_read(sg, cg, c_tile, {.page_id = i}, {.offset_bytes = i * c_tile});
        }
        csc.reserve_back(1);
        noc.async_read(ssc, csc, c_tile, {.page_id = 0}, {.offset_bytes = 0});
        ceps.reserve_back(1);
        noc.async_read(seps, ceps, c_tile, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cg.push_back(Wt);
        csc.push_back(1);
        ceps.push_back(1);
    }

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t base = (row_start + r) * Wt;
        ca.reserve_back(Wt);
        cbb.reserve_back(Wt);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc.async_read(sa, ca, a_tile, {.page_id = base + j}, {.offset_bytes = j * a_tile});
        }
        for (uint32_t j = 0; j < Wt; ++j) {
            noc.async_read(sb, cbb, b_tile, {.page_id = base + j}, {.offset_bytes = j * b_tile});
        }
        noc.async_read_barrier();
        ca.push_back(Wt);
        cbb.push_back(Wt);
    }
}
