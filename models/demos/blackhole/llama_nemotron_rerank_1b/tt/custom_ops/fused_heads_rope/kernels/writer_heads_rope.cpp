// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the fused head-split + RoPE op: per work unit (b, s, g) the q_per_kv
// rotated Q heads go to Q [B, n_heads, S, D], the rotated K head and the V head to
// K / V [B, n_kv, S, D], each at seq tile s.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t n_heads = get_compile_time_arg_val(0);
    constexpr uint32_t n_kv = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr uint32_t q_per_kv = n_heads / n_kv;
    constexpr uint32_t unit_tiles = (q_per_kv + 2) * Wt;
    constexpr uint32_t cb_out = 16;

    const auto sq = TensorAccessor(q_args, q_addr);
    const auto sk = TensorAccessor(k_args, k_addr);
    const auto sv = TensorAccessor(v_args, v_addr);
    const uint32_t t = get_tile_size(cb_out);

    Noc noc;
    CircularBuffer co(cb_out);
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t u = unit_start + w;
        const uint32_t g = u % n_kv;
        const uint32_t s = (u / n_kv) % St;
        const uint32_t b = u / (n_kv * St);
        co.wait_front(unit_tiles);
        uint32_t off = 0;
        for (uint32_t hh = 0; hh < q_per_kv; ++hh) {
            const uint32_t base = ((b * n_heads + g * q_per_kv + hh) * St + s) * Wt;
            for (uint32_t j = 0; j < Wt; ++j, ++off) {
                noc.async_write(co, sq, t, {.offset_bytes = off * t}, {.page_id = base + j});
            }
        }
        const uint32_t kv_base = ((b * n_kv + g) * St + s) * Wt;
        for (uint32_t j = 0; j < Wt; ++j, ++off) {
            noc.async_write(co, sk, t, {.offset_bytes = off * t}, {.page_id = kv_base + j});
        }
        for (uint32_t j = 0; j < Wt; ++j, ++off) {
            noc.async_write(co, sv, t, {.offset_bytes = off * t}, {.page_id = kv_base + j});
        }
        noc.async_write_barrier();
        co.pop_front(unit_tiles);
    }
}
