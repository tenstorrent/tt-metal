// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the fused head-split + RoPE op. The rotation tile once, then per work
// unit (batch b, seq tile s, KV group g) the group's tiles of one fused-QKV tile-row:
// its q_per_kv Q heads, its K head and its V head (Wt tiles each), and the cos / sin
// tiles of seq tile s.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t trans_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_units = get_arg_val<uint32_t>(4);
    const uint32_t unit_start = get_arg_val<uint32_t>(5);

    constexpr uint32_t n_heads = get_compile_time_arg_val(0);
    constexpr uint32_t n_kv = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr auto qkv_args = TensorAccessorArgs<4>();
    constexpr auto cos_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    constexpr uint32_t q_per_kv = n_heads / n_kv;
    constexpr uint32_t row_tiles = (n_heads + 2 * n_kv) * Wt;
    constexpr uint32_t unit_tiles = (q_per_kv + 2) * Wt;
    constexpr uint32_t cb_in = 0, cb_cos = 9, cb_sin = 10, cb_trans = 11;

    const auto s_qkv = TensorAccessor(qkv_args, qkv_addr);
    const auto s_cos = TensorAccessor(cos_args, cos_addr);
    const auto s_sin = TensorAccessor(sin_args, sin_addr);
    const auto s_trans = TensorAccessor(trans_args, trans_addr);
    const uint32_t in_tile = get_tile_size(cb_in);
    const uint32_t rot_tile = get_tile_size(cb_cos);

    Noc noc;
    CircularBuffer cin(cb_in), ccos(cb_cos), csin(cb_sin);
    {
        CircularBuffer ctr(cb_trans);
        ctr.reserve_back(1);
        noc.async_read(s_trans, ctr, rot_tile, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        ctr.push_back(1);
    }
    // cos / sin are read once per seq tile, not per unit: consecutive units (g fastest) share s.
    uint32_t last_s = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t u = unit_start + w;
        const uint32_t g = u % n_kv;
        const uint32_t s = (u / n_kv) % St;
        const uint32_t b = u / (n_kv * St);
        const uint32_t row = (b * St + s) * row_tiles;
        const bool new_s = s != last_s;
        cin.reserve_back(unit_tiles);
        if (new_s) {
            ccos.reserve_back(Wt);
            csin.reserve_back(Wt);
        }
        uint32_t off = 0;
        // Q heads g*q_per_kv .. +q_per_kv are contiguous in the fused row.
        for (uint32_t j = 0; j < q_per_kv * Wt; ++j, ++off) {
            noc.async_read(
                s_qkv, cin, in_tile, {.page_id = row + g * q_per_kv * Wt + j}, {.offset_bytes = off * in_tile});
        }
        for (uint32_t j = 0; j < Wt; ++j, ++off) {
            noc.async_read(
                s_qkv, cin, in_tile, {.page_id = row + (n_heads + g) * Wt + j}, {.offset_bytes = off * in_tile});
        }
        for (uint32_t j = 0; j < Wt; ++j, ++off) {
            noc.async_read(
                s_qkv, cin, in_tile, {.page_id = row + (n_heads + n_kv + g) * Wt + j}, {.offset_bytes = off * in_tile});
        }
        if (new_s) {
            for (uint32_t j = 0; j < Wt; ++j) {
                noc.async_read(s_cos, ccos, rot_tile, {.page_id = s * Wt + j}, {.offset_bytes = j * rot_tile});
                noc.async_read(s_sin, csin, rot_tile, {.page_id = s * Wt + j}, {.offset_bytes = j * rot_tile});
            }
        }
        noc.async_read_barrier();
        cin.push_back(unit_tiles);
        if (new_s) {
            ccos.push_back(Wt);
            csin.push_back(Wt);
            last_s = s;
        }
    }
}
