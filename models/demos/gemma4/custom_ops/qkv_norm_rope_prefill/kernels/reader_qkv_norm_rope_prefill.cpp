// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the prefill qkv_norm_rope op: the q / k gamma tiles once (row 0 valid, kept for the whole program),
// the unit scaler tile, then per work unit the [32, head_dim] tile row of the head tensor (COLS_T tiles) into
// CB_X and, for q / k units, the cos / sin tile rows of the same positions into CB_COS / CB_SIN.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gq = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gk = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cols_t = get_compile_time_arg_val(6);
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gq_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t gq_addr = get_arg_val<uint32_t>(3);
    const uint32_t gk_addr = get_arg_val<uint32_t>(4);
    const uint32_t cos_addr = get_arg_val<uint32_t>(5);
    const uint32_t sin_addr = get_arg_val<uint32_t>(6);
    const uint32_t n_units = get_arg_val<uint32_t>(7);
    // per unit: kind (0 q, 1 k, 2 v), first tile id of the row in its tensor, first tile id of the cos / sin row
    constexpr uint32_t unit_args = 8;

    const auto s_q = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto s_k = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto s_v = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto s_gq = TensorAccessor(gq_args, gq_addr, tile_bytes);
    const auto s_gk = TensorAccessor(gk_args, gk_addr, tile_bytes);
    const auto s_cos = TensorAccessor(cos_args, cos_addr, tile_bytes);
    const auto s_sin = TensorAccessor(sin_args, sin_addr, tile_bytes);

    // unit scaler for the row reduce: bf16 1.0 in the first row of every face, zeros elsewhere
    CircularBuffer cs(cb_scaler);
    cs.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            p[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t j = 0; j < 8; ++j) {
                p[f * 128 + j] = 0x3F803F80u;
            }
        }
    }
    cs.push_back(1);
    // the gamma rows (tile c of the [1, 1, 32, D] row-0 tile row), kept for the whole program
    CircularBuffer cgq(cb_gq), cgk(cb_gk);
    cgq.reserve_back(cols_t);
    cgk.reserve_back(cols_t);
    const uint32_t a_gq = get_write_ptr(cb_gq), a_gk = get_write_ptr(cb_gk);
    for (uint32_t c = 0; c < cols_t; ++c) {
        noc_async_read(s_gq.get_noc_addr(c), a_gq + c * tile_bytes, tile_bytes);
        noc_async_read(s_gk.get_noc_addr(c), a_gk + c * tile_bytes, tile_bytes);
    }
    noc_async_read_barrier();
    cgq.push_back(cols_t);
    cgk.push_back(cols_t);

    CircularBuffer cx(cb_x), ccos(cb_cos), csin(cb_sin);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t kind = get_arg_val<uint32_t>(unit_args + 3 * u);
        const uint32_t tile0 = get_arg_val<uint32_t>(unit_args + 3 * u + 1);
        const uint32_t rope0 = get_arg_val<uint32_t>(unit_args + 3 * u + 2);
        cx.reserve_back(cols_t);
        const uint32_t a_x = get_write_ptr(cb_x);
        for (uint32_t c = 0; c < cols_t; ++c) {
            const uint64_t src = kind == 0   ? s_q.get_noc_addr(tile0 + c)
                                 : kind == 1 ? s_k.get_noc_addr(tile0 + c)
                                             : s_v.get_noc_addr(tile0 + c);
            noc_async_read(src, a_x + c * tile_bytes, tile_bytes);
        }
        if (kind != 2) {
            ccos.reserve_back(cols_t);
            csin.reserve_back(cols_t);
            const uint32_t a_cos = get_write_ptr(cb_cos), a_sin = get_write_ptr(cb_sin);
            for (uint32_t c = 0; c < cols_t; ++c) {
                noc_async_read(s_cos.get_noc_addr(rope0 + c), a_cos + c * tile_bytes, tile_bytes);
                noc_async_read(s_sin.get_noc_addr(rope0 + c), a_sin + c * tile_bytes, tile_bytes);
            }
        }
        noc_async_read_barrier();
        cx.push_back(cols_t);
        if (kind != 2) {
            ccos.push_back(cols_t);
            csin.push_back(cols_t);
        }
    }
}
