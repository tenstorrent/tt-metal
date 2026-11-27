// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for QkvNormRope: publishes the core's q / k / v shards (tensor-backed CBs: reserve + push, no copy),
// builds the unit reduce scaler tile, streams the gamma_q / gamma_k column tiles, and for cos / sin copies this
// user's row (row `user` of the [32, D] position-gathered table) into row 0 of a tile per column (the SFPU row
// broadcast reads row 0 only).
// compile-time args: [cb_q, cb_k, cb_v, cb_gq, cb_gk, cb_cos, cb_sin, cb_scaler, cols_t]
//                    + TensorAccessorArgs(gq, gk, cos, sin)
// runtime args: [gq_addr, gk_addr, cos_addr, sin_addr, user]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t gq_addr = get_arg_val<uint32_t>(0);
    const uint32_t gk_addr = get_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t user = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gk = get_compile_time_arg_val(4);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(6);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(7);
    constexpr uint32_t cols_t = get_compile_time_arg_val(8);
    constexpr auto gq_args = TensorAccessorArgs<9>();
    constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;

    // the shards are already in place: make them visible to compute
    for (uint32_t cb : {cb_q, cb_k, cb_v}) {
        CircularBuffer c(cb);
        c.reserve_back(cols_t);
        c.push_back(cols_t);
    }
    // unit scaler for the row reduce: bf16 1.0 in the first row of every face, zeros elsewhere; the second page
    // of the CB is the staging tile for the cos / sin row copies
    CircularBuffer cs(cb_scaler);
    cs.reserve_back(2);
    const uint32_t scaler_addr = get_write_ptr(cb_scaler);
    const uint32_t stage_addr = scaler_addr + tile_bytes;
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_addr);
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            p[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t j = 0; j < 8; ++j) {
                p[f * 128 + j] = 0x3F803F80u;
            }
        }
    }
    cs.push_back(1);  // the scaler; the staging page is never pushed
    const auto s_gq = TensorAccessor(gq_args, gq_addr, tile_bytes);
    const auto s_gk = TensorAccessor(gk_args, gk_addr, tile_bytes);
    const auto s_cos = TensorAccessor(cos_args, cos_addr, tile_bytes);
    const auto s_sin = TensorAccessor(sin_args, sin_addr, tile_bytes);
    CircularBuffer cgq(cb_gq), cgk(cb_gk), ccos(cb_cos), csin(cb_sin);
    cgq.reserve_back(cols_t);
    cgk.reserve_back(cols_t);
    ccos.reserve_back(cols_t);
    csin.reserve_back(cols_t);
    const uint32_t a_gq = get_write_ptr(cb_gq), a_gk = get_write_ptr(cb_gk);
    const uint32_t a_cos = get_write_ptr(cb_cos), a_sin = get_write_ptr(cb_sin);
    for (uint32_t c = 0; c < cols_t; ++c) {
        noc_async_read(s_gq.get_noc_addr(c), a_gq + c * tile_bytes, tile_bytes);
        noc_async_read(s_gk.get_noc_addr(c), a_gk + c * tile_bytes, tile_bytes);
    }
    // row `user` of the cos / sin tile c -> row 0 of the CB tile c: row u sits in faces (u/16)*2 (+1) at word
    // offset (u%16)*8; row 0 of the CB tile in faces 0 / 1 at word offset 0
    const uint32_t src_face = (user >> 4) * 2;
    const uint32_t src_word = (user & 15) * 8;
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_addr);
    for (uint32_t which = 0; which < 2; ++which) {
        const auto& acc = which == 0 ? s_cos : s_sin;
        const uint32_t dst_base = which == 0 ? a_cos : a_sin;
        for (uint32_t c = 0; c < cols_t; ++c) {
            noc_async_read(acc.get_noc_addr(c), stage_addr, tile_bytes);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_base + c * tile_bytes);
            for (uint32_t j = 0; j < 8; ++j) {
                dst[j] = stage[src_face * 128 + src_word + j];              // columns 0-15
                dst[128 + j] = stage[(src_face + 1) * 128 + src_word + j];  // columns 16-31
            }
        }
    }
    noc_async_read_barrier();
    cgq.push_back(cols_t);
    cgk.push_back(cols_t);
    ccos.push_back(cols_t);
    csin.push_back(cols_t);
}
