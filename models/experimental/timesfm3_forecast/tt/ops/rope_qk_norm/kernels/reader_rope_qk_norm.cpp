// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the fused RoPE + QK RMSNorm. Units run seq-tile-major (u = st * G * RH + bh, bh over the q heads, the
// k heads and, from a fused QKV tensor, the v heads), so a core re-reads cos / sin only when its seq tile changes.
// With from_qkv (R3-5), q / k / v head tiles are read straight from the (R * S, 3 * H * Wt) QKV matmul output (q and
// k accessors both point at it) and the v heads go to a pass-through buffer for the writer.
// Compile-time args: Wt, St, RH (rows * heads of one tensor), H, from_qkv, accessor args of q, k, cos, sin, trans, gq, gk.
// Runtime: q, k, cos, sin, trans, gq, gk addresses, num_units, start_unit.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

template <typename Accessor>
FORCE_INLINE void read_tiles(Noc& noc, CircularBuffer& cb, const Accessor& src, uint32_t bytes, uint32_t first, uint32_t n) {
    cb.reserve_back(n);
    for (uint32_t j = 0; j < n; ++j) {
        noc.async_read(src, cb, bytes, {.page_id = first + j}, {.offset_bytes = j * bytes});
    }
    noc.async_read_barrier();
    cb.push_back(n);
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t trans_addr = get_arg_val<uint32_t>(4);
    const uint32_t gq_addr = get_arg_val<uint32_t>(5);
    const uint32_t gk_addr = get_arg_val<uint32_t>(6);
    const uint32_t num_units = get_arg_val<uint32_t>(7);
    const uint32_t start_unit = get_arg_val<uint32_t>(8);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t RH = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr bool from_qkv = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t G = from_qkv ? 3 : 2;
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto gq_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();

    Noc noc;
    CircularBuffer cb_x(tt::CBIndex::c_0);
    CircularBuffer cb_cos(tt::CBIndex::c_1);
    CircularBuffer cb_sin(tt::CBIndex::c_2);
    CircularBuffer cb_trans(tt::CBIndex::c_3);
    CircularBuffer cb_scaler(tt::CBIndex::c_4);
    CircularBuffer cb_g(tt::CBIndex::c_5);
    CircularBuffer cb_v(tt::CBIndex::c_6);
    const uint32_t bytes = get_tile_size(tt::CBIndex::c_0);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto cos = TensorAccessor(c_args, cos_addr);
    const auto sin = TensorAccessor(s_args, sin_addr);
    const auto trans = TensorAccessor(t_args, trans_addr);
    const auto gq = TensorAccessor(gq_args, gq_addr);
    const auto gk = TensorAccessor(gk_args, gk_addr);

    cb_scaler.reserve_back(1);
    volatile tt_l1_ptr uint32_t* scaler = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_scaler.get_write_ptr());
    for (uint32_t i = 0; i < 512; ++i) {
        scaler[i] = 0;
    }
    constexpr uint32_t one_one_bf16 = 0x3F803F80;
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t i = 0; i < 8; ++i) {
            scaler[face * 128 + i] = one_one_bf16;
        }
    }
    cb_scaler.push_back(1);
    read_tiles(noc, cb_trans, trans, bytes, 0, 1);
    cb_g.reserve_back(2 * Wt);
    for (uint32_t j = 0; j < Wt; ++j) {
        noc.async_read(gq, cb_g, bytes, {.page_id = j}, {.offset_bytes = j * bytes});
        noc.async_read(gk, cb_g, bytes, {.page_id = j}, {.offset_bytes = (Wt + j) * bytes});
    }
    noc.async_read_barrier();
    cb_g.push_back(2 * Wt);

    uint32_t cur_st = 0xFFFFFFFF;
    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t st = u / (G * RH);
        const uint32_t bh = u % (G * RH);
        const uint32_t part = bh / RH;
        if (from_qkv) {
            const uint32_t rh = bh % RH;
            const uint32_t page = ((rh / H) * St + st) * (3 * H * Wt) + (part * H + rh % H) * Wt;
            if (part == 2) {
                read_tiles(noc, cb_v, q, bytes, page, Wt);
                continue;
            }
            if (st != cur_st) {
                read_tiles(noc, cb_cos, cos, bytes, st * Wt, Wt);
                read_tiles(noc, cb_sin, sin, bytes, st * Wt, Wt);
                cur_st = st;
            }
            read_tiles(noc, cb_x, q, bytes, page, Wt);
            continue;
        }
        if (st != cur_st) {
            read_tiles(noc, cb_cos, cos, bytes, st * Wt, Wt);
            read_tiles(noc, cb_sin, sin, bytes, st * Wt, Wt);
            cur_st = st;
        }
        const uint32_t page = ((bh % RH) * St + st) * Wt;
        if (bh < RH) {
            read_tiles(noc, cb_x, q, bytes, page, Wt);
        } else {
            read_tiles(noc, cb_x, k, bytes, page, Wt);
        }
    }
}
