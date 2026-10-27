// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Scan reader for one (value head bh, V-block vb).  Reads the initial state S [K, Vt] (zeros without
// an initial state) twice: once for the compute and once as h_0 for the writer.  Then per chunk the
// prep blocks: v_beta (its V-slice), and kd, q_decay, intra, k_dec_t, dl, T_inv in full.  All fp32.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_fw/device/kernels/chunk_gated_delta_rule_fw_common.hpp"

using namespace chunk_gated_delta_rule_fw::scan;

constexpr uint32_t Ct = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Vt = get_compile_time_arg_val(2);       // this core's V-block (tiles)
constexpr uint32_t Vt_full = get_compile_time_arg_val(3);  // full V (tiles): the DRAM row stride
constexpr uint32_t HAS_H0 = get_compile_time_arg_val(4);
constexpr uint32_t NC = get_compile_time_arg_val(6);

void kernel_main() {
    constexpr auto vb_args = TensorAccessorArgs<8>();
    constexpr auto kd_args = TensorAccessorArgs<vb_args.next_compile_time_args_offset()>();
    constexpr auto qd_args = TensorAccessorArgs<kd_args.next_compile_time_args_offset()>();
    constexpr auto it_args = TensorAccessorArgs<qd_args.next_compile_time_args_offset()>();
    constexpr auto kc_args = TensorAccessorArgs<it_args.next_compile_time_args_offset()>();
    constexpr auto dl_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto ti_args = TensorAccessorArgs<dl_args.next_compile_time_args_offset()>();
    constexpr auto s0_args = TensorAccessorArgs<ti_args.next_compile_time_args_offset()>();

    const uint32_t bh = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);

    const uint32_t tb = get_tile_size(cb_vbeta);
    const auto vb_acc = TensorAccessor(vb_args, get_arg_val<uint32_t>(2), tb);
    const auto kd_acc = TensorAccessor(kd_args, get_arg_val<uint32_t>(3), tb);
    const auto qd_acc = TensorAccessor(qd_args, get_arg_val<uint32_t>(4), tb);
    const auto it_acc = TensorAccessor(it_args, get_arg_val<uint32_t>(5), tb);
    const auto kc_acc = TensorAccessor(kc_args, get_arg_val<uint32_t>(6), tb);
    const auto dl_acc = TensorAccessor(dl_args, get_arg_val<uint32_t>(7), tb);
    const auto ti_acc = TensorAccessor(ti_args, get_arg_val<uint32_t>(8), tb);
    const auto s0_acc = TensorAccessor(s0_args, get_arg_val<uint32_t>(9), tb);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t kc = Kt * Ct;
    constexpr uint32_t kv = Kt * Vt;

    Noc noc;

    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t first_page, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; ++t) {
            noc.async_read(acc, cb, tb, {.page_id = first_page + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // R tile rows of this core's V-slice (columns [vb*Vt, vb*Vt+Vt)) of a [.., R, V] block starting
    // at tile `row_base`, packed [R, Vt] into the CB.
    auto read_vslice = [&](const auto& acc, uint32_t cb_id, uint32_t row_base, uint32_t R) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(R * Vt);
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t src = row_base + r * Vt_full + vb * Vt;
            for (uint32_t vt = 0; vt < Vt; ++vt) {
                noc.async_read(acc, cb, tb, {.page_id = src + vt}, {.offset_bytes = (r * Vt + vt) * tb});
            }
        }
        noc.async_read_barrier();
        cb.push_back(R * Vt);
    };

    auto push_zeros = [&](uint32_t cb_id) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(kv);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for (uint32_t i = 0; i < kv * (tb / sizeof(uint32_t)); ++i) {
            p[i] = 0U;
        }
        cb.push_back(kv);
    };

    // cb_hout carries h_1.. from the compute, so h_0 goes through its own reader-only CB.
    if constexpr (HAS_H0) {
        read_vslice(s0_acc, cb_S, bh * Kt * Vt_full, Kt);
        read_vslice(s0_acc, cb_h0, bh * Kt * Vt_full, Kt);
    } else {
        push_zeros(cb_S);
        push_zeros(cb_h0);
    }

    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t wi = bh * NC + c;
        read_vslice(vb_acc, cb_vbeta, wi * Ct * Vt_full, Ct);
        read_into(kd_acc, cb_kd, wi * ck, ck);
        read_into(qd_acc, cb_qdecay, wi * ck, ck);
        read_into(it_acc, cb_intra, wi * cc, cc);
        read_into(kc_acc, cb_kdec_t, wi * kc, kc);
        read_into(dl_acc, cb_dl, wi, 1);
        read_into(ti_acc, cb_Tinv, wi * cc, cc);
    }
}
