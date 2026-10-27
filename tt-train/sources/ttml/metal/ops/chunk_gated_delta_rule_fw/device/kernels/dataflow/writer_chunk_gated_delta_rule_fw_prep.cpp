// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Prep writer.  For each (value head, chunk) work-item wi = bh * NC + c, drains the fp32 per-chunk
// blocks to DRAM at tile offset wi * (tiles per chunk):
//   v_beta [C,V], A = T_inv [C,C], kd [C,K], intra [C,C], q_decay [C,K], k_dec_t [K,C], dl [1 tile],
//   g_cumsum [C,1].
// A and g_cumsum are the saved intermediates; the rest feed the scan.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_fw/device/kernels/chunk_gated_delta_rule_fw_common.hpp"

using namespace chunk_gated_delta_rule_fw::prep;

constexpr uint32_t Ct = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Vt = get_compile_time_arg_val(2);

void kernel_main() {
    constexpr auto vb_args = TensorAccessorArgs<3>();
    constexpr auto kd_args = TensorAccessorArgs<vb_args.next_compile_time_args_offset()>();
    constexpr auto qd_args = TensorAccessorArgs<kd_args.next_compile_time_args_offset()>();
    constexpr auto it_args = TensorAccessorArgs<qd_args.next_compile_time_args_offset()>();
    constexpr auto kc_args = TensorAccessorArgs<it_args.next_compile_time_args_offset()>();
    constexpr auto dl_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<dl_args.next_compile_time_args_offset()>();
    constexpr auto gc_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);

    const uint32_t tb = get_tile_size(cb_vbeta);
    const auto vb_acc = TensorAccessor(vb_args, get_arg_val<uint32_t>(2), tb);
    const auto kd_acc = TensorAccessor(kd_args, get_arg_val<uint32_t>(3), tb);
    const auto qd_acc = TensorAccessor(qd_args, get_arg_val<uint32_t>(4), tb);
    const auto it_acc = TensorAccessor(it_args, get_arg_val<uint32_t>(5), tb);
    const auto kc_acc = TensorAccessor(kc_args, get_arg_val<uint32_t>(6), tb);
    const auto dl_acc = TensorAccessor(dl_args, get_arg_val<uint32_t>(7), tb);
    const auto a_acc = TensorAccessor(a_args, get_arg_val<uint32_t>(8), tb);
    const auto gc_acc = TensorAccessor(gc_args, get_arg_val<uint32_t>(9), tb);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kc = Kt * Ct;

    Noc noc;

    auto drain = [&](uint32_t cb_id, const auto& acc, uint32_t n, uint32_t first_page) {
        CircularBuffer cb(cb_id);
        cb.wait_front(n);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < n; ++t) {
            noc.async_write(src, acc, tb, {.offset_bytes = t * tb}, {.page_id = first_page + t});
        }
        noc.async_write_barrier();
        cb.pop_front(n);
    };

    // Roughly the compute's push order.
    for (uint32_t i = 0; i < wi_count; ++i) {
        const uint32_t wi = wi_start + i;
        drain(cb_vbeta, vb_acc, cv, wi * cv);
        drain(cb_Tinv, a_acc, cc, wi * cc);
        drain(cb_kd, kd_acc, ck, wi * ck);
        drain(cb_intra, it_acc, cc, wi * cc);
        drain(cb_qdecay, qd_acc, ck, wi * ck);
        drain(cb_kdec_t, kc_acc, kc, wi * kc);
        drain(cb_dl, dl_acc, 1, wi);
        drain(cb_gcum, gc_acc, Ct, wi * Ct);
    }
}
