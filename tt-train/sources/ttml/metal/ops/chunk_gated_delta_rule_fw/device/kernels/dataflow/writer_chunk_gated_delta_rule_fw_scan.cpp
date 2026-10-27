// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Scan writer for one (value head bh = b*HV + hv, V-block vb).  Each output receives this core's
// V-slice, columns [vb*Vt, vb*Vt+Vt) of V.  Per chunk c, in the compute's push order:
//   h      [BH, NC, K, V]   state entering chunk c (h_0 from the reader, h_{c>=1} from the compute)
//   v_new  [BH, 1, T, V]
//   o      [B, 1, T, HV*V]  token-major, value head hv at column block hv (or head-major [BH, 1, T, V])
// and once at the end final_state [B, HV, K, V].  All fp32.

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
constexpr uint32_t Vt_full = get_compile_time_arg_val(3);  // full V (tiles)
constexpr uint32_t HV = get_compile_time_arg_val(5);
constexpr uint32_t NC = get_compile_time_arg_val(6);
constexpr uint32_t O_HEAD_MAJOR = get_compile_time_arg_val(7);  // o is [BH, 1, T, V]

constexpr uint32_t Tt = NC * Ct;

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<8>();
    constexpr auto fs_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<fs_args.next_compile_time_args_offset()>();
    constexpr auto vn_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();

    const uint32_t bh = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);

    const uint32_t tb = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_args, get_arg_val<uint32_t>(2), tb);
    const auto fs_acc = TensorAccessor(fs_args, get_arg_val<uint32_t>(3), tb);
    const auto h_acc = TensorAccessor(h_args, get_arg_val<uint32_t>(4), tb);
    const auto vn_acc = TensorAccessor(vn_args, get_arg_val<uint32_t>(5), tb);

    const uint32_t b = bh / HV;
    const uint32_t hv = bh % HV;

    Noc noc;

    // R tile rows of `cb_id` ([R, Vt]) into this core's V-slice of rows starting at tile `row_base`,
    // `row_stride` tiles apart.
    auto scatter_vslice = [&](uint32_t cb_id, const auto& acc, uint32_t row_base, uint32_t row_stride, uint32_t R) {
        CircularBuffer cb(cb_id);
        cb.wait_front(R * Vt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t dst = row_base + r * row_stride + vb * Vt;
            for (uint32_t vt = 0; vt < Vt; ++vt) {
                noc.async_write(src, acc, tb, {.offset_bytes = (r * Vt + vt) * tb}, {.page_id = dst + vt});
            }
        }
        noc.async_write_barrier();
        cb.pop_front(R * Vt);
    };

    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t wi = bh * NC + c;
        scatter_vslice(c == 0 ? cb_h0 : cb_hout, h_acc, wi * Kt * Vt_full, Vt_full, Kt);
        scatter_vslice(cb_vnout, vn_acc, wi * Ct * Vt_full, Vt_full, Ct);
        if constexpr (O_HEAD_MAJOR) {
            scatter_vslice(cb_out, o_acc, wi * Ct * Vt_full, Vt_full, Ct);
        } else {
            scatter_vslice(cb_out, o_acc, (b * Tt + c * Ct) * HV * Vt_full + hv * Vt_full, HV * Vt_full, Ct);
        }
    }
    scatter_vslice(cb_final, fs_acc, bh * Kt * Vt_full, Vt_full, Kt);
}
