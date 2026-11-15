// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the prefill qkv_norm_rope op: per work unit the COLS_T output tiles of CB_OUT go to the unit's row in
// the output tensor of its kind (same tile ids as the input row).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cols_t = get_compile_time_arg_val(1);
    constexpr auto q_args = TensorAccessorArgs<2>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_units = get_arg_val<uint32_t>(3);
    constexpr uint32_t unit_args = 4;  // per unit: kind, first tile id of the row

    const auto s_q = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto s_k = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto s_v = TensorAccessor(v_args, v_addr, tile_bytes);
    CircularBuffer cout(cb_out);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t kind = get_arg_val<uint32_t>(unit_args + 2 * u);
        const uint32_t tile0 = get_arg_val<uint32_t>(unit_args + 2 * u + 1);
        cout.wait_front(cols_t);
        const uint32_t a_out = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < cols_t; ++c) {
            const uint64_t dst = kind == 0   ? s_q.get_noc_addr(tile0 + c)
                                 : kind == 1 ? s_k.get_noc_addr(tile0 + c)
                                             : s_v.get_noc_addr(tile0 + c);
            noc_async_write(a_out + c * tile_bytes, dst, tile_bytes);
        }
        noc_async_write_barrier();
        cout.pop_front(cols_t);
    }
}
