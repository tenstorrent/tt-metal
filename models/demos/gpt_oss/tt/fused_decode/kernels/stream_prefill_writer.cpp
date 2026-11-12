// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill matmul on a decode stream weight, writer (BRISC, NOC1) (tt/stream_prefill_linear.py StreamPrefillLinear).
// First the zero output columns [z0, z1) of this core's rows (output width > N: the LM head's power-of-two padding),
// from one zeroed page (all-zero bytes are 0 in BF16 and BFP8_b); then the compute kernel's tiles in its order (M block
// -> weight group -> column -> DST block of rows -> row) to tile (row, col[c]) of the [.., M, NT * 32] output, col[c]
// the logical output column of the core's slot c (the stream's slot order undone).
//
// common runtime args: [out_addr]
// runtime args: [m0, m_cnt, cols, z0, z1, col[0], ..., col[cols - 1]]
// compile-time args: [cb_out, cb_zero, NT, MB, G, accessor (out)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t COL_ARG = 5;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t m_cnt = get_arg_val<uint32_t>(1);
    const uint32_t cols = get_arg_val<uint32_t>(2);
    const uint32_t z0 = get_arg_val<uint32_t>(3);
    const uint32_t z1 = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t MB = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr uint32_t RT = 4;  // stream_prefill_compute.cpp
    const uint32_t tile = get_tile_size(cb_out);
    const auto s_out = TensorAccessor(out_args, out_addr, tile);

    if (z1 > z0) {
        cb_reserve_back(cb_zero, 1);  // (this kernel's own page: never pushed)
        const uint32_t zl1 = get_write_ptr(cb_zero);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zl1);
        for (uint32_t i = 0; i < tile / 4; ++i) {
            z[i] = 0;
        }
        for (uint32_t r = 0; r < m_cnt; ++r) {
            for (uint32_t c = z0; c < z1; ++c) {
                noc_async_write(zl1, s_out.get_noc_addr((m0 + r) * NT + c), tile);
            }
            noc_async_writes_flushed();
        }
    }

    const uint32_t n_groups = (cols + G - 1) / G;
    for (uint32_t mb = 0; mb < m_cnt; mb += MB) {
        const uint32_t rows = m_cnt - mb < MB ? m_cnt - mb : MB;
        for (uint32_t g = 0; g < n_groups; ++g) {
            const uint32_t gc = cols - g * G < G ? cols - g * G : G;
            for (uint32_t c = 0; c < gc; ++c) {
                const uint32_t col = get_arg_val<uint32_t>(COL_ARG + g * G + c);
                for (uint32_t r = 0; r < rows; ++r) {  // (the DST blocks of rows are in row order)
                    cb_wait_front(cb_out, 1);
                    noc_async_write(get_read_ptr(cb_out), s_out.get_noc_addr((m0 + mb + r) * NT + col), tile);
                    noc_async_writes_flushed();
                    cb_pop_front(cb_out, 1);
                }
            }
        }
    }
    noc_async_write_barrier();
}
