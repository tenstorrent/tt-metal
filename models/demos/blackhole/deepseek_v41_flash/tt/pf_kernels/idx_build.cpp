// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill sparse_sdpa index rows (cores-for-prefill, tt/pf_retile.py build_idx): row r of one user = [window row (WIN
// uint32) | selected ids row (TOPK uint32)], and for the first PERM_ROWS rows the valid window entries are moved in
// front of the selected ids by a per-row permutation (out[k] = in[perm[r][k]]): what slice + concat + slice + gather +
// slice + concat did in 8 ops. Rows are interleaved over the cores (core i: rows i, i + stride, ...). Everything is
// uint32 row-major, one page per row.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t WIN = get_compile_time_arg_val(1);
    constexpr uint32_t TOPK = get_compile_time_arg_val(2);
    constexpr uint32_t PERM_ROWS = get_compile_time_arg_val(3);
    constexpr auto w_args = TensorAccessorArgs<4>();
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t K = WIN + TOPK;
    constexpr uint32_t ROWB = K * 4;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t stride = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t w_base = get_arg_val<uint32_t>(3);  // page of row 0 of the user in the window tensor
    const uint32_t i_base = get_arg_val<uint32_t>(4);  // ... in the ids tensor
    const uint32_t p_base = get_arg_val<uint32_t>(5);  // ... in the perm tensor
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(3);

    Noc noc;
    const auto w_acc = TensorAccessor(w_args, w_addr, WIN * 4);
    const auto i_acc = TensorAccessor(i_args, i_addr, TOPK * 4);
    const auto p_acc = TensorAccessor(p_args, p_addr, ROWB);
    const auto o_acc = TensorAccessor(o_args, o_addr, ROWB);
    experimental::CB c(cb);
    c.reserve_back(1);
    const uint32_t base = c.get_write_ptr();
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);         // [K] window | ids
    volatile tt_l1_ptr uint32_t* pm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + ROWB);  // [K] perm row
    volatile tt_l1_ptr uint32_t* ou = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 2 * ROWB);  // [K] result

    for (uint32_t n = 0; n < count; ++n) {
        const uint32_t r = first + n * stride;
        noc.async_read(w_acc, c, WIN * 4, {.page_id = w_base + r, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(i_acc, c, TOPK * 4, {.page_id = i_base + r, .offset_bytes = 0}, {.offset_bytes = WIN * 4});
        const bool permute = r < PERM_ROWS;
        if (permute) {
            noc.async_read(p_acc, c, ROWB, {.page_id = p_base + r, .offset_bytes = 0}, {.offset_bytes = ROWB});
        }
        noc.async_read_barrier();
        uint32_t src_off = 0;
        if (permute) {
            for (uint32_t k = 0; k < K; ++k) {
                ou[k] = in[pm[k]];
            }
            src_off = 2 * ROWB;
        }
        noc.async_write(c, o_acc, ROWB, {.offset_bytes = src_off}, {.page_id = r, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
