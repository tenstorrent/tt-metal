// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// sparse_sdpa index rows (tt/fuse_attn.py ids_build): per query row (user u, row t) of the chunk
//   in  = [window ids (128 u32) | selected ids (512 u32)]                      (prefill_sparse.attend_dyn: concat([wu,
//   ids[u]], dim=3)) out = in                                       for t >= 128 out[k] = in[perm[t][k]] for t <  128
//   (compact_rows: the sentinels of the first rows of the first chunk go to the tail)
// written to page t of the user's own [1,1,C,640] u32 row-major buffer.  Rows are dealt round-robin over the cores (the
// 128 permuted rows of a user are spread); every core issues all its reads, then computes, then issues all its writes.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_buf = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t NU = get_compile_time_arg_val(2);
    constexpr uint32_t PERM_PER_USER = get_compile_time_arg_val(3);
    constexpr uint32_t MAXR = get_compile_time_arg_val(4);  // rows per core (upper bound)
    constexpr uint32_t WIN = 128, TOPK = 512, W = WIN + TOPK;
    constexpr uint32_t WB = WIN * 4, TB = TOPK * 4, RB = W * 4;
    constexpr auto w_args = TensorAccessorArgs<5>();
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();

    const uint32_t first = get_arg_val<uint32_t>(0);   // first row of this core
    const uint32_t stride = get_arg_val<uint32_t>(1);  // number of cores
    const uint32_t nrows = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto w_acc = TensorAccessor(w_args, w_addr, WB);
    const auto i_acc = TensorAccessor(i_args, i_addr, TB);
    const auto p_acc = TensorAccessor(p_args, p_addr, RB);
    experimental::CB cb(cb_buf);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    // per local row slot j: [row RB | perm RB]
    for (uint32_t j = 0; j < nrows; ++j) {
        const uint32_t g = first + j * stride;  // global row = u * C + t
        const uint32_t u = g / C, t = g - u * C;
        const uint32_t off = j * 2 * RB;
        noc.async_read(w_acc, cb, WB, {.page_id = g, .offset_bytes = 0}, {.offset_bytes = off});
        noc.async_read(i_acc, cb, TB, {.page_id = g, .offset_bytes = 0}, {.offset_bytes = off + WB});
        if (t < WIN) {
            const uint32_t pu = PERM_PER_USER ? u : 0;
            noc.async_read(p_acc, cb, RB, {.page_id = pu * WIN + t, .offset_bytes = 0}, {.offset_bytes = off + RB});
        }
    }
    noc.async_read_barrier();
    for (uint32_t j = 0; j < nrows; ++j) {
        const uint32_t g = first + j * stride;
        const uint32_t u = g / C, t = g - u * C;
        if (t < WIN) {
            volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * 2 * RB);
            volatile tt_l1_ptr uint32_t* pm = in + W;
            // out[k] = in[perm[k]] in place: copy the row to a scratch (the tail of the buffer), then read the scratch
            // through perm
            volatile tt_l1_ptr uint32_t* scratch =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + nrows * 2 * RB);
            for (uint32_t k = 0; k < W; ++k) {
                scratch[k] = in[k];
            }
            for (uint32_t k = 0; k < W; ++k) {
                in[k] = scratch[pm[k]];
            }
        }
    }
    for (uint32_t j = 0; j < nrows; ++j) {
        const uint32_t g = first + j * stride;
        const uint32_t u = g / C, t = g - u * C;
        const uint32_t o_addr = get_common_arg_val<uint32_t>(3 + u);
        const auto o_acc = TensorAccessor(o_args, o_addr, RB);
        noc.async_write(cb, o_acc, RB, {.offset_bytes = j * 2 * RB}, {.page_id = t, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
