// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 integ: natural-order cache -> this column's slices of the sequence-sharded pool (512-token pages, column c owns tokens [64c, 64c+64) of every page).
// Pages (rows of PAGE bytes, or tile pages of PAGE bytes) are copied in slices of U consecutive pages:
//   src page (p * 8U + c * U + u)  ->  dst page (p * U + u)        p = pool page (512 tokens), u in [0, U)
// ROW mode: PAGE = 704 (SCALED_FP8 rows) / 1152 (bf16), U = 64 rows;  TILE mode (bfp8 / bf16 key tiles [T,128]: 4 tile pages per 32 rows): U = 8 tile pages.
// col id of the device: element 0 of a uint32 page (sharded colid tensor). Work: pool pages p = core, core + NCORES, ... ; npages = common arg 3.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t U = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t NCORES = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto col_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t a_src = get_common_arg_val<uint32_t>(0), a_dst = get_common_arg_val<uint32_t>(1),
                   a_col = get_common_arg_val<uint32_t>(2), npages = get_common_arg_val<uint32_t>(3),
                   dst_page0 = get_common_arg_val<uint32_t>(4);  // first destination pool page (U-page units) written
    Noc noc;
    const auto src_acc = TensorAccessor(src_args, a_src, PAGE);
    const auto dst_acc = TensorAccessor(dst_args, a_dst, PAGE);
    const auto col_acc = TensorAccessor(col_args, a_col, 64);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    noc.async_read(col_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = U * PAGE});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t c = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + U * PAGE);
    for (uint32_t p = core; p < npages; p += NCORES) {
        const uint32_t s0 = p * 8 * U + c * U;
        const uint32_t d0 = (dst_page0 + p) * U;
        for (uint32_t u = 0; u < U; ++u) {
            noc.async_read(src_acc, cb, PAGE, {.page_id = s0 + u, .offset_bytes = 0}, {.offset_bytes = u * PAGE});
        }
        noc.async_read_barrier();
        for (uint32_t u = 0; u < U; ++u) {
            noc.async_write(cb, dst_acc, PAGE, {.offset_bytes = u * PAGE}, {.page_id = d0 + u, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
}
