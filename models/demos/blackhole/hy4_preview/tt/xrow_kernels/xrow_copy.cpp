// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B45x: page range copy src -> dst (interleaved DRAM tensors, PAGE bytes per page), for the cross-row chain copy of the
// Hy4 pool.
//   dst page (dst_page0 + i) = src page (src_page0 + off + i), i < npages
// off and skip come from this device's control page (uint32 words: [0] = off in pages, [1] = skip != 0 => copy nothing
// on this device). Stage-in (pool -> stage, every row): off 0. Stage-out (gathered block of the donor row -> pool): off
// = donor_row * pages-per-block, skip on every row but the receiving one. Pages move through L1 in batches of B pages
// per core; pages i = core*B + k*NCORES*B ...
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr uint32_t NCORES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto ctl_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t a_src = get_common_arg_val<uint32_t>(0), a_dst = get_common_arg_val<uint32_t>(1),
                   a_ctl = get_common_arg_val<uint32_t>(2);
    uint32_t npages = get_common_arg_val<uint32_t>(3);
    const uint32_t src_page0 = get_common_arg_val<uint32_t>(4);
    const uint32_t dst_page0 = get_common_arg_val<uint32_t>(5);
    Noc noc;
    const auto src_acc = TensorAccessor(src_args, a_src, PAGE);
    const auto dst_acc = TensorAccessor(dst_args, a_dst, PAGE);
    const auto ctl_acc = TensorAccessor(ctl_args, a_ctl, 64);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    noc.async_read(ctl_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B * PAGE});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t off = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B * PAGE);
    const uint32_t skip = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B * PAGE + 4);
    if (skip != 0) {
        npages = 0;
    }
    for (uint32_t p = core * B; p < npages; p += NCORES * B) {
        const uint32_t n = (npages - p) < B ? (npages - p) : B;
        for (uint32_t u = 0; u < n; ++u) {
            noc.async_read(
                src_acc, cb, PAGE, {.page_id = src_page0 + off + p + u, .offset_bytes = 0}, {.offset_bytes = u * PAGE});
        }
        noc.async_read_barrier();
        for (uint32_t u = 0; u < n; ++u) {
            noc.async_write(
                cb, dst_acc, PAGE, {.offset_bytes = u * PAGE}, {.page_id = dst_page0 + p + u, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
}
