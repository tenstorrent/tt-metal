// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 integ: in-place raw row write into a pool:  dst[ids[i]] = src[i] for i in [0, N)   (ids = 0xFFFFFFFF: skip).
//   src  ROW_MAJOR [.., W] (rows are pages of PAGE bytes: the 64 B aligned stride, 704 for SCALED_FP8 656-byte rows)
//   ids  uint32 ROW_MAJOR page of IDP bytes (element i = destination row of source row i on THIS device; sharded per device)
//   dst  same dtype / layout as src (the pool)
// One core. The decode append of one token per user: the owner column of the token passes its local row, every other column 0xFFFFFFFF.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t PAGE = get_compile_time_arg_val(2);
    constexpr uint32_t IDP = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto ids_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    const uint32_t a_src = get_common_arg_val<uint32_t>(0), a_ids = get_common_arg_val<uint32_t>(1),
                   a_dst = get_common_arg_val<uint32_t>(2);
    Noc noc;
    const auto src_acc = TensorAccessor(src_args, a_src, PAGE);
    const auto ids_acc = TensorAccessor(ids_args, a_ids, IDP);
    const auto dst_acc = TensorAccessor(dst_args, a_dst, PAGE);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    noc.async_read(ids_acc, cb, IDP, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t d = ids[i];
        if (d != 0xFFFFFFFFu) {
            noc.async_read(src_acc, cb, PAGE, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = IDP});
            noc.async_read_barrier();
            noc.async_write(cb, dst_acc, PAGE, {.offset_bytes = IDP}, {.page_id = d, .offset_bytes = 0});
            noc.async_write_barrier();
        }
    }
}
