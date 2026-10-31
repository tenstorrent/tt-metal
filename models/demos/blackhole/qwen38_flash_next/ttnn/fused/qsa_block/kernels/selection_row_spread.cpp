// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// selection_row_spread (QWEN38_QSA_SEL_SPREAD=1, unpaged only): the same row as selection_row.cpp
// (sparse[i] = (((ids[i >> 2] << 2) + offsets[r][i]) & keep[i]) | fill[i] for i < 2048, (sentinel[i - 2048] & keep[i])
// | fill[i] after), with the 2080 columns cut into segments of whole 16-column blocks, one segment per data-movement
// RISC (the same source is built for NCRISC with SEL_HALF 0 and for BRISC with SEL_HALF 1; each RISC has its own L1
// scratch). The column loop works on 4 columns per ids word with the loads of one group issued together. Compile-time
// args: TensorAccessorArgs ids, offsets, sentinel, keep, fill, out.  Defines: SEL_HALF, SEL_MAXC (the widest segment in
// columns). Runtime args: 0-5 the addresses, 6 rows, 7 offset rows (1 or >= rows), 8 first column (a multiple of 16), 9
// columns (a multiple of 16).

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t off_addr = get_arg_val<uint32_t>(1);
    const uint32_t sentinel_addr = get_arg_val<uint32_t>(2);
    const uint32_t keep_addr = get_arg_val<uint32_t>(3);
    const uint32_t fill_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t rows = get_arg_val<uint32_t>(6);
    const uint32_t offset_rows = get_arg_val<uint32_t>(7);
    const uint32_t col0 = get_arg_val<uint32_t>(8);
    const uint32_t cols = get_arg_val<uint32_t>(9);
    constexpr uint32_t EXPANDED = 2048, TAIL = 32;
    constexpr uint32_t REGION =
        (SEL_MAXC * 4 + 64 + 63) & ~63u;  // one buffer: the data, 63 bytes of ids alignment slack
    constexpr uint32_t SENT_BYTES = 192;
    constexpr uint32_t STRIDE = 5 * REGION + SENT_BYTES;
    constexpr auto ids_args = TensorAccessorArgs<0>();
    constexpr auto off_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto sentinel_args = TensorAccessorArgs<off_args.next_compile_time_args_offset()>();
    constexpr auto keep_args = TensorAccessorArgs<sentinel_args.next_compile_time_args_offset()>();
    constexpr auto fill_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<fill_args.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto off = TensorAccessor(off_args, off_addr);
    const auto sentinel = TensorAccessor(sentinel_args, sentinel_addr);
    const auto keep = TensorAccessor(keep_args, keep_addr);
    const auto fill = TensorAccessor(fill_args, fill_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    cb_reserve_back(0, 1);
    const uint32_t base = get_write_ptr(0) + SEL_HALF * STRIDE;
    const uint32_t ids_buf = base, off_buf = base + REGION, keep_buf = base + 2 * REGION, fill_buf = base + 3 * REGION,
                   out_buf = base + 4 * REGION, sentinel_buf = base + 5 * REGION;
    const uint32_t main_cols = col0 >= EXPANDED ? 0 : (col0 + cols > EXPANDED ? EXPANDED - col0 : cols);
    const uint32_t tail_cols = cols - main_cols;
    const uint32_t ids_skew = col0 & 63u;  // ids bytes start at col0 (4 columns per word): DRAM reads are 64-aligned

    if (offset_rows == 1 && main_cols) {
        noc_async_read(off.get_noc_addr(0, col0 * 4), off_buf, main_cols * 4);
    }
    if (tail_cols) {
        noc_async_read(sentinel.get_noc_addr(0, 0), sentinel_buf, TAIL * 4);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        FUSED_ZONE("fz_qs_sel_row");
        if (main_cols) {
            if (offset_rows != 1) {
                noc_async_read(off.get_noc_addr(r, col0 * 4), off_buf, main_cols * 4);
            }
            noc_async_read(ids.get_noc_addr(r, col0 - ids_skew), ids_buf, (main_cols >> 2) * 4 + ids_skew);
        }
        noc_async_read(keep.get_noc_addr(r, col0 * 4), keep_buf, cols * 4);
        noc_async_read(fill.get_noc_addr(r, col0 * 4), fill_buf, cols * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* ids_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_buf + ids_skew);
        volatile tt_l1_ptr uint32_t* off_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(off_buf);
        volatile tt_l1_ptr uint32_t* keep_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(keep_buf);
        volatile tt_l1_ptr uint32_t* fill_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fill_buf);
        volatile tt_l1_ptr uint32_t* out_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_buf);
        volatile tt_l1_ptr uint32_t* sentinel_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sentinel_buf);
        for (uint32_t i = 0; i < main_cols; i += 4) {
            const uint32_t id4 = ids_l1[i >> 2] << 2;
            const uint32_t o0 = off_l1[i], o1 = off_l1[i + 1], o2 = off_l1[i + 2], o3 = off_l1[i + 3];
            const uint32_t k0 = keep_l1[i], k1 = keep_l1[i + 1], k2 = keep_l1[i + 2], k3 = keep_l1[i + 3];
            const uint32_t f0 = fill_l1[i], f1 = fill_l1[i + 1], f2 = fill_l1[i + 2], f3 = fill_l1[i + 3];
            out_l1[i] = ((id4 + o0) & k0) | f0;
            out_l1[i + 1] = ((id4 + o1) & k1) | f1;
            out_l1[i + 2] = ((id4 + o2) & k2) | f2;
            out_l1[i + 3] = ((id4 + o3) & k3) | f3;
        }
        for (uint32_t i = main_cols; i < cols; i += 4) {
            const uint32_t s = i + col0 - EXPANDED;
            const uint32_t s0 = sentinel_l1[s], s1 = sentinel_l1[s + 1], s2 = sentinel_l1[s + 2],
                           s3 = sentinel_l1[s + 3];
            const uint32_t k0 = keep_l1[i], k1 = keep_l1[i + 1], k2 = keep_l1[i + 2], k3 = keep_l1[i + 3];
            const uint32_t f0 = fill_l1[i], f1 = fill_l1[i + 1], f2 = fill_l1[i + 2], f3 = fill_l1[i + 3];
            out_l1[i] = (s0 & k0) | f0;
            out_l1[i + 1] = (s1 & k1) | f1;
            out_l1[i + 2] = (s2 & k2) | f2;
            out_l1[i + 3] = (s3 & k3) | f3;
        }
        noc_async_write(out_buf, out.get_noc_addr(r, col0 * 4), cols * 4);
        noc_async_write_barrier();
    }
}
