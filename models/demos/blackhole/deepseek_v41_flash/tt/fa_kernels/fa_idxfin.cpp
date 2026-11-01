// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Indexer merge post-processing (tt/fuse_attn.py idx_final), per query row c:  fin[c][k] = pos[c][k] < NCAND ?
// cand[c][pos[c][k]] : 0xFFFFFFFF (pos: the top-512 positions of the merged [C, cols*512] values, cand: the
// all-gathered global slots) in place of typecast / to_layout / clamp / typecast / concat(skip column) / to_layout /
// gather.  Rows are dealt round-robin; all reads of a core are issued first, then compute, then all writes.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_buf = get_compile_time_arg_val(0);
    constexpr uint32_t NCAND = get_compile_time_arg_val(1);
    constexpr uint32_t MAXR = get_compile_time_arg_val(2);
    constexpr uint32_t TOPK = 512;
    constexpr uint32_t CB = NCAND * 4, PB = TOPK * 4;
    constexpr uint32_t SLOT = CB + 2 * PB;  // cand row | pos row | out row
    constexpr auto p_args = TensorAccessorArgs<3>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t stride = get_arg_val<uint32_t>(1);
    const uint32_t nrows = get_arg_val<uint32_t>(2);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto p_acc = TensorAccessor(p_args, p_addr, PB);
    const auto c_acc = TensorAccessor(c_args, c_addr, CB);
    const auto o_acc = TensorAccessor(o_args, o_addr, PB);
    experimental::CB cb(cb_buf);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t j = 0; j < nrows; ++j) {
        const uint32_t g = first + j * stride;
        noc.async_read(c_acc, cb, CB, {.page_id = g, .offset_bytes = 0}, {.offset_bytes = j * SLOT});
        noc.async_read(p_acc, cb, PB, {.page_id = g, .offset_bytes = 0}, {.offset_bytes = j * SLOT + CB});
    }
    noc.async_read_barrier();
    for (uint32_t j = 0; j < nrows; ++j) {
        volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + j * SLOT);
        volatile tt_l1_ptr uint32_t* pos = cand + NCAND;
        volatile tt_l1_ptr uint32_t* out = pos + TOPK;
        for (uint32_t k = 0; k < TOPK; ++k) {
            const uint32_t p = pos[k];
            out[k] = p < NCAND ? cand[p] : 0xFFFFFFFFu;
        }
    }
    for (uint32_t j = 0; j < nrows; ++j) {
        const uint32_t g = first + j * stride;
        noc.async_write(cb, o_acc, PB, {.offset_bytes = j * SLOT + CB + PB}, {.page_id = g, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
