// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Indexer local top-512 post-processing (tt/fuse_attn.py idx_local), per query row c of [C, 512] ids (local slots of
// this mesh column, 0xFFFFFFFF = fewer visible):
//   valid        = id < LC
//   vals[c][k]   = valid ? score[c][id] : -inf            (bf16)   <- gather of the score row, ttnn.gather + where
//   pg[c][k]     = valid ? id * COLS + cid : 0xFFFFFFFF   (u32)    <- global slot = local * cols + column (cid: fp32
//   [1,1,1,1] tile of this column)
// replacing the typecast / to_layout / lt / where / typecast / gather / multiply / add / typecast chain of
// DSV41PrefillIndexer.select_single. The score row (W bf16) is read whole into L1, the 512 values are picked in L1;
// rows are dealt round-robin over the cores with the next row prefetched.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_buf = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);     // score row elements (Lc + 32)
    constexpr uint32_t LC = get_compile_time_arg_val(2);    // valid id limit
    constexpr uint32_t COLS = get_compile_time_arg_val(3);  // mesh columns
    constexpr uint32_t TOPK = 512;
    constexpr uint32_t ROWB = ((W * 2 + 63) / 64) * 64;  // score row bytes, 64 B aligned slot
    constexpr uint32_t IDB = TOPK * 4;
    constexpr uint32_t SLOT = ROWB + IDB;
    constexpr uint32_t VB = TOPK * 2, PB = TOPK * 4;
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto i_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t stride = get_arg_val<uint32_t>(1);
    const uint32_t nrows = get_arg_val<uint32_t>(2);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(4);

    Noc noc;
    const auto s_acc = TensorAccessor(s_args, s_addr, W * 2);
    const auto i_acc = TensorAccessor(i_args, i_addr, IDB);
    const auto v_acc = TensorAccessor(v_args, v_addr, VB);
    const auto p_acc = TensorAccessor(p_args, p_addr, PB);
    const auto c_acc = TensorAccessor(c_args, c_addr, 4096);
    experimental::CB cb(cb_buf);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    // layout: [slot0 | slot1 | cid tile (4096 B) | vals0 | pg0 | vals1 | pg1]
    const uint32_t cid_off = 2 * SLOT;
    const uint32_t out_off = cid_off + 4096;
    noc.async_read(c_acc, cb, 4096, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = cid_off});
    if (nrows > 0) {
        noc.async_read(s_acc, cb, W * 2, {.page_id = first, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(i_acc, cb, IDB, {.page_id = first, .offset_bytes = 0}, {.offset_bytes = ROWB});
    }
    noc.async_read_barrier();
    const float cidf = *reinterpret_cast<volatile tt_l1_ptr float*>(base + cid_off);
    const uint32_t cid = (uint32_t)cidf;
    for (uint32_t j = 0; j < nrows; ++j) {
        const uint32_t g = first + j * stride;
        const uint32_t cur = (j & 1) * SLOT;
        const uint32_t oo = out_off + (j & 1) * (VB + PB);
        if (j + 1 < nrows) {  // prefetch the next row into the other slot
            const uint32_t nx = first + (j + 1) * stride;
            const uint32_t no = ((j + 1) & 1) * SLOT;
            noc.async_read(s_acc, cb, W * 2, {.page_id = nx, .offset_bytes = 0}, {.offset_bytes = no});
            noc.async_read(i_acc, cb, IDB, {.page_id = nx, .offset_bytes = 0}, {.offset_bytes = no + ROWB});
        }
        if (j >= 2) {
            noc.async_write_barrier();  // the writes of row j-2 left this output slot
        }
        volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + cur);
        volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + cur + ROWB);
        volatile tt_l1_ptr uint16_t* vo = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + oo);
        volatile tt_l1_ptr uint32_t* po = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + oo + VB);
        for (uint32_t k = 0; k < TOPK; ++k) {
            const uint32_t id = ids[k];
            if (id < LC) {
                vo[k] = row[id];
                po[k] = id * COLS + cid;
            } else {
                vo[k] = 0xFF80;
                po[k] = 0xFFFFFFFFu;
            }
        }
        noc.async_write(cb, v_acc, VB, {.offset_bytes = oo}, {.page_id = g, .offset_bytes = 0});
        noc.async_write(cb, p_acc, PB, {.offset_bytes = oo + VB}, {.page_id = g, .offset_bytes = 0});
        noc.async_read_barrier();  // the prefetch of row j+1
    }
    noc.async_write_barrier();
}
