// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Gate core of the GDN step split 4 (QWEN38_GDN_CONV_SPLIT=4), reader: reads the a and b tiles of the projection and
// the dt_bias / neg_exp_A tiles of head h in one batch, moves element (lane, head) of the a and b tiles to element (0,
// 0) (as reader_vb3.cpp) and publishes CB_AB, CB_DTNA.  Compile-time args: 0 HEADS, 1 A_TILE, 2 B_TILE; then
// TensorAccessorArgs of projected, dtna.  Runtime args: 0 projected, 1 dtna addresses, 2 lane, 3 head.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_AB = 4, CB_DTNA = 5;
constexpr uint32_t HEADS = get_compile_time_arg_val(0);
constexpr uint32_t A_TILE = get_compile_time_arg_val(1), B_TILE = get_compile_time_arg_val(2);
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096, FACE_ELEMS = 256;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_ELEMS + (row & 15) * 16 + (col & 15);
}
}  // namespace

void kernel_main() {
    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(1);
    const uint32_t lane = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    constexpr auto p_args = TensorAccessorArgs<3>();
    constexpr auto dtna_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto p = TensorAccessor(p_args, p_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);

    FUSED_ZONE("fz_gv_g_r");
    cb_reserve_back(CB_AB, 2);
    const uint32_t ab_l1 = get_write_ptr(CB_AB);
    noc_async_read_page(A_TILE, p, ab_l1);
    noc_async_read_page(B_TILE, p, ab_l1 + BF16_TILE);
    cb_reserve_back(CB_DTNA, 2);
    const uint32_t dt_l1 = get_write_ptr(CB_DTNA);
    noc_async_read_page(head, dtna, dt_l1);
    noc_async_read_page(HEADS + head, dtna, dt_l1 + FP32_TILE);
    noc_async_read_barrier();
    {
        volatile tt_l1_ptr uint16_t* a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1);
        volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1 + BF16_TILE);
        const uint32_t src = face_element(lane, head);
        a[0] = a[src];
        b[0] = b[src];
    }
    cb_push_back(CB_AB, 2);
    cb_push_back(CB_DTNA, 2);
}
