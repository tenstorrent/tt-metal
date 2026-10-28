// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the V4.1 query head layout (heads_q_compute.cpp): the row-major [1, H, S, D] output, one page per (head,
// token) row. Per unit (tile row r, head h) its 32 rows: the untilized no-rope block (NT * 64 bytes per row) and
// the rotated tail block (RT * 64 bytes) at byte NT * 64 of the row.
//
// With OUT_H > H the output holds OUT_H heads (``q_heads(out_heads=)``: sparse_sdpa takes 32
// heads at least and the decode step's local heads are 16): unit (r, h) with h < OUT_H - H also writes the 32 rows
// of head H + h as zeros, from one page of cb_zero the kernel clears once.
//
// compile_time_args = [NT, RT, H, S, OUT_H, TensorAccessorArgs(out)]
// runtime args = [out_addr, unit_start, unit_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_H = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr uint32_t cb_rm_nope = 16, cb_rm_tail = 17, cb_zero = 18;
    constexpr uint32_t NOPE_ROW = NT * 32 * 2, TAIL_ROW = RT * 32 * 2;  // bf16 bytes
    constexpr uint32_t ROW = NOPE_ROW + TAIL_ROW;
    static_assert(OUT_H >= H, "out_heads >= heads");
    constexpr uint32_t PAD = OUT_H - H;  // the zero heads after the real ones

    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    DataflowBuffer nope(cb_rm_nope);
    DataflowBuffer tail(cb_rm_tail);
    uint32_t zero_l1 = 0;
    if constexpr (PAD > 0) {
        DataflowBuffer zero(cb_zero);
        zero_l1 = zero.get_write_ptr();
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
        for (uint32_t i = 0; i < ROW / 4; ++i) {
            z[i] = 0;
        }
    }
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t r = u / H;
        const uint32_t h = u - r * H;
        const uint32_t first = h * S + r * 32;
        nope.wait_front(NT);
        tail.wait_front(RT);
        for (uint32_t i = 0; i < 32; ++i) {
            noc.async_write(nope, out, NOPE_ROW, {.offset_bytes = i * NOPE_ROW}, {.page_id = first + i});
            noc.async_write(
                tail, out, TAIL_ROW, {.offset_bytes = i * TAIL_ROW}, {.page_id = first + i, .offset_bytes = NOPE_ROW});
        }
        if constexpr (PAD > 0) {
            if (h < PAD) {
                const uint32_t pad_first = (H + h) * S + r * 32;
                for (uint32_t i = 0; i < 32; ++i) {
                    noc_async_write(zero_l1, out.get_noc_addr(pad_first + i, 0), ROW);
                }
            }
        }
        noc.async_writes_flushed();
        nope.pop_front(NT);
        tail.pop_front(RT);
    }
    noc.async_write_barrier();
}
