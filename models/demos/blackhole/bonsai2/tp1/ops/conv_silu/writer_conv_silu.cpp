// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for conv_silu (GDN milestone 1), one core per value head.
//
//   conv  [1,1,C]  this head's q|k|v channel tiles after conv+SiLU (from cb_work)
//   shift [K,1,C]  the ring for the next token: rows 1..K-1 of the old state, then the new input
//
// async_write takes an L1 handle (the CB) plus a byte offset, not a raw pointer, so the source is
// cb + .offset_bytes. The ring shift is done here rather than with a host slot index: a slot written
// from the host would be baked into a decode trace and every replay would reuse it.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_qkv = 0, cb_state = 1, cb_diag = 2, cb_work = 3, cb_shift = 4;

void kernel_main() {
    constexpr uint32_t kd_t = get_compile_time_arg_val(0);
    constexpr uint32_t v_t = get_compile_time_arg_val(1);
    constexpr uint32_t c_t = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t per_core = get_compile_time_arg_val(4);

    constexpr auto conv_a = TensorAccessorArgs<5>();
    constexpr auto shift_a = TensorAccessorArgs<conv_a.next_compile_time_args_offset()>();

    const uint32_t conv_addr = get_arg_val<uint32_t>(0);
    const uint32_t shift_addr = get_arg_val<uint32_t>(1);
    const uint32_t nv = get_arg_val<uint32_t>(2);
    const uint32_t gqa = get_arg_val<uint32_t>(3);
    const uint32_t hv = get_arg_val<uint32_t>(6);

    const uint32_t qk_h = hv / gqa;
    const uint32_t nk = nv / gqa;
    const uint32_t q_col = qk_h * kd_t;
    const uint32_t k_col = nk * kd_t + qk_h * kd_t;
    const uint32_t v_col = 2 * nk * kd_t + hv * v_t;

    const uint32_t tb = get_tile_size(cb_work);
    const auto cv = TensorAccessor(conv_a, conv_addr, tb);
    const auto sh = TensorAccessor(shift_a, shift_addr, tb);

    Noc noc;
    CircularBuffer work(cb_work);
    CircularBuffer shift(cb_shift);

    // ---- conv + silu row ----------------------------------------------------
    // cb_work arrives one group at a time (compute pushes a group, we pop it), which is what breaks
    // the reader/compute/writer cycle.
    const auto write_group = [&](uint32_t dst_first_channel_tile, uint32_t n) {
        work.wait_front(n);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_write(
                work, cv, tb, {.offset_bytes = i * tb}, {.page_id = dst_first_channel_tile + i});
        }
        noc.async_writes_flushed();
        work.pop_front(n);
    };
    write_group(q_col, kd_t);
    write_group(k_col, kd_t);
    write_group(v_col, v_t);

    // ---- the shifted ring ---------------------------------------------------
    // row j <- old state row j+1 for j < K-1; row K-1 <- the new input row.
    shift.wait_front(K * per_core);
    for (uint32_t j = 0; j < K; j++) {
        const uint32_t dst_base = j * c_t;
        const uint32_t src_row0 = j * per_core;
        CircularBuffer& src = shift;
        const auto& acc_dst = sh;
        for (uint32_t i = 0; i < kd_t; i++) {
            noc.async_write(src, acc_dst, tb, {.offset_bytes = (src_row0 + i) * tb}, {.page_id = dst_base + q_col + i});
        }
        for (uint32_t i = 0; i < kd_t; i++) {
            noc.async_write(
                src, acc_dst, tb, {.offset_bytes = (src_row0 + kd_t + i) * tb}, {.page_id = dst_base + k_col + i});
        }
        for (uint32_t i = 0; i < v_t; i++) {
            noc.async_write(
                src, acc_dst, tb, {.offset_bytes = (src_row0 + 2 * kd_t + i) * tb}, {.page_id = dst_base + v_col + i});
        }
    }
    noc.async_write_barrier();
    shift.pop_front(K * per_core);
}
