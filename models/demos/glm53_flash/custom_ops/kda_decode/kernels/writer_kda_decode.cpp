// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the fused KDA decode block. Per work item (head h, user tile row tb): the raw q/k/v rows
// of every user as the newest conv history row (bf16 RM, 256 B each, from the untilized block), the
// updated recurrent state of every user (in place), and the head's [32, V] output block (bf16 tiles
// of the packed [B, H*V] output).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 10, cb_state_out = 11, cb_rawrm = 12;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t HK_t = get_compile_time_arg_val(5);
    constexpr uint32_t HV_t = get_compile_time_arg_val(6);
    constexpr uint32_t parts = get_compile_time_arg_val(7);
    constexpr uint32_t row_bytes = Kt * 32 * 2;      // one head's bf16 raw row (K elements)
    constexpr uint32_t out_row_bytes = Vt * 32 * 2;  // one head's bf16 output row (V elements)
    constexpr uint32_t conv_taps = 3;

    constexpr auto o_a = TensorAccessorArgs<8>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto conv_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_arg_val<uint32_t>(3);
    const uint32_t st_addr = get_arg_val<uint32_t>(4);
    const uint32_t conv_addr = get_arg_val<uint32_t>(5);
    const uint32_t conv_page_bytes = get_arg_val<uint32_t>(6);
    const uint32_t o_page_bytes = get_arg_val<uint32_t>(7);  // one RM output row [H*V] bf16

    const uint32_t t32 = get_tile_size(cb_state_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, o_page_bytes);
    const auto st_acc = TensorAccessor(st_a, st_addr, t32);
    const auto conv_acc = TensorAccessor(conv_a, conv_addr, conv_page_bytes);

    Noc noc;
    CircularBuffer out(cb_out), st(cb_state_out), rawrm(cb_rawrm);

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t n_row = (B - tb * 32u < 32u) ? (B - tb * 32u) : 32u;
        const uint32_t up = (n_row + parts - 1) / parts;
        const uint32_t u0 = part * up;
        const uint32_t u1 = (u0 + up < n_row) ? (u0 + up) : n_row;
        if (u0 >= u1) {
            continue;
        }

        // newest conv history row (row 2) of every user, per branch, from the untilized raw block
        for (uint32_t br = 0; br < 3; br++) {
            const uint32_t col_off = (br * HK_t + h * Kt) * 32 * 2;
            rawrm.wait_front(Kt);
            for (uint32_t r = u0; r < u1; r++) {
                const uint32_t page = (tb * 32 + r) * conv_taps + (conv_taps - 1);
                noc.async_write(
                    rawrm,
                    conv_acc,
                    row_bytes,
                    {.offset_bytes = r * row_bytes},
                    {.page_id = page, .offset_bytes = col_off});
            }
            noc.async_write_barrier();
            rawrm.pop_front(Kt);
        }
        // updated state of every user, in place
        for (uint32_t r = u0; r < u1; r++) {
            st.wait_front(Kt * Vt);
            const uint32_t base = ((tb * 32 + r) * H + h) * Kt * Vt;
            for (uint32_t t = 0; t < Kt * Vt; t++) {
                noc.async_write(st, st_acc, t32, {.offset_bytes = t * t32}, {.page_id = base + t});
            }
            noc.async_write_barrier();
            st.pop_front(Kt * Vt);
        }
        // output rows of this part's users: row r of the untilized block -> RM output row (tb*32 + r), head slot h
        out.wait_front(Vt);
        for (uint32_t r = u0; r < u1; r++) {
            noc.async_write(
                out,
                o_acc,
                out_row_bytes,
                {.offset_bytes = r * out_row_bytes},
                {.page_id = tb * 32 + r, .offset_bytes = h * out_row_bytes});
        }
        noc.async_write_barrier();
        out.pop_front(Vt);
    }
}
