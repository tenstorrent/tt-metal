// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per token-tile r: the coefficients -- HMODE: the pre tile (raw or lane layout); EXPAND: the raw comb and post tiles
// (page r each) into CB_RAW (the compute kernel builds the x8 tiles); otherwise the n comb x8 tiles (pages r*n + j) and
// the post x8 tile (page r) into CB_COEF. Per position p = r * Ct + c: the n stream tiles x[r, i, c] (page (r*n + i)*Ct
// + c) into CB_X and y[r, c] (page r*Ct + c, fp32 or bf16) into CB_Y.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_pos = get_arg_val<uint32_t>(4);
    const uint32_t num_pos = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_raw = get_compile_time_arg_val(2);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t Ct = get_compile_time_arg_val(5);
    constexpr bool expand = get_compile_time_arg_val(6) != 0;
    constexpr bool hmode = get_compile_time_arg_val(7) != 0;  // pre-mix: the coefficient tile is `post` (= pre), no y
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    const uint32_t x_page = get_local_cb_interface(cb_x).fifo_page_size;
    const uint32_t y_page = get_local_cb_interface(cb_y).fifo_page_size;
    const uint32_t coef_page = get_local_cb_interface(expand ? cb_raw : cb_coef).fifo_page_size;

    const auto s_x = TensorAccessor(x_args, x_addr);
    const auto s_y = TensorAccessor(y_args, y_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);

    Noc noc;
    CircularBuffer cbx(cb_x), cby(cb_y), cbraw(cb_raw), cbcoef(cb_coef);

    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t p = start_pos; p < start_pos + num_pos; ++p) {
        const uint32_t r = p / Ct;
        const uint32_t c = p - r * Ct;

        if (r != cur_r) {
            if constexpr (hmode) {
                // the pre tile of token-tile r: raw -> CB_RAW (expanded by the compute kernel) or lane layout ->
                // CB_COEF
                CircularBuffer& cbc = expand ? cbraw : cbcoef;
                cbc.reserve_back(1);
                noc.async_read(s_post, cbc, coef_page, {.page_id = r}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cbc.push_back(1);
            } else if constexpr (expand) {
                // CB_RAW holds exactly one set: the reserve waits for the compute kernel to have consumed the last
                cbraw.reserve_back(2);
                noc.async_read(s_comb, cbraw, coef_page, {.page_id = r}, {.offset_bytes = 0});
                noc.async_read(s_post, cbraw, coef_page, {.page_id = r}, {.offset_bytes = coef_page});
                noc.async_read_barrier();
                cbraw.push_back(2);
            } else {
                cbcoef.reserve_back(n + 1);
                for (uint32_t j = 0; j < n; ++j) {
                    noc.async_read(s_comb, cbcoef, coef_page, {.page_id = r * n + j}, {.offset_bytes = j * coef_page});
                }
                noc.async_read(s_post, cbcoef, coef_page, {.page_id = r}, {.offset_bytes = n * coef_page});
                noc.async_read_barrier();
                cbcoef.push_back(n + 1);
            }
            cur_r = r;
        }

        cbx.reserve_back(n);
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_read(s_x, cbx, x_page, {.page_id = (r * n + i) * Ct + c}, {.offset_bytes = i * x_page});
        }
        if constexpr (hmode) {
            noc.async_read_barrier();
            cbx.push_back(n);
            continue;
        }
        cby.reserve_back(1);
        noc.async_read(s_y, cby, y_page, {.page_id = r * Ct + c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbx.push_back(n);
        cby.push_back(1);
    }
}
