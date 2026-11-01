// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused prefill attention glue (tt/fuse_attn.py), reader.  One BLOCK = one tile row (32 token rows) x one head (16
// column tiles = 512 dims). Block b -> tile row tr_g = b / NH, head h = b % NH, user u = tr_g / CB32, tile row of the
// user tr_l = tr_g % CB32.
//   IN tile mode : the 16 tiles of the block are read from a tile tensor, page = u*IN_U + h*IN_H + tr_l*IN_TR + IN_COL0
//   + j. IN rm mode   : the 32 rows (1 KiB each) of head h, rows tr_l*32 .. +31 of the user's own [1,32,C,512]
//   row-major buffer
//                  (page h*C + tr_l*32 + r) are read contiguously into 16 tile-sized pages (the compute kernel tilizes
//                  them).
// Besides, per block: the cos / sin table tiles of columns 448..511 (2 + 2 tiles, page tabrow*16 + 14 + jj) and, once,
// the pair-swap matrix tile P (page 0 of the [1,1,64,64] tile tensor).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_c = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_p = get_compile_time_arg_val(3);
    constexpr uint32_t IN_RM = get_compile_time_arg_val(4);
    constexpr uint32_t NH = get_compile_time_arg_val(5);
    constexpr uint32_t CB32 = get_compile_time_arg_val(6);  // C / 32
    constexpr uint32_t C = get_compile_time_arg_val(7);
    constexpr uint32_t TAB_PER_USER = get_compile_time_arg_val(8);
    constexpr uint32_t IN_U = get_compile_time_arg_val(9);
    constexpr uint32_t IN_H = get_compile_time_arg_val(10);
    constexpr uint32_t IN_TR = get_compile_time_arg_val(11);
    constexpr uint32_t IN_COL0 = get_compile_time_arg_val(12);
    constexpr uint32_t NU = get_compile_time_arg_val(13);  // users (number of RM input buffers)
    constexpr uint32_t NT = get_compile_time_arg_val(14);  // tiles per head block
    constexpr auto in_args = TensorAccessorArgs<15>();
    constexpr auto c_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const uint32_t b0 = get_arg_val<uint32_t>(0);
    const uint32_t nb = get_arg_val<uint32_t>(1);
    // common args: [c, s, p, in_0 .. in_{NU-1}]
    const uint32_t c_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto c_acc = TensorAccessor(c_args, c_addr, 2048);
    const auto s_acc = TensorAccessor(s_args, s_addr, 2048);
    const auto p_acc = TensorAccessor(p_args, p_addr, 2048);
    experimental::CB cbin(cb_in), cbc(cb_c), cbs(cb_s), cbp(cb_p);

    cbp.reserve_back(1);
    noc.async_read(p_acc, cbp, 2048, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cbp.push_back(1);

    for (uint32_t b = b0; b < b0 + nb; ++b) {
        const uint32_t tr_g = b / NH;
        const uint32_t h = b - tr_g * NH;
        const uint32_t u = tr_g / CB32;
        const uint32_t tr_l = tr_g - u * CB32;
        const uint32_t tabrow = TAB_PER_USER ? tr_g : tr_l;

        cbc.reserve_back(2);
        cbs.reserve_back(2);
        cbin.reserve_back(NT);
        for (uint32_t jj = 0; jj < 2; ++jj) {
            const uint32_t tp = tabrow * 16 + 14 + jj;
            noc.async_read(c_acc, cbc, 2048, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = jj * 2048});
            noc.async_read(s_acc, cbs, 2048, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = jj * 2048});
        }
        if constexpr (IN_RM) {
            const uint32_t in_addr = get_common_arg_val<uint32_t>(3 + u);
            const auto in_acc = TensorAccessor(in_args, in_addr, 1024);
            const uint32_t row0 = h * C + tr_l * 32;
            for (uint32_t r = 0; r < 32; ++r) {
                noc.async_read(
                    in_acc, cbin, 1024, {.page_id = row0 + r, .offset_bytes = 0}, {.offset_bytes = r * 1024});
            }
        } else {
            const uint32_t in_addr = get_common_arg_val<uint32_t>(3);
            const auto in_acc = TensorAccessor(in_args, in_addr, 2048);
            const uint32_t base = u * IN_U + h * IN_H + tr_l * IN_TR + IN_COL0;
            for (uint32_t j = 0; j < NT; ++j) {
                noc.async_read(
                    in_acc, cbin, 2048, {.page_id = base + j, .offset_bytes = 0}, {.offset_bytes = j * 2048});
            }
        }
        noc.async_read_barrier();
        cbc.push_back(2);
        cbs.push_back(2);
        cbin.push_back(NT);
    }
}
