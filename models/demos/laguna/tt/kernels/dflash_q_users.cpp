// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched DFlash draft (Laguna; dflash_q_users.py): moves query rows between the head-split layout [1, NH, U * Q, 128]
// (head h, user u's Q rows at rows u * Q ..) and the per-user attention layout [U, NKV, RP, 128] (user u, KV head g:
// rows (h % G) * Q + i for its G = NH / NKV query heads h, then zero padding to RP rows).
//   mode 0 (gather):  out = per-user layout from in = head-split layout (padding rows zero)
//   mode 1 (scatter): out = head-split layout from in = per-user layout
// Q divides 16, so a user's Q rows sit inside one face row band of a tile on both sides: each (user, head, column
// tile, face half) is one Q * 32-byte copy. Core c handles output tiles c * per .. c * per + per - 1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    constexpr uint32_t U = get_compile_time_arg_val(2);
    constexpr uint32_t NH = get_compile_time_arg_val(3);
    constexpr uint32_t NKV = get_compile_time_arg_val(4);
    constexpr uint32_t Q = get_compile_time_arg_val(5);
    constexpr uint32_t RP = get_compile_time_arg_val(6);  // padded rows per (user, KV head) in the per-user layout
    constexpr uint32_t per = get_compile_time_arg_val(7);  // output tiles per core
    constexpr auto in_args = TensorAccessorArgs<8>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t G = NH / NKV, HDT = 4, S = U * Q;
    constexpr uint32_t ST = S / 32, RPT = RP / 32;
    constexpr uint32_t n_out = mode == 0 ? U * NKV * RPT * HDT : NH * ST * HDT;
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), 2048);
    const uint32_t buf = get_write_ptr(0);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = core * per; o < core * per + per && o < n_out; ++o) {
        for (uint32_t off = 0; off < 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, buf + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        const uint32_t c = o % HDT;
        // the output tile's 32 / Q row groups: each group is Q rows of one (user, head)
        for (uint32_t grp = 0; grp < 32 / Q; ++grp) {
            uint32_t src_tile, src_row, dst_row = grp * Q;
            if constexpr (mode == 0) {
                // out tile o = ((u * NKV + g) * RPT + rt) * HDT + c; out row rt * 32 + dst_row = (h % G) * Q + i
                const uint32_t rt = (o / HDT) % RPT, g = (o / HDT / RPT) % NKV, u = o / HDT / RPT / NKV;
                const uint32_t ro = rt * 32 + dst_row;
                if (ro >= G * Q) {
                    continue;  // padding rows stay zero
                }
                const uint32_t h = g * G + ro / Q, s = u * Q;
                src_tile = (h * ST + s / 32) * HDT + c;
                src_row = s % 32;
            } else {
                // out tile o = (h * ST + st) * HDT + c; out rows st * 32 + dst_row .. = user u's Q rows of head h
                const uint32_t st = (o / HDT) % ST, h = o / HDT / ST;
                const uint32_t u = (st * 32 + dst_row) / Q, g = h / G, ro = (h % G) * Q;
                src_tile = ((u * NKV + g) * RPT + ro / 32) * HDT + c;
                src_row = ro % 32;
            }
            const uint64_t src = in.get_noc_addr(src_tile);
            const uint32_t so = (src_row / 16) * 1024 + (src_row % 16) * 32;
            const uint32_t dofs = (dst_row / 16) * 1024 + (dst_row % 16) * 32;
            noc_async_read(src + so, buf + dofs, Q * 32);
            noc_async_read(src + so + 512, buf + dofs + 512, Q * 32);
        }
        noc_async_read_barrier();
        noc_async_write(buf, out.get_noc_addr(o), 2048);
        noc_async_write_barrier();
    }
}
