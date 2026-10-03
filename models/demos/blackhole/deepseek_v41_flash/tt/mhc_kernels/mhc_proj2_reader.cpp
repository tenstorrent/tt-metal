// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC projection v2, reader.  Core (g, r) owns the column-tile range [r*TPJ, (r+1)*TPJ) of ALL four streams and the
// token group g (TG tokens).  It gathers, for every column tile j of its range, one tile Xj with row 4*tl + i = x[tok0
// + tl, stream i, 32 columns] (rows 0..3 of the stream-row tile of token tok0 + tl, copied with two 256 B reads) and
// the 4*TPJ weight tiles.

#include <stdint.h>
#include "tools/profiler/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32)
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t TG = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);   // 32-col tiles per stream row (D/32)
    constexpr uint32_t TPJ = get_compile_time_arg_val(4);  // column tiles per core
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t tok0 = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cbx(cb_x), cbw(cb_w);

    DeviceZoneScopedN("PR_ALL");
    cbx.reserve_back(TPJ);
    noc.async_write_zeros(cbx, TPJ * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        DeviceZoneScopedN("PR_zerodone");
    }
    for (uint32_t j = 0; j < TPJ; ++j) {
        for (uint32_t tl = 0; tl < TG; ++tl) {
            const uint32_t page = (tok0 + tl) * NT + r * TPJ + j;
            noc.async_read(
                x_acc, cbx, 256, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = j * TILE + RO(4 * tl)});
            noc.async_read(
                x_acc,
                cbx,
                256,
                {.page_id = page, .offset_bytes = 1024},
                {.offset_bytes = j * TILE + 1024 + RO(4 * tl)});
        }
    }
    {
        DeviceZoneScopedN("PR_Xissued");
    }
    cbw.reserve_back(4 * TPJ);
    for (uint32_t n = 0; n < 4 * TPJ; ++n) {
        noc.async_read(w_acc, cbw, TILE, {.page_id = r * 4 * TPJ + n, .offset_bytes = 0}, {.offset_bytes = n * TILE});
    }
    {
        DeviceZoneScopedN("PR_Wissued");
    }
    noc.async_read_barrier();
    cbx.push_back(TPJ);
    cbw.push_back(4 * TPJ);
}
