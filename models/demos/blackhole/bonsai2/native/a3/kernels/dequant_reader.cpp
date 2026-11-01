// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill dequant, reader (NCRISC). Reads A3's on-device ternary weights as they are:
//   codes:  bfp2 tiles, DRAM width-sharded over the banks, each bank's shard column-major
//           (column j's K-stick of kt tiles at offset j*kt*w_tile_bytes);
//   scales: fp32 32x32 tiles in a3.scale_rows layout, same sharding (column j's s_tiles at
//           offset j*s_tiles*4096; super-group G lives in tile G/4, rows (G%4)*8..+7).
// One work unit = (column c, super-group G) = 32 code tiles (8 groups of 128) + the one
// scale tile that holds their 8 scale rows.
// Runtime args: w_addr, s_addr, u0 (first unit), nu (units on this core). Unit u: c = u % ncols, G = u / ncols
// (column fastest, so a core's consecutive units hit different DRAM banks).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);
    constexpr uint32_t ng = get_compile_time_arg_val(0);       // super-groups per column (kt / 32)
    constexpr uint32_t s_tiles = get_compile_time_arg_val(1);  // scale tiles per column
    constexpr uint32_t cpb = get_compile_time_arg_val(2);      // weight columns per DRAM bank
    constexpr uint32_t wtb = get_compile_time_arg_val(3);      // bytes per code tile
    constexpr uint32_t ncols = get_compile_time_arg_val(4);    // real weight columns
    constexpr uint32_t cb_w = 0, cb_s = 1;
    constexpr uint32_t kt = ng * 32;

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u % ncols, G = u / ncols;
        const uint32_t bank = c / cpb, j = c % cpb;
        cb_reserve_back(cb_s, 1);
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, s_addr + (j * s_tiles + G / 4) * 4096),
                       get_write_ptr(cb_s), 4096);
        cb_reserve_back(cb_w, 32);
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr + (j * kt + G * 32) * wtb), get_write_ptr(cb_w),
                       32 * wtb);
        noc_async_read_barrier();
        cb_push_back(cb_s, 1);
        cb_push_back(cb_w, 32);
    }
}
