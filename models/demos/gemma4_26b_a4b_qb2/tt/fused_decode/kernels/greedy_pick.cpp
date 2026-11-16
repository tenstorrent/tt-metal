// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 greedy terminal, pick (BRISC, one core) (fused_decode/greedy.py: GreedyB1).
//
// Reads the gathered per-device (key, class, global id) (elements 0..2 of row 0 of each of the `devices` UINT32
// tiles of `gathered`, L1), keeps the largest key (ties: the lowest id) and writes the token as element 0 of the
// [1, 1, 1, 32] UINT32 row-major token tensor (one 128 B DRAM page; elements 1..31 are zero).
//
// runtime args: [gathered_addr, tokens_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gathered_addr = get_arg_val<uint32_t>(0);
    const uint32_t tokens_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t devices = get_compile_time_arg_val(1);
    constexpr auto g_args = TensorAccessorArgs<2>();
    constexpr auto t_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto s_g = TensorAccessor(g_args, gathered_addr, 4096);
    const auto s_t = TensorAccessor(t_args, tokens_addr, 128);

    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t d = 0; d < devices; ++d) {
        noc_async_read(s_g.get_noc_addr(d, 0), scr + 16 * d, 16);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    // max key, then the lowest id among the devices in its softcap class
    uint32_t best_key = s[0], low = s[1];
    for (uint32_t d = 1; d < devices; ++d) {
        if (s[4 * d] > best_key) {
            best_key = s[4 * d];
            low = s[4 * d + 1];
        }
    }
    uint32_t best_id = 0xFFFFFFFF;
    for (uint32_t d = 0; d < devices; ++d) {
        if (s[4 * d + 1] == low && s[4 * d + 2] < best_id) {
            best_id = s[4 * d + 2];
        }
    }
    const uint32_t row = scr + 128;  // 64 B aligned
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
    t[0] = best_id;
    for (uint32_t i = 1; i < 32; ++i) {
        t[i] = 0;
    }
    noc_async_write(row, s_t.get_noc_addr(0), 128);
    noc_async_write_barrier();
}
