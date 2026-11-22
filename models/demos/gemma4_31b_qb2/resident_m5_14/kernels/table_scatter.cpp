// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// (M5 A5) row-granular write_tables: one core copies n staged slots' table rows into the engine's tables, in DRAM.
// The host uploads the staging rows ([kStage, 6, P] int32, ROW_MAJOR: one page of P words per (staged slot, group))
// and an index row (word 0: n, words 1..n: the slots). This program runs eagerly on CQ 0, so it lands before the next
// launch's trace and after the previous one (no host sync).
// Compile-time args: 0 groups (6), 1 page bytes (P * 4). Runtime args: 0 staging, 1 index, 2 tables (DRAM addresses).
// CB 0: one page of page bytes (the row in flight) + 128 B (the index row), in that order.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t groups = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    const uint32_t stage_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t tables_addr = get_arg_val<uint32_t>(2);
    auto& iface = get_local_cb_interface(0);
    const uint32_t buf = iface.fifo_limit - iface.fifo_size;
    const uint32_t idx_l1 = buf + page_bytes;
    const InterleavedAddrGen<true> ig{.bank_base_address = idx_addr, .page_size = 128};
    noc_async_read(ig.get_noc_addr(0), idx_l1, 128);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    const InterleavedAddrGen<true> sg{.bank_base_address = stage_addr, .page_size = page_bytes};
    const InterleavedAddrGen<true> tg{.bank_base_address = tables_addr, .page_size = page_bytes};
    const uint32_t n = idx[0];
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t slot = idx[1 + i];
        for (uint32_t g = 0; g < groups; g++) {
            noc_async_read(sg.get_noc_addr(i * groups + g), buf, page_bytes);
            noc_async_read_barrier();
            noc_async_write(buf, tg.get_noc_addr(slot * groups + g), page_bytes);
            noc_async_write_barrier();
        }
    }
}
