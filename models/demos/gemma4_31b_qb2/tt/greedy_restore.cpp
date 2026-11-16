// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// (S1, M1_INTERFACE_M5_ADDENDUM_2 A4, lane S) the greedy restore after TTSampling, in the sampling trace:
// tokens[r] = greedy_mask[r] ? argmax_ids[r] : tokens[r] for the 32 rows. ttnn.where cannot do it: its ternary op needs
// TILE tensors, and the engine's three are [1, 1, 1, 32] ROW_MAJOR replicated DRAM tensors, one 128 B page each.
// One core per chip, the pattern of resident_m5_4's table_scatter.cpp. No allocation: the tensors' DRAM addresses are
// runtime args, so the program is captured in the trace like any other.
// Runtime args: 0 greedy_mask, 1 argmax_ids, 2 tokens (DRAM addresses). CB 0: 3 x 128 B (mask, argmax ids, tokens).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page = 128;
    constexpr uint32_t rows = 32;
    const uint32_t mask_addr = get_arg_val<uint32_t>(0);
    const uint32_t argmax_addr = get_arg_val<uint32_t>(1);
    const uint32_t tokens_addr = get_arg_val<uint32_t>(2);
    auto& iface = get_local_cb_interface(0);
    const uint32_t buf = iface.fifo_limit - iface.fifo_size;
    const InterleavedAddrGen<true> mg{.bank_base_address = mask_addr, .page_size = page};
    const InterleavedAddrGen<true> ag{.bank_base_address = argmax_addr, .page_size = page};
    const InterleavedAddrGen<true> tg{.bank_base_address = tokens_addr, .page_size = page};
    noc_async_read(mg.get_noc_addr(0), buf, page);
    noc_async_read(ag.get_noc_addr(0), buf + page, page);
    noc_async_read(tg.get_noc_addr(0), buf + 2 * page, page);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    volatile tt_l1_ptr uint32_t* argmax = mask + rows;
    volatile tt_l1_ptr uint32_t* tokens = mask + 2 * rows;
    bool any = false;
    for (uint32_t r = 0; r < rows; r++) {
        if (mask[r] != 0) {
            tokens[r] = argmax[r];
            any = true;
        }
    }
    if (any) {
        noc_async_write(buf + 2 * page, tg.get_noc_addr(0), page);
        noc_async_write_barrier();
    }
}
