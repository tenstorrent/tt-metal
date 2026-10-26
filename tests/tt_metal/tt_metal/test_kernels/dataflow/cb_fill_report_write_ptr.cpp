// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Busy-waits `delay_cycles`, then fills every page of circular buffer `cb_index` with
// `pattern + word index` and reports the CB's write pointer to `report_addr`. The host learns which
// L1 this launch's CB config pointed the kernel at, and can check the data landed there and nowhere
// else; a non-zero delay keeps the launch in flight for a known time.

#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

void kernel_main() {
    constexpr uint32_t cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    const uint32_t report_addr = get_arg_val<uint32_t>(0);
    const uint32_t pattern = get_arg_val<uint32_t>(1);
    const uint32_t delay_cycles = get_arg_val<uint32_t>(2);

    riscv_wait(delay_cycles);

    cb_reserve_back(cb_index, num_pages);
    const uint32_t write_ptr = get_write_ptr(cb_index);
    auto* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_ptr);
    constexpr uint32_t num_words = num_pages * page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; i++) {
        data[i] = pattern + i;
    }
    cb_push_back(cb_index, num_pages);

    auto* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = write_ptr;
}
