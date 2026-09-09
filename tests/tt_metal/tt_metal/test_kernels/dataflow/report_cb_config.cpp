// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "internal/circular_buffer_init.h"

void kernel_main() {
    constexpr uint32_t cb_index = get_compile_time_arg_val(0);
    const uint32_t report_addr = get_arg_val<uint32_t>(0);
    const LocalCBInterface& cb = get_local_cb_interface(cb_index);
    auto* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = cb.fifo_page_size;
    report[1] = cb.fifo_num_pages;
}
