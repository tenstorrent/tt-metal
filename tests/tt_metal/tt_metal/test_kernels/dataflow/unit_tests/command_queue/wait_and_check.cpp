// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "risc_common.h"

#include "debug/dprint.h"

void kernel_main() {
#if defined(CLEANUP)
    constexpr uint32_t dst_addr = get_compile_time_arg_val(0);

    *(tt_l1_ptr uint32_t*)dst_addr = 0;  // Clear the destination address

#elif defined(WRITER)
    constexpr uint32_t dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_noc_addr_x = get_compile_time_arg_val(1);
    constexpr uint32_t dst_noc_addr_y = get_compile_time_arg_val(2);
    riscv_wait(100'000'000);

    *(tt_l1_ptr uint32_t*)dst_addr = 1;
    noc_async_write(dst_addr, NOC_XY_ADDR(dst_noc_addr_x, dst_noc_addr_y, dst_addr), sizeof(uint32_t));
    noc_async_write_barrier();
#else
    constexpr uint32_t dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_addr2 = get_compile_time_arg_val(1);
    auto* dst_ptr = (volatile tt_l1_ptr uint32_t*)dst_addr;
    auto* dst_ptr2 = (volatile tt_l1_ptr uint32_t*)dst_addr2;

    DPRINT << "dst_addr: " << dst_addr << ", dst_addr2: " << dst_addr2 << ENDL();

    dst_ptr2[0] = *dst_ptr;

    global_program_barrier();
    dst_ptr2[1] = *dst_ptr;
    DPRINT << "dst_ptr2[0]: " << dst_ptr2[0] << ", dst_ptr2[1]: " << dst_ptr2[1] << ENDL();
#endif
}
