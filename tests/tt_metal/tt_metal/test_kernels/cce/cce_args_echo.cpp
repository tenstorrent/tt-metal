// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "dev_mem_map.h"
#include "risc_common.h"

void kernel_main() {
    constexpr uint32_t echo_base_offset = get_compile_time_arg_val(0);
    constexpr uint32_t ct_marker = get_compile_time_arg_val(1);
    constexpr uint32_t echo_magic = get_compile_time_arg_val(2);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_runtime_args = get_compile_time_arg_val(4);
    constexpr uint32_t num_common_args = get_compile_time_arg_val(5);

    uint32_t hart_id;
    asm volatile("csrr %0, mhartid" : "=r"(hart_id));

    const uint64_t slot_address = static_cast<uint64_t>(echo_base_offset) +
                                  static_cast<uint64_t>(MEM_L1_UNCACHED_BASE) +
                                  static_cast<uint64_t>(hart_id) * slot_bytes;
    volatile uint32_t* slot = reinterpret_cast<volatile uint32_t*>(slot_address);

    slot[1] = hart_id;
    slot[2] = ct_marker;
    for (uint32_t i = 0; i < num_runtime_args; i++) {
        slot[3 + i] = get_arg_val<uint32_t>(i);
    }
    for (uint32_t i = 0; i < num_common_args; i++) {
        slot[3 + num_runtime_args + i] = get_common_arg_val<uint32_t>(i);
    }
    asm volatile("fence rw, rw" ::: "memory");
    slot[0] = echo_magic;
}
