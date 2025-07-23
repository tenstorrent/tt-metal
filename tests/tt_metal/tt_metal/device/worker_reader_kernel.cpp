// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compile_time_args.h"
#include <cstdint>
#include "debug/dprint.h"
#include "dataflow_api.h"

constexpr uint32_t num_writes = get_compile_time_arg_val(0);
constexpr uint32_t buffer_base = get_compile_time_arg_val(1);
constexpr uint32_t arg_base = get_compile_time_arg_val(2);
constexpr uint32_t heartbeat = get_compile_time_arg_val(3);
constexpr uint32_t debug_dump_addr = get_compile_time_arg_val(4);
constexpr uint32_t other_x = get_compile_time_arg_val(5);
constexpr uint32_t other_y = get_compile_time_arg_val(6);

void kernel_main() {
    while (true) {
        noc_async_read(buffer_base, buffer_base + sizeof(uint32_t) * num_writes, sizeof(uint32_t) * num_writes);
        noc_async_read_barrier();
    }
}
