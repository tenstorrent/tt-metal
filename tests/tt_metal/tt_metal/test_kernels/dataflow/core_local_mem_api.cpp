// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "hw/inc/compile_time_args.h"
#include "hw/inc/dataflow_api.h"
#include "hw/inc/tt-1xx/risc_common.h"

template <bool use_legacy_api>
void access_memory(uint32_t src_addr, uint32_t end_addr, uint32_t num_iterations, volatile uint64_t* results) {
    uint64_t start = c_tensix_core::read_wall_clock();
    for (uint32_t i = 0; i < num_iterations; i++) {
        if constexpr (use_legacy_api) {
            volatile uint32_t* data = reinterpret_cast<volatile uint32_t*>(src_addr);
            while (data < reinterpret_cast<volatile uint32_t*>(end_addr)) {
                [[maybe_unused]] volatile uint32_t word = data[0];
                data++;
            }
        } else {
            experimental::CoreLocalMem<std::uint32_t> mem(src_addr);
            while (mem.get_address() < end_addr) {
                [[maybe_unused]] volatile uint32_t word = mem[0];
                mem++;
            }
        }
    }
    uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - start;
    results[0] = cycles_elapsed & 0xFFFFFFFF;
    results[1] = (cycles_elapsed >> 32) & 0xFFFFFFFF;
}

void kernel_main() {
    constexpr uint32_t cycles_addr = get_compile_time_arg_val(0);
    constexpr uint32_t src_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(3);
    constexpr uint32_t neighbor_worker_core_x = get_compile_time_arg_val(4);
    constexpr uint32_t neighbor_worker_core_y = get_compile_time_arg_val(5);
    constexpr uint32_t pattern = get_compile_time_arg_val(6);

    uint32_t end_addr = src_addr + num_bytes;

    // Try reading
    access_memory<false>(src_addr, end_addr, num_iterations, &((volatile uint64_t*)(cycles_addr))[0]);
    access_memory<true>(src_addr, end_addr, num_iterations, &((volatile uint64_t*)(cycles_addr))[1]);

    // Try writing with pattern
    experimental::CoreLocalMem<std::uint32_t> mem(src_addr);
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); i++) {
        mem[i] = pattern;
    }

    // Try sending with NoC API
    experimental::Noc noc;
    experimental::UnicastEndpoint unicast_endpoint;
    noc.async_write(
        mem,
        unicast_endpoint,
        num_bytes,
        {},
        {
            .noc_x = neighbor_worker_core_x,
            .noc_y = neighbor_worker_core_y,
            .addr = src_addr,
        },
        0);
    noc.async_write_barrier();
}
