// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "hw/inc/api/compile_time_args.h"
#include "hw/inc/api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "hw/inc/internal/tt-1xx/risc_common.h"

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

    // Try using a more complex type
    struct TestStruct {
        uint32_t foo;
        uint64_t bar;
    };

    experimental::CoreLocalMem<TestStruct> struct_mem(src_addr);
    struct_mem->foo = pattern;
    struct_mem->bar = pattern + 1;
    while (struct_mem->foo != pattern) {
    }
    while (struct_mem->bar != pattern + 1) {
    }

    // Try writing with operator[]
    experimental::CoreLocalMem<std::uint32_t> mem(src_addr);
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); i++) {
        mem[i] = pattern + i;
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

    // Try clear data here before reading from neighbor core
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); i++) {
        mem[i] = 0;
    }

    // Try reading from neighbor core
    noc.async_read(
        unicast_endpoint,
        mem,
        num_bytes,
        {
            .noc_x = neighbor_worker_core_x,
            .noc_y = neighbor_worker_core_y,
            .addr = src_addr,
        },
        {});
    noc.async_read_barrier();

    // Try reading with operator[]
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); i++) {
        if (mem[i] != pattern + i) {
            while (true) {
            }
        }
    }

    // Pointer arithmetic (hangs if incorrect)
    while (mem[0] != pattern) {
    }
    while (mem[4] != pattern + 4) {
    }
    while (mem[8] != pattern + 8) {
    }

    // Add offset to pointer
    uint32_t mid_index = num_bytes / sizeof(uint32_t) / 2;
    auto mid = mem + mid_index;

    // get_address
    auto mid_addr = mid.get_address();
    if (mid_addr != src_addr + mid_index * sizeof(uint32_t)) {
        while (true) {
        }
    }

    // get_unsafe_ptr
    auto unsafe_ptr = mid.get_unsafe_ptr();
    if (reinterpret_cast<uintptr_t>(unsafe_ptr) != src_addr + mid_index * sizeof(uint32_t)) {
        while (true) {
        }
    }

    uint32_t middle_value = pattern + mid_index;
    while (mid[0] != middle_value) {
    }
    while (mid[4] != middle_value + 4) {
    }
    while (mid[8] != middle_value + 8) {
    }

    // Subtract two pointers
    auto diff = mid - mem;
    if ((uint32_t)diff != mid_index) {
        while (true) {
        }
    }

    // Increment and decrement operators
    auto inc_mem = mem;
    inc_mem++;
    while (inc_mem[0] != pattern + 1) {
    }
    while (inc_mem[4] != pattern + 5) {
    }
    while (inc_mem[8] != pattern + 9) {
    }
    auto dec_mem = mid;
    dec_mem--;
    while (dec_mem[0] != middle_value - 1) {
    }
    while (dec_mem[4] != middle_value + 3) {
    }
    while (dec_mem[8] != middle_value + 7) {
    }

    --inc_mem;
    while (inc_mem[0] != pattern) {
    }
    while (inc_mem[4] != pattern + 4) {
    }
    while (inc_mem[8] != pattern + 8) {
    }
    ++dec_mem;
    while (dec_mem[0] != middle_value) {
    }
    while (dec_mem[4] != middle_value + 4) {
    }
    while (dec_mem[8] != middle_value + 8) {
    }

    // Copy constructor
    auto copy_mem = mem;
    if (copy_mem != mem) {
        while (true) {
        }
    }
    while (copy_mem[0] != pattern) {
    }
    while (copy_mem[4] != pattern + 4) {
    }
    while (copy_mem[8] != pattern + 8) {
    }

    // Copy assignment operator
    auto copy_assign_mem = mem;
    copy_assign_mem = mem;
    if (copy_assign_mem != mem) {
        while (true) {
        }
    }
    while (copy_assign_mem[0] != pattern) {
    }
    while (copy_assign_mem[4] != pattern + 4) {
    }
    while (copy_assign_mem[8] != pattern + 8) {
    }
}
