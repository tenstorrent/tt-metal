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

__attribute__((noinline)) void do_nothing(uint32_t i) {
#pragma GCC unroll 1
    for (volatile uint32_t i = 0; i < 2000; ++i) {
        __asm__ volatile("nop");
    }
    invalidate_l1_cache();
}

void kernel_main() {
    volatile uint32_t msg_count = 0;
    DPRINT << "Kernel started " << HEX() << buffer_base << ENDL();
    auto buffer_ptr = reinterpret_cast<volatile uint32_t*>(buffer_base);
    auto respond_ptr = reinterpret_cast<volatile uint32_t*>(buffer_base + 4);
    auto status_ptr = reinterpret_cast<volatile uint32_t*>(buffer_base + 8);
    auto arg_ptr = reinterpret_cast<volatile uint32_t*>(arg_base);
    // post a heartbeat to the same address for the host api to read
    auto heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(heartbeat);
    volatile uint32_t* const debug_dump = reinterpret_cast<volatile uint32_t*>(debug_dump_addr);
    uint32_t heartbeat_cnt = 0;

    uint32_t last_seen = 0;

    debug_dump[1] = 0xbbbb;
    while(true) {
        invalidate_l1_cache();
        volatile uint32_t heartbeat = 0xabcd0000 | heartbeat_cnt;
        heartbeat_cnt++;
        heartbeat_cnt &= 0xFFFF;
        heartbeat_ptr[0] = heartbeat;

        // wait for a message
        invalidate_l1_cache();
        volatile uint32_t val = buffer_ptr[0];
        if (val == 0xdeadbeef) {
            continue;
        }
        volatile uint32_t upper = (val & 0xffff0000) >> 16;
        volatile uint32_t lower = val & 0xffff;
        
       if (upper != lower) {
            invalidate_l1_cache();
            respond_ptr[0] = val;
            status_ptr[0] = 1;
            while (true) {
                do_nothing(0);
            }
       } else {
        if (upper == last_seen && upper != 0) {
            invalidate_l1_cache();
            respond_ptr[0] = val;
            status_ptr[0] = 2;
            while (true) {
                do_nothing(0);
            }
        }
        invalidate_l1_cache();
        last_seen = upper;
        buffer_ptr[0] = 0xdeadbeef;
       }
    }
}
