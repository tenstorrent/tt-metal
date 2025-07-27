// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compile_time_args.h"
#include <cstdint>
#include "debug/dprint.h"

constexpr uint32_t num_writes = get_compile_time_arg_val(0);
constexpr uint32_t buffer_base = get_compile_time_arg_val(1);
constexpr uint32_t arg_base = get_compile_time_arg_val(2);

__attribute__((noinline)) void do_nothing(uint32_t i) {
#pragma GCC unroll 1
    for (volatile uint32_t i = 0; i < 2000; ++i) {
        __asm__ volatile("nop");
    }
    invalidate_l1_cache();
}

void kernel_main() {
    volatile uint32_t msg_count = 0;
    DPRINT << "Kernel started" << ENDL();
    auto buffer_ptr = reinterpret_cast<volatile uint32_t*>(buffer_base);
    auto arg_ptr = reinterpret_cast<volatile uint32_t*>(arg_base);
    // post a heartbeat to the same address for the host api to read
    auto heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(0x7CC70);
    volatile uint32_t* const debug_dump_addr = reinterpret_cast<volatile uint32_t*>(0x36b0);

    uint32_t heartbeat_cnt = 0;

    // poll for msg. when there is a message pretend to process it and clear
    invalidate_l1_cache();
    while (true) {
        volatile uint32_t heartbeat = 0xabcd0000 | heartbeat_cnt;
        heartbeat_cnt++;
        heartbeat_cnt &= 0xFFFF;
        heartbeat_ptr[0] = heartbeat;

        invalidate_l1_cache();
#pragma GCC unroll 1
        for (volatile uint32_t i = 0; i < 2000; ++i) {
            __asm__ volatile("nop");
        }
        for (uint32_t i = 0; i < 4; ++i) {
            volatile uint32_t val = buffer_ptr[i];
            volatile uint32_t msg_status = val & 0xffff0000;
            volatile uint32_t msg_type = val & 0xffff;
            if (msg_status == 0xca110000) {
                invalidate_l1_cache();
                msg_count++;
                buffer_ptr[i] = 0xd0e50000 | msg_type;
                // arg_ptr[0] = 0;
                do_nothing(msg_type);

                if (msg_type != msg_count || arg_ptr[0] != msg_count) {
                    debug_dump_addr[1] = 0xfffffff1;
                    debug_dump_addr[2] = buffer_ptr[i];
                    debug_dump_addr[3] = msg_type;
                    debug_dump_addr[4] = msg_count;
                    debug_dump_addr[5] = arg_ptr[0];
                    while (true) {
                        __asm__ volatile("nop");
                    }
                }
            } else if (msg_status == 0xdead0000) {
                return;
            } else {
                DPRINT << "No message   " << HEX() << msg_status << ENDL();
            }
        }
    }
}
