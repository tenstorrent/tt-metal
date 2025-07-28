// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compile_time_args.h"
#include <cstdint>
#include "debug/dprint.h"

constexpr uint32_t num_writes = get_compile_time_arg_val(0);
constexpr uint32_t buffer_base = get_compile_time_arg_val(1);
constexpr uint32_t arg_base = get_compile_time_arg_val(2);
constexpr uint32_t heartbeat = get_compile_time_arg_val(3);
constexpr uint32_t debug_dump_addr = get_compile_time_arg_val(4);

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
    auto heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(heartbeat);
    volatile uint32_t* const debug_dump = reinterpret_cast<volatile uint32_t*>(debug_dump_addr);

    uint32_t heartbeat_cnt = 0;
    debug_dump[6] = 0;

    // poll for msg. when there is a message pretend to process it and clear
    invalidate_l1_cache();
    volatile uint32_t local_buffer_value = 0;
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
            invalidate_l1_cache();
            volatile uint32_t val = buffer_ptr[i];
            volatile uint32_t msg_status = val & 0xffff0000;
            volatile uint32_t msg_type = val & 0xffff;
            volatile uint32_t arg_val = arg_ptr[0];
            if (msg_status == 0xca110000) {
                invalidate_l1_cache();
                msg_count++;
                local_buffer_value = buffer_ptr[i];
                buffer_ptr[i] = 0xd0e50000 | msg_type;
                if (buffer_ptr[i] != (0xd0e50000 | msg_type)) {
                    debug_dump[1] = 0xfffffff2;
                    debug_dump[2] = buffer_ptr[i];
                    debug_dump[3] = msg_type;
                    debug_dump[4] = msg_count;
                    debug_dump[5] = arg_val;
                    debug_dump[6]++;
                    debug_dump[7] = local_buffer_value;
                    while (true) {
                        __asm__ volatile("nop");
                    }
                }
                arg_ptr[0] = 0;
                do_nothing(msg_type);

                if (msg_type != msg_count || arg_val != msg_count) {
                    debug_dump[1] = 0xfffffff1;
                    debug_dump[2] = buffer_ptr[i];
                    debug_dump[3] = msg_type;
                    debug_dump[4] = msg_count;
                    debug_dump[5] = arg_val;
                    debug_dump[6]++;
                    debug_dump[7] = local_buffer_value;
                    // expect the correct message because this was the wrong one
                    msg_count--;
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
