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
    auto arg_ptr = reinterpret_cast<volatile uint32_t*>(arg_base);
    // post a heartbeat to the same address for the host api to read
    auto heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(heartbeat);
    volatile uint32_t* const debug_dump = reinterpret_cast<volatile uint32_t*>(debug_dump_addr);
    uint32_t heartbeat_cnt = 0;

    debug_dump[1] = 0xbbbb;
    for (uint32_t i = 1; i < num_writes; ++i) {
        // wait for a message
        invalidate_l1_cache();
        volatile uint32_t val = buffer_ptr[i];
        volatile uint32_t msg_status = val & 0xffff0000;
        volatile uint32_t msg_type = val & 0xffff;
        // uint32_t j = 0;
        while (msg_status != 0xca110000) {
            invalidate_l1_cache();
            volatile uint32_t heartbeat = 0xabcd0000 | heartbeat_cnt;
            heartbeat_cnt++;
            heartbeat_cnt &= 0xFFFF;
            heartbeat_ptr[0] = heartbeat;
            val = buffer_ptr[i];
            msg_status = val & 0xffff0000;
            msg_type = val & 0xffff;
            uint32_t non_volatile_val = buffer_ptr[i];
            debug_dump[1] = 0xaaaa;
            debug_dump[2] = non_volatile_val;
            debug_dump[3] = buffer_ptr[i];
            // debug_dump[3] = j++;
            // yes. sometimes misaligned. just a check.
            noc_async_write((uint32_t)&buffer_ptr[i], get_noc_addr(other_x, other_y, 0x20000), 4);
            noc_async_write_barrier();
        }

        // change it to done
        debug_dump[1] = 0xcccc;
        buffer_ptr[i] = 0xd0e50000 | msg_type;
        debug_dump[1] = 0xdddd;

        // if the message type is not in sync then hang
        if (msg_type != i) {
            debug_dump[1] = 0xeeee;
            while (true) {
                __asm__ volatile("nop");
            }
        }
    }
}
