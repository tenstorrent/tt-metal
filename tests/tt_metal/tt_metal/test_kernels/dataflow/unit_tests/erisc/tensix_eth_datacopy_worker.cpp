// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix DataMovement kernel for TensixEthEthTensixUniDir benchmark.
// Compile-time arg is_initiator selects between initiator (sender tensix) and echo (receiver tensix) modes.
//
// Initiator mode: fills test_data with iota pattern, writes to eth slot, signals eth push_counter,
//                 then spins on local tensix_sem for round-trip completion.
//                 Per-iteration cycle deltas are stored in an L1 timestamp buffer for host readback.
// Echo mode:      spins on local tensix_sem for incoming data, copies landing_buffer to eth slot,
//                 signals eth push_counter.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

void kernel_main() {
    constexpr bool is_initiator = get_compile_time_arg_val(0) != 0;

    uint32_t arg_idx = 0;
    const uint32_t eth_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_slot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_push_counter_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_data_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* tensix_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_sem_addr);
    // Initialize semaphore to 0 before starting
    *tensix_sem = 0;

    uint64_t eth_slot_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, eth_slot_addr);
    uint64_t eth_push_counter_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, eth_push_counter_addr);

    // Wait for eth kernel to signal that it's ready (handshake complete, push_counter initialized)
    while (*tensix_sem == 0) {
    }
    *tensix_sem = 0;

    if constexpr (is_initiator) {
        // Read sender/receiver encoding for profiler link identification
        const uint32_t sender_encoding = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t receiver_encoding = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t timestamp_buf_addr = get_arg_val<uint32_t>(arg_idx++);
        const uint64_t sender_receiver_encoding = ((uint64_t)sender_encoding << 32) | receiver_encoding;
        DeviceTimestampedData("SR_ENCODE", sender_receiver_encoding);

        // Timestamp buffer: per-iteration cycle deltas stored as uint32_t
        volatile tt_l1_ptr uint32_t* ts_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(timestamp_buf_addr);

        // Fill test_data buffer with iota pattern
        // test_data is at local_data_addr + message_size + 16 (after landing_buffer + sem + padding)
        uint32_t test_data_addr = local_data_addr + message_size + 16;
        tt_l1_ptr uint8_t* test_data = reinterpret_cast<tt_l1_ptr uint8_t*>(test_data_addr);
        for (uint32_t j = 0; j < message_size; j++) {
            test_data[j] = j;
        }

        // Warmup iteration (not timed)
        noc_async_write(test_data_addr, eth_slot_noc_addr, message_size);
        noc_semaphore_inc(eth_push_counter_noc_addr, 1);
        while (*tensix_sem == 0) {
        }
        *tensix_sem = 0;

        {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            for (uint32_t i = 0; i < num_messages - 1; i++) {
                // Read wall clock BEFORE this iteration
                uint32_t t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);

                // Write test_data to eth slot
                noc_async_write(test_data_addr, eth_slot_noc_addr, message_size);

                // Signal eth that data is ready (NOC ordering guarantees data arrives first)
                noc_semaphore_inc(eth_push_counter_noc_addr, 1);

                // Wait for round-trip completion (eth writes echo to our landing_buffer)
                while (*tensix_sem == 0) {
                }
                *tensix_sem = 0;

                // Read wall clock AFTER this iteration
                uint32_t t1 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
                ts_buf[i] = t1 - t0;
            }
        }
    } else {
        // Echo mode: landing_buffer is at local_data_addr (offset 0)
        uint32_t landing_buffer_addr = local_data_addr;
        // Wait for eth to deliver incoming data to our landing_buffer
        while (*tensix_sem == 0) {
        }
        *tensix_sem = 0;

        // Echo: copy landing_buffer to eth slot
        noc_async_write(landing_buffer_addr, eth_slot_noc_addr, message_size);

        // Signal eth that echo data is ready (NOC ordering guarantees data arrives first)
        noc_semaphore_inc(eth_push_counter_noc_addr, 1);

        for (uint32_t i = 0; i < num_messages - 1; i++) {
            // Wait for eth to deliver incoming data to our landing_buffer
            while (*tensix_sem == 0) {
            }
            *tensix_sem = 0;

            // Echo: copy landing_buffer to eth slot
            noc_async_write(landing_buffer_addr, eth_slot_noc_addr, message_size);

            // Signal eth that echo data is ready (NOC ordering guarantees data arrives first)
            noc_semaphore_inc(eth_push_counter_noc_addr, 1);
        }
    }
}
