// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "dataflow_api.h"

#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_transmission.hpp"

#include <cstdint>
#include <cstddef>

static constexpr size_t max_senders_allowed = 32;
std::array<uint32_t, max_senders_allowed> message_received_buffer;
std::array<uint64_t, max_senders_allowed> last_timestamp_recorded;

inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

void kernel_main() {
    using namespace tt::fabric;
    size_t arg_idx = 0;

    size_t host_sync_address = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* host_sync_ptr = reinterpret_cast<volatile uint32_t*>(host_sync_address);
    volatile uint64_t* timestamps_buffer_address =
        reinterpret_cast<volatile uint64_t*>(get_arg_val<uint32_t>(arg_idx++));
    size_t num_samples = get_arg_val<uint32_t>(arg_idx++);
    size_t num_senders = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* device_semaphore_addrs_ptr =
        reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx));
    arg_idx += num_senders;

    for (size_t i = 0; i < max_senders_allowed; i++) {
        message_received_buffer[i] = 1;
    }

    *host_sync_ptr = 1;
    size_t timestamp_buffer_index = 0;
    for (size_t s = 0; s < num_samples; s++) {
        size_t num_messages_received = 0;
        size_t index = 0;
        // Wait for all messages to be received from remote devices
        while (num_messages_received < num_senders) {
            if (!message_received_buffer[index] && device_semaphore_addrs_ptr[index] != 0) {
                uint64_t timestamp = get_timestamp();
                last_timestamp_recorded[index] = timestamp;
                message_received_buffer[index] = 1;
                num_messages_received++;
            }
            index++;
            if (index == num_senders) {
                index = 0;
            }
        }

        // Record the timestamps
        for (size_t i = 0; i < num_senders; i++) {
            timestamps_buffer_address[timestamp_buffer_index++] = last_timestamp_recorded[i];
        }

        // Tell the host that we have received all messages
        for (size_t i = 0; i < num_senders; i++) {
            message_received_buffer[i] = 0;
        }

        *host_sync_ptr++;
    }
}
