// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

// Receiver worker kernel - waits for signal from sender that all data has been received
void kernel_main() {
    // Runtime args
    // For global semaphore, we get the address directly (not a semaphore ID)
    volatile uint32_t* const completion_semaphore_address =
        reinterpret_cast<volatile uint32_t* const>(get_arg_val<uint32_t>(0));
    const uint32_t expected_num_signals = get_arg_val<uint32_t>(1);

    DPRINT << "receiver_signal_wait: Starting. Waiting for " << expected_num_signals
           << " signals at global semaphore addr: " << (uint32_t)completion_semaphore_address << "\n";

    // Wait for the sender to signal completion
    noc_semaphore_wait(completion_semaphore_address, expected_num_signals);

    DPRINT << "receiver_signal_wait: Received completion signal. All data has been received.\n";

    // Reset the global semaphore to 0 before exit for potential reuse
    *completion_semaphore_address = 0;

    DPRINT << "receiver_signal_wait: Reset global semaphore to 0. Exiting.\n";
}
