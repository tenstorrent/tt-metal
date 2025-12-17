// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "risc_common.h"

void kernel_main() {
    bool is_primary = get_arg_val<uint32_t>(0);
    uint32_t handshake_addr = get_arg_val<uint32_t>(1);
    uint32_t handshake_init_value = get_arg_val<uint32_t>(2);

    auto handshake_ptr = reinterpret_cast<volatile uint32_t*>(handshake_addr);

    if (is_primary) {
        // write handshake value
        handshake_ptr[0] = handshake_init_value;

        // wait for secondary ack
        uint32_t expected_value = handshake_init_value + 1;
        do {
            invalidate_l1_cache();
        } while (handshake_ptr[0] != expected_value);
    } else {
        // wait for init value
        do {
            invalidate_l1_cache();
        } while (handshake_ptr[0] != handshake_init_value);
        // write ack value
        handshake_ptr[0] = handshake_init_value + 1;
    }
}
