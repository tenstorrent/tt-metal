// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"

#define ARGS(X)      \
    X(transfer_size) \
    X(transfer_count)

void kernel_main() {
    ARG_INIT(ARGS);

    for (uint32_t i = 0; i < transfer_count; i++) {
        eth_wait_for_bytes(transfer_size);
        eth_receiver_done();
    }
}
