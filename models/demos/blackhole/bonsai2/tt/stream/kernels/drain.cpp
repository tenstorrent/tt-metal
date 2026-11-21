// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Test consumer (stream.py drain): take this receiver's next n slots from its ring and give them back.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "rx_stream.hpp"

void kernel_main() {
    // common args: ring, rcount, consumed (unused), freed array, then per-receiver slot counts by receiver index
    const uint32_t idx = reinterpret_cast<volatile uint32_t*>(RX_INFO)[0];
    const uint32_t n = get_common_arg_val<uint32_t>(4 + idx);
    RxRing rx =
        rx_open(get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(3));
    for (uint32_t i = 0; i < n; ++i) {
        rx_wait(rx);
        rx_release(rx);
    }
    rx_close(rx);
}
