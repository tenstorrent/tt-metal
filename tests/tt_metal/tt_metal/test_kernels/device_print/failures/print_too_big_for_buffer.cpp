// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // A print too big for its print buffer: 1024 words don't fit in any core's print buffer
    uint32_t words[1024] = {};
    DEVICE_PRINT("Failure: {}\n", dp_typed_array_t<1024>(static_cast<uint16_t>(DataFormat::Float32), words));
}
