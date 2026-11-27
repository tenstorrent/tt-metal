// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() {
    // A spec that suits no number, on a typed array
    uint32_t words[4] = {};
    DEVICE_PRINT("Failure: {:s}\n", dp_typed_array_t<4>(static_cast<uint16_t>(DataFormat::Float32), words));
}
