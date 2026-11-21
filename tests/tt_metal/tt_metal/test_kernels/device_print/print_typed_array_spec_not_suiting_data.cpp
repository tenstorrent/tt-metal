// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() {
    // {:.1f} suits Float16_b values, so they print with it; {:x} doesn't, so they print as {} would, after a warning
    // line.
    uint32_t words[2] = {};
    DPRINT("[{:.1f}]\n", dp_typed_array_t<2>(static_cast<uint16_t>(DataFormat::Float16_b), words));
    DPRINT("[{:x}]\n", dp_typed_array_t<2>(static_cast<uint16_t>(DataFormat::Float16_b), words));
}
