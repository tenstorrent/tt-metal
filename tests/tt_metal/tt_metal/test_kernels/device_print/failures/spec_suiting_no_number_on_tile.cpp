// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() {
    // A spec that suits no number, on a tile
    DEVICE_PRINT(
        "Failure: {:s}\n", TSLICE(tt::CBIndex::c_0, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR));
}
