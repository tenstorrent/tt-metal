// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/new_dprint.h"

void kernel_main() {
    // Placeholder index exceeds number of arguments
    NEW_DPRINT("Bad index: n = {1}\n", 42);
}
