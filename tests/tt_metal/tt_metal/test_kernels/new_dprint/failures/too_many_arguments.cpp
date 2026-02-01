// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/new_dprint.h"

void kernel_main() {
    // All arguments must be referenced when using indexed placeholders
    NEW_DPRINT("Unreferenced: n = {}, m = {}\n", 42, 5, 10);
}
