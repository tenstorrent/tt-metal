// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/new_dprint.h"

void kernel_main() {
    // Cannot mix indexed ({0}) and non-indexed ({}) placeholders
    NEW_DPRINT("Failure: n = {}, n = {0}\n", 42);
}
