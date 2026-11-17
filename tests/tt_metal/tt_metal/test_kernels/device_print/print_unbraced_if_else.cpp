// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() {
    // A print is one statement, so it can be an unbraced if or else branch. DPRINT_DATA0 and DPRINT_DATA1 sit in
    // branches that aren't taken, so every RISC prints the same lines. clang-format would add the braces.
    bool taken = true;
    // clang-format off
    if (taken) DPRINT("if\n"); else DPRINT("else\n");
    if (!taken) DPRINT_DATA0("DPRINT_DATA0\n"); else DPRINT("else after DPRINT_DATA0\n");
    if (!taken) DPRINT_DATA1("DPRINT_DATA1\n"); else DPRINT("else after DPRINT_DATA1\n");
    // clang-format on
}
