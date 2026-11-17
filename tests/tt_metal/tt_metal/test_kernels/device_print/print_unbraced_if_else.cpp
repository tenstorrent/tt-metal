// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() {
    // A print is one statement, so it can be an unbraced if or else branch. After the first line, each print macro sits
    // in an if branch that isn't taken, with a DPRINT in its else, so a macro that captured the else fails the test,
    // and every RISC prints the same lines whichever RISCs the macro prints on. clang-format would add the braces.
    bool taken = true;
    // clang-format off
    if (taken) DPRINT("if\n"); else DPRINT("else\n");
    if (!taken) DPRINT("DPRINT\n"); else DPRINT("else after DPRINT\n");
    if (!taken) DPRINT_DATA0("DPRINT_DATA0\n"); else DPRINT("else after DPRINT_DATA0\n");
    if (!taken) DPRINT_DATA1("DPRINT_DATA1\n"); else DPRINT("else after DPRINT_DATA1\n");
    if (!taken) DPRINT_UNPACK("DPRINT_UNPACK\n"); else DPRINT("else after DPRINT_UNPACK\n");
    if (!taken) DPRINT_MATH("DPRINT_MATH\n"); else DPRINT("else after DPRINT_MATH\n");
    if (!taken) DPRINT_PACK("DPRINT_PACK\n"); else DPRINT("else after DPRINT_PACK\n");
    // clang-format on
}
