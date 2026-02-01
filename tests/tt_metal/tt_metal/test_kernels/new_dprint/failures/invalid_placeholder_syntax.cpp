// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/new_dprint.h"

void kernel_main() {
    // Invalid format string: unescaped '{' must be followed by '{', '}', or a digit
    NEW_DPRINT("{n = {0}}}\n", 42);
}
