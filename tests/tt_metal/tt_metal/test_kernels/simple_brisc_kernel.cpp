// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug_print.h"

void kernel_main() {
    DPRINT << 'Q' << 'X' << 'Y' << 'Z' << ENDL();
    for (volatile int i = 0; i < 1000000; i++);
}
