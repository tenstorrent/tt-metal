// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include "expf_hash.h"
int main(int argc, char** argv) {
    uint32_t step = argc > 1 ? (uint32_t)strtoul(argv[1], 0, 0) : 61;
    printf("%llu\n", (unsigned long long)x280s_expf_hash(step));
    return 0;
}
