// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "htif.h"

int main(void) {
    unsigned long sum = 0;
    for (unsigned long i = 1; i <= 100; ++i) {
        sum += i * i;
    }
    /* 100*101*201/6 = 338350 */
    htif_puts("sum_sq_1_100=");
    htif_put_u64_dec(sum);
    htif_puts("\n");
    if (sum != 338350UL) {
        htif_puts("FAIL arith\n");
        return 1;
    }
    htif_puts("PASS arith\n");
    return 0;
}
