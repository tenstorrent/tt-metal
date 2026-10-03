// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* Host-only: the library must give identical results whatever MXCSR the caller left behind
 * (FTZ, DAZ, round-toward-zero / up), and must restore the caller's MXCSR. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <xmmintrin.h>
#include "../x280s.h"

#define V 8192
static float row[V];
static x280s_work_t work;

static uint64_t lcg(uint64_t* s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s >> 11;
}

int main(void) {
    uint64_t s = 1;
    for (int i = 0; i < V; i++) {
        row[i] = (float)((int64_t)(lcg(&s) % 20000) - 10000) * 0.01f - 80.0f;
    }
    row[5] = 1e-39f; /* a subnormal logit */
    const unsigned modes[] = {
        0x1f80,
        0x1f80 | 0x8040 /* FTZ|DAZ */,
        0x1f80 | 0x6000 /* RZ */,
        0x1f80 | 0x4000 /* RU */,
        0x1f80 | 0x8040 | 0x2000 /* FTZ|DAZ|RD */};
    x280s_params_t p = {0.25f, 0, 0.97f, 0, 42};
    x280s_stats_t ref, st;
    int fails = 0;
    for (uint64_t step = 0; step < 50; step++) {
        _mm_setcsr(0x1f80);
        int32_t t0 = x280s_sample_row(row, X280S_DTYPE_F32, V, 1, &p, 3, step, &work, &ref);
        for (unsigned m = 1; m < sizeof modes / sizeof modes[0]; m++) {
            _mm_setcsr(modes[m]);
            int32_t t = x280s_sample_row(row, X280S_DTYPE_F32, V, 1, &p, 3, step, &work, &st);
            unsigned after = _mm_getcsr();
            _mm_setcsr(0x1f80);
            if (t != t0 || memcmp(&st, &ref, sizeof st) != 0 || (after & ~0x3fu) != (modes[m] & ~0x3fu)) {
                printf("FAIL step %llu mode %#x\n", (unsigned long long)step, modes[m]);
                fails++;
            }
        }
    }
    printf("fpenv test %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
