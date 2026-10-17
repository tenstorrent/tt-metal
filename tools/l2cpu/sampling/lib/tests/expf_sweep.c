// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* Host-only exhaustive accuracy sweep of x280s_expf against double-precision exp (libm).
 * Covers every fp32 x in [-104, 88.75] plus the special values. Prints max ulp error. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../x280s.h"

static float fb(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint32_t bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

int main(void) {
    double max_ulp = 0;
    float worst = 0;
    uint64_t n = 0, exact = 0, over1 = 0;
    for (uint64_t b = 0; b < 0x100000000ull; b++) {
        float x = fb((uint32_t)b);
        if (!(x >= -104.0f && x <= 88.75f)) {
            continue;
        }
        float y = x280s_expf(x);
        double ref = exp((double)x);
        float rn = (float)ref;
        n++;
        if (bf(y) == bf(rn)) {
            exact++;
            continue;
        }
        /* ulp of the correctly rounded result (subnormal-aware) */
        double ulp;
        if (rn == 0.0f || fabsf(rn) < 1.1754943508222875e-38f) {
            ulp = ldexp(1.0, -149);
        } else {
            ulp = (double)nextafterf(rn, INFINITY) - (double)rn;
        }
        if (isinf(rn)) {
            if (!isinf(y)) {
                printf("overflow mismatch x=%a y=%a\n", x, y);
            }
            continue;
        }
        double e = fabs((double)y - ref) / ulp;
        if (e > 1.0) {
            over1++;
        }
        if (e > max_ulp) {
            max_ulp = e;
            worst = x;
        }
    }
    printf(
        "swept %llu values, correctly rounded %llu (%.4f%%), >1ulp %llu, max err %.4f ulp at x=%a (%.9g)\n",
        (unsigned long long)n,
        (unsigned long long)exact,
        100.0 * exact / n,
        (unsigned long long)over1,
        max_ulp,
        worst,
        worst);
    int bad = 0;
    bad |= !(isnan(x280s_expf(NAN)));
    bad |= !(x280s_expf(INFINITY) == INFINITY);
    bad |= !(x280s_expf(-INFINITY) == 0.0f);
    bad |= !(x280s_expf(0.0f) == 1.0f) || !(x280s_expf(-0.0f) == 1.0f);
    bad |= !(x280s_expf(-1000.0f) == 0.0f) || !(x280s_expf(1000.0f) == INFINITY);
    printf("specials %s\n", bad ? "FAIL" : "ok");
    return (bad || max_ulp > 2.0) ? 1 : 0;
}
