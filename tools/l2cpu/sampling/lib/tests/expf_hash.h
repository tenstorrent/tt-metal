// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* Shared by the host program tests/expf_hash.c and the QEMU image: a hash of x280s_expf over
 * every `step`-th fp32 bit pattern (covers NaN, inf, subnormals, the whole range). */
#ifndef X280S_EXPF_HASH_H
#define X280S_EXPF_HASH_H
#include <stdint.h>
#include "../x280s.h"

static inline uint64_t x280s_expf_hash(uint32_t step) {
    uint64_t h = 0xcbf29ce484222325ull;
    uint64_t b = 0;
    for (; b < 0x100000000ull; b += step) {
        union {
            uint32_t u;
            float f;
        } in, out;
        in.u = (uint32_t)b;
        out.f = x280s_expf(in.f);
        h = (h ^ out.u) * 0x100000001b3ull;
    }
    return h;
}
#endif
