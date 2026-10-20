// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_UBENCH_REGION_H
#define X280_UBENCH_REGION_H

#include <stdint.h>

/* An address and a byte count. Used at two scales: the window a kernel may
 * touch, and each allocation carved from it. Wire-visible, so C-compatible and
 * free of implicit padding. */
typedef struct ubench_region {
    uint64_t addr;
    uint64_t bytes;
#if defined(__cplusplus)
    bool valid() const { return bytes != 0u; }
    uint64_t end() const { return addr + bytes; }
#endif
} ubench_region_t;

#endif
