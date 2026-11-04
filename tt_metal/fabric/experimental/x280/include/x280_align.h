// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * X280 socket-layout and PCIe transfer alignment.
 */
#ifndef X280_ALIGN_H
#define X280_ALIGN_H

#include <stdint.h>

/* Blackhole L1 socket sections and PCIe payloads are both 16-byte aligned. */
#define X280_SOCKET_ALIGNMENT 16u
#define X280_PCIE_ALIGNMENT X280_SOCKET_ALIGNMENT

static inline uint32_t x280_align_up(uint32_t value, uint32_t alignment) {
    /* All callers pass a power-of-two alignment. */
    return (value + (alignment - 1u)) & ~(alignment - 1u);
}

#endif /* X280_ALIGN_H */
